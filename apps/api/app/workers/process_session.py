"""Synchronous structured-memory processing with complete transcript preservation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from app.db import repositories
from app.db.client import db_client
from app.models.memory import ConsentLevel, MemoryFragment
from app.services.memory_chunking import StoryUnit, build_story_units
from app.services.memory_extractor import ExtractedMemory, MemoryExtractorService
from app.services.transcription_service import TranscriptionService
from app.workers.index_memory import index_memories

logger = logging.getLogger(__name__)
INGESTION_FORMAT = "structured-story-v2"


def _transcript_from_segments(segments: list[dict]) -> str:
    return "\n".join(str(segment.get("text") or "").strip() for segment in segments if segment.get("text")).strip()


def _fallback_search_document(unit: StoryUnit) -> str:
    return (
        f"Title: {unit.title}. Category: {unit.category}. Summary: {unit.summary}. "
        f"Keywords: {', '.join(unit.keywords)}. Source evidence: {unit.content}"
    )


def _memory_from_story(
    unit: StoryUnit,
    enriched: ExtractedMemory | None,
    *,
    session_id: str,
    subject_id: str,
    index: int,
) -> MemoryFragment:
    """Create a durable structured memory while retaining its original story."""
    semantic = dict(enriched.semantic_metadata) if enriched else {}
    extracted_category = str(semantic.get("category") or "")
    category = unit.category if extracted_category in {"", "Stories"} and unit.category != "Stories" else (extracted_category or unit.category)
    tags = list(dict.fromkeys([
        category.lower(), *unit.keywords,
        *[str(value) for value in semantic.get("keywords", []) if value],
    ]))[:32]
    metadata = {
        "source": "session_transcript",
        "ingestion_format": INGESTION_FORMAT,
        "title": str(semantic.get("title") or unit.title),
        "summary": str(semantic.get("summary") or unit.summary),
        "category": category,
        "important_facts": semantic.get("important_facts", []),
        "user_preferences": semantic.get("user_preferences", []),
        "people": semantic.get("people", []),
        "places": semantic.get("places", []),
        "objects": semantic.get("objects", []),
        "time_reference": semantic.get("time_reference"),
        "keywords": tags,
        "tags": tags,
        "importance_score": float(semantic.get("importance_score") or unit.importance_score),
        "importance_level": semantic.get("importance_level") or (
            "critical" if unit.importance_score >= 0.95 else "high" if unit.importance_score >= 0.8 else "medium"
        ),
        "related_memory_ids": [],
    }
    topics = semantic.get("topics") if isinstance(semantic.get("topics"), list) else []
    people = semantic.get("people") if isinstance(semantic.get("people"), list) else []
    import uuid
    memory_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"session:{session_id}:memory:{index}"))
    return MemoryFragment(
        id=memory_id, session_id=session_id, subject_id=subject_id,
        content=unit.content,
        emotion_tags=(enriched.emotion_tags if enriched else ["reflective"]),
        topics=[str(topic) for topic in topics] or [category.lower()],
        people_mentioned=[str(person) for person in people],
        consent_level=ConsentLevel.PRIVATE,
        confidence_score=(enriched.confidence_score if enriched else 0.8),
        created_at=datetime.now(timezone.utc),
        search_document=(enriched.search_document if enriched else _fallback_search_document(unit)),
        semantic_metadata=metadata,
    )


async def _link_related_memories(conn, memories: list[MemoryFragment]) -> list[MemoryFragment]:
    """Link sibling stories when they share a category, person, or topic."""
    linked: list[MemoryFragment] = []
    for memory in memories:
        metadata = dict(memory.semantic_metadata or {})
        category = str(metadata.get("category") or "Stories")
        people = {str(person).lower() for person in memory.people_mentioned}
        topics = {str(topic).lower() for topic in memory.topics}
        related_ids: list[str] = []
        for other in memories:
            if other.id == memory.id:
                continue
            other_metadata = other.semantic_metadata or {}
            same_category = category == str(other_metadata.get("category") or "Stories")
            shares_person = bool(people & {str(person).lower() for person in other.people_mentioned})
            shares_topic = bool(topics & {str(topic).lower() for topic in other.topics})
            if same_category or shares_person or shares_topic:
                related_ids.append(str(other.id))
        metadata["related_memory_ids"] = related_ids[:8]
        await conn.execute(
            "UPDATE public.memories SET semantic_metadata = $1::jsonb WHERE id = $2",
            json.dumps(metadata), memory.id,
        )
        linked.append(memory.model_copy(update={"semantic_metadata": metadata}))
    return linked


async def _process_session_async(session_id: str) -> None:
    """Preserve a transcript, then create one searchable memory per story."""
    logger.info("Starting synchronous structured memory processing session=%s", session_id)
    if not db_client.pool:
        await db_client.connect()
    if not db_client.pool:
        raise RuntimeError("Database is unavailable for session processing")

    async with db_client.pool.acquire() as conn:
        session = await repositories.get_session(conn, session_id)
        if not session:
            logger.error("Session %s not found", session_id)
            return
        existing_memories = await repositories.list_memories_for_session(conn, session_id)
        if existing_memories:
            logger.info("Session %s already has %s structured source memories. Skipping extraction.", session_id, len(existing_memories))
            saved_memories = existing_memories
        owner_id = await conn.fetchval("SELECT user_id FROM public.sessions WHERE id = $1", session_id)
        if not owner_id:
            raise RuntimeError("Session has no owning user")
        transcript = (session.transcript or "").strip()
        audio_url = session.audio_url

    if not existing_memories:
        segments: list[dict] = []
        if not transcript:
            if not audio_url:
                logger.info("Session %s has no transcript or audio; no memory created", session_id)
                return
            try:
                audio_file_path = await TranscriptionService().download_audio(audio_url)
                segments = await TranscriptionService().transcribe_and_segment(audio_file_path)
                transcript = _transcript_from_segments(segments)
            except Exception:
                logger.exception("Audio transcription failed for session %s", session_id)
                raise
            if not transcript:
                logger.info("Session %s transcription contained no speech", session_id)
                return
            async with db_client.pool.acquire() as conn:
                await conn.execute(
                    """UPDATE public.sessions
                       SET transcript = $1, transcript_segments = $2::jsonb
                       WHERE id = $3""",
                    transcript, json.dumps(segments), session_id,
                )

        units = build_story_units(transcript, ["interview"])
        if not units:
            logger.info("Session %s has no complete story units", session_id)
            return
        logger.info("Split session %s into %d complete story units", session_id, len(units))

        enriched_by_index: dict[int, ExtractedMemory] = {}
        try:
            enriched_by_index = await MemoryExtractorService().extract_structured_memories([unit.content for unit in units])
            logger.info("Enriched %d/%d structured memories for session %s", len(enriched_by_index), len(units), session_id)
        except Exception:
            logger.exception("Semantic enrichment failed for session %s; raising to allow retry", session_id)
            raise

        memories = [
            _memory_from_story(
                unit, enriched_by_index.get(index), session_id=session_id, subject_id=str(session.subject_id), index=index
            )
            for index, unit in enumerate(units)
        ]

        async with db_client.pool.acquire() as conn:
            async with conn.transaction():
                saved_memories = [await repositories.create_memory(conn, memory, owner_id) for memory in memories]
                saved_memories = await _link_related_memories(conn, saved_memories)

    async with db_client.pool.acquire() as conn:
        await repositories.enqueue_memory_indexing(conn, session_id, owner_id)

    logger.info("Processed session %s into %d structured memories. Enqueued indexing.", session_id, len(saved_memories))


async def process_session(session_id: str) -> None:
    await _process_session_async(session_id)

async def _handle_job_wrapper(job_id: str, handler_coro, *args) -> None:
    if not db_client.pool:
        await db_client.connect()
        
    async with db_client.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT payload FROM processing_jobs WHERE id = $1", job_id)
        if not row:
            return
        payload = row['payload']
        if isinstance(payload, str):
            payload = json.loads(payload)
            
        attempts = payload.get("attempts", 0) + 1
        payload["attempts"] = attempts
        await conn.execute("UPDATE processing_jobs SET payload = $1::jsonb WHERE id = $2", json.dumps(payload), job_id)

    try:
        await handler_coro(*args)
        async with db_client.pool.acquire() as conn:
            await repositories.complete_processing_job(conn, job_id, "completed")
    except Exception as e:
        logger.exception("Job %s failed on attempt %d", job_id, attempts)
        async with db_client.pool.acquire() as conn:
            if attempts >= 3:
                await repositories.complete_processing_job(conn, job_id, "failed")
            else:
                if isinstance(e, (ValueError, TypeError)):
                    await repositories.complete_processing_job(conn, job_id, "failed")
                else:
                    await repositories.complete_processing_job(conn, job_id, "queued")

async def run_session_job(job_id: str, session_id: str, already_claimed: bool = False) -> None:
    if not already_claimed:
        if not db_client.pool:
            await db_client.connect()
        async with db_client.pool.acquire() as conn:
            claimed, _ = await repositories.claim_processing_job(conn, job_id)
            if not claimed:
                return
    await _handle_job_wrapper(job_id, _process_session_async, session_id)


async def _run_index_memory_async(session_id: str) -> None:
    if not db_client.pool:
        await db_client.connect()
        
    async with db_client.pool.acquire() as conn:
        owner_id = await conn.fetchval("SELECT user_id FROM public.sessions WHERE id = $1", session_id)
        if not owner_id:
            raise ValueError("Session has no owning user")
            
        from app.services.embedding_service import EmbeddingService
        expected_embedding_model = f"{EmbeddingService().settings.gemini_embedding_model}:{INDEX_FORMAT_VERSION}"
        
        # Find all unindexed or outdated memories for this session
        rows = await conn.fetch(
            """
            SELECT m.*
            FROM public.memories m
            WHERE m.session_id = $1
              AND (
                NOT EXISTS (SELECT 1 FROM public.memory_chunks c WHERE c.memory_id = m.id)
                OR EXISTS (
                  SELECT 1 FROM public.memory_chunks c
                  WHERE c.memory_id = m.id
                    AND (c.indexed_at IS NULL OR c.embedding_model IS DISTINCT FROM $2)
                )
              )
            ORDER BY m.created_at ASC
            """,
            session_id, expected_embedding_model
        )
        
        if not rows:
            logger.info("No unindexed memories found for session %s.", session_id)
            # Find subject_id to enqueue retraining
            subject_id = await conn.fetchval("SELECT subject_id FROM public.sessions WHERE id = $1", session_id)
            if subject_id:
                await repositories.enqueue_persona_retraining(conn, str(subject_id), str(owner_id))
            return
            
        memory_payloads = [dict(row) for row in rows]
        for payload in memory_payloads:
            for field in ("emotion_tags", "topics", "people_mentioned", "semantic_metadata"):
                if isinstance(payload.get(field), str):
                    payload[field] = json.loads(payload[field])

    logger.info("Indexing %d memories for session %s", len(memory_payloads), session_id)
    async with db_client.pool.acquire() as conn:
        await index_memories(memory_payloads, str(owner_id), conn=conn)
        
        # Once successfully indexed, enqueue persona retraining
        subject_id = await conn.fetchval("SELECT subject_id FROM public.sessions WHERE id = $1", session_id)
        if subject_id:
            await repositories.enqueue_persona_retraining(conn, str(subject_id), str(owner_id))

async def run_index_memory_job(job_id: str, session_id: str, already_claimed: bool = False) -> None:
    if not already_claimed:
        if not db_client.pool:
            await db_client.connect()
        async with db_client.pool.acquire() as conn:
            claimed, _ = await repositories.claim_processing_job(conn, job_id)
            if not claimed:
                return
    await _handle_job_wrapper(job_id, _run_index_memory_async, session_id)


async def _run_retrain_persona_async(subject_id: str) -> None:
    from app.workers.retrain_persona import retrain_persona
    await retrain_persona(subject_id)

async def run_retrain_persona_job(job_id: str, subject_id: str, already_claimed: bool = False) -> None:
    if not already_claimed:
        if not db_client.pool:
            await db_client.connect()
        async with db_client.pool.acquire() as conn:
            claimed, _ = await repositories.claim_processing_job(conn, job_id)
            if not claimed:
                return
    await _handle_job_wrapper(job_id, _run_retrain_persona_async, subject_id)
