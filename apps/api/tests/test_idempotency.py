import asyncio
import pytest
from unittest.mock import MagicMock
import uuid

import app.workers.process_session as process_module
from app.workers.process_session import _process_session_async, run_session_job

class FakeConnectionIdempotency:
    def __init__(self):
        self.memories = []
        self.chunks = []
        self.jobs = {}
    
    async def fetchval(self, query, *args):
        if "SELECT user_id FROM public.sessions WHERE id" in query:
            return "user-1"
        return None
        
    async def fetchrow(self, query, *args):
        if "UPDATE processing_jobs SET status" in query and "RETURNING id" in query:
            return {"id": args[0]}
        if "SELECT payload FROM processing_jobs" in query:
            return {"payload": {"session_id": "session-1"}}
        return None

    async def fetch(self, query, *args):
        if "SELECT * FROM memories WHERE session_id" in query:
            return [m for m in self.memories if m['session_id'] == str(args[0])]
        return []

    async def execute(self, query, *args):
        if "UPDATE public.sessions" in query:
            pass
        elif "UPDATE public.memories" in query:
            pass
        elif "INSERT INTO public.memory_chunks" in query:
            self.chunks.append(args)
        elif "UPDATE processing_jobs SET status" in query:
            self.jobs[args[1]] = args[0]
        elif "UPDATE public.memory_chunks SET indexed_at" in query:
            pass

    def transaction(self):
        class Tx:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
        return Tx()


def test_idempotent_session_processing(monkeypatch):
    conn = FakeConnectionIdempotency()
    
    class FakeDBClient:
        pool = MagicMock()
        async def connect(self): pass
        
    fake_pool = MagicMock()
    fake_pool.acquire.return_value.__aenter__.return_value = conn
    FakeDBClient.pool = fake_pool
    
    monkeypatch.setattr(process_module, "db_client", FakeDBClient)
    
    # Mock repositories
    class FakeSession:
        subject_id = "user-1"
        transcript = "This is a test transcript."
        audio_url = None
        
    async def fake_get_session(conn, session_id):
        return FakeSession()
        
    monkeypatch.setattr(process_module.repositories, "get_session", fake_get_session)
    
    # Track extractions
    extractions = []
    class FakeExtractor:
        async def extract_structured_memories(self, units):
            from app.services.memory_extractor import ExtractedMemory
            extractions.extend(units)
            return {0: ExtractedMemory(
                content="Test content",
                emotion_tags=[],
                topics=["Test Topic"],
                people_mentioned=[],
                confidence_score=0.9,
                search_document="Test content",
                semantic_metadata={}
            )}
            
    monkeypatch.setattr(process_module, "MemoryExtractorService", FakeExtractor)
    
    # Mock repositories.create_memory to add to conn.memories
    async def fake_create_memory(conn, memory, owner_id):
        conn.memories.append(memory.model_dump(mode="json"))
        return memory
        
    monkeypatch.setattr(process_module.repositories, "create_memory", fake_create_memory)
    
    # Track Pinecone Indexing
    index_calls = []
    async def fake_enqueue_memory_indexing(conn, session_id, owner_id):
        index_calls.append(session_id)
        
    monkeypatch.setattr(process_module.repositories, "enqueue_memory_indexing", fake_enqueue_memory_indexing)
    
    # Mock retrain_persona
    async def fake_retrain(subject_id): pass
    monkeypatch.setattr(process_module, "retrain_persona", fake_retrain, raising=False)
    
    # 1. First run
    asyncio.run(run_session_job("job-1", "session-1", already_claimed=True))
    assert len(extractions) > 0, "Should have extracted stories"
    assert len(conn.memories) > 0, "Should have saved memories"
    assert len(index_calls) == 1, "Should have enqueued memories"
    assert conn.jobs["job-1"] == "completed"
    
    # Save state
    extractions_count = len(extractions)
    memories_count = len(conn.memories)
    
    # 2. Same session processed twice -> no duplicate memories
    asyncio.run(run_session_job("job-2", "session-1", already_claimed=True))
    assert len(extractions) == extractions_count, "Should NOT extract again"
    assert len(conn.memories) == memories_count, "Should NOT create new memories"
    assert len(index_calls) == 2, "Should re-enqueue existing memories"
    assert conn.jobs["job-2"] == "completed"

def test_pinecone_failure_leaves_job_failed_and_retryable(monkeypatch):
    conn = FakeConnectionIdempotency()
    class FakeDBClient:
        pool = MagicMock()
        async def connect(self): pass
    fake_pool = MagicMock()
    fake_pool.acquire.return_value.__aenter__.return_value = conn
    FakeDBClient.pool = fake_pool
    monkeypatch.setattr(process_module, "db_client", FakeDBClient)
    
    class FakeSession:
        subject_id = "user-1"
        transcript = "Test."
        audio_url = None
    async def fake_get_session(conn, session_id): return FakeSession()
    monkeypatch.setattr(process_module.repositories, "get_session", fake_get_session)
    
    extractions = []
    class FakeExtractor:
        async def extract_structured_memories(self, units): 
            from app.services.memory_extractor import ExtractedMemory
            extractions.extend(units)
            return {0: ExtractedMemory(
                content="Test content",
                emotion_tags=[],
                topics=["Test Topic"],
                people_mentioned=[],
                confidence_score=0.9,
                search_document="Test content",
                semantic_metadata={}
            )}
    monkeypatch.setattr(process_module, "MemoryExtractorService", FakeExtractor)
    
    async def fake_create_memory(conn, memory, owner_id):
        conn.memories.append(memory.model_dump(mode="json"))
        return memory
    monkeypatch.setattr(process_module.repositories, "create_memory", fake_create_memory)
    
    # Mock retrain_persona
    async def fake_retrain(subject_id): pass
    monkeypatch.setattr(process_module, "retrain_persona", fake_retrain, raising=False)
    
    # Force Pinecone to fail
    async def failing_index(conn, session_id, owner_id):
        raise Exception("Enqueue failure")
    monkeypatch.setattr(process_module.repositories, "enqueue_memory_indexing", failing_index)
    
    # 3. Pinecone indexing fails -> job remains retryable
    asyncio.run(run_session_job("job-3", "session-1", already_claimed=True))
    assert conn.jobs["job-3"] == "queued"
    assert len(conn.memories) > 0
    
    # Now fix Pinecone
    index_calls = []
    async def working_index(conn, session_id, owner_id):
        index_calls.append(session_id)
    monkeypatch.setattr(process_module.repositories, "enqueue_memory_indexing", working_index)
    
    # 4. Retry successfully indexes without duplicating
    mem_count = len(conn.memories)
    asyncio.run(run_session_job("job-3-retry", "session-1", already_claimed=True))
    assert conn.jobs["job-3-retry"] == "completed"
    assert len(index_calls) == 1
    assert len(conn.memories) == mem_count # No duplicates


def test_deterministic_vector_id_generation():
    from app.workers.index_memory import vector_id_for
    from app.services.memory_chunking import MemoryChunk
    
    class FakeChunk:
        vector_id_suffix = "chunk-0"
        chunk_index = 0
        category = "Test"
        content = "Test content"
        search_text = "Test search"
        keywords = []
    
    chunk = FakeChunk()
    
    # 7. Same memory chunk always generates the same Pinecone vector ID
    memory_id = str(uuid.uuid4())
    id1 = vector_id_for(memory_id, chunk)
    id2 = vector_id_for(memory_id, chunk)
    
    assert id1 == id2
    assert memory_id in id1

