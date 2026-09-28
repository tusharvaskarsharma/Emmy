import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException
import asyncpg

from app.routers import memories as memory_router
from app.models.memory import MemoryFragment, ConsentLevel

class FakeConnection:
    def __init__(self):
        self.execute_calls = []

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        
    @asynccontextmanager
    async def transaction(self):
        yield self


def test_delete_memory_external_failures_prevent_db_deletion(monkeypatch):
    class FakeMemoryErasureService:
        async def erase_memory_external(self, user_id, memory_id):
            return {"pinecone": "failed", "storage_memories": "success"}
            
    monkeypatch.setattr(memory_router, "MemoryErasureService", FakeMemoryErasureService)
    
    class FakeRepo:
        @staticmethod
        async def get_memory(conn, memory_id, user_id):
            return MemoryFragment(
                id="memory-1", session_id="session-1", subject_id="subject-1", 
                content="test", emotion_tags=[], topics=[], people_mentioned=[],
                consent_level=ConsentLevel.PRIVATE, confidence_score=0.9
            )
            
    monkeypatch.setattr(memory_router.repositories, "get_memory", FakeRepo.get_memory)
    
    conn = FakeConnection()
    
    with pytest.raises(HTTPException) as error:
        asyncio.run(memory_router.delete_memory("memory-1", {"sub": "user-1"}, conn))
        
    assert error.value.status_code == 503
    assert len(conn.execute_calls) == 0


def test_delete_memory_success_cascades_safely(monkeypatch):
    class FakeMemoryErasureService:
        async def erase_memory_external(self, user_id, memory_id):
            return {"pinecone": "success", "storage_memories": "success"}
            
    monkeypatch.setattr(memory_router, "MemoryErasureService", FakeMemoryErasureService)
    
    class FakeRepo:
        @staticmethod
        async def get_memory(conn, memory_id, user_id):
            return MemoryFragment(
                id="memory-1", session_id="session-1", subject_id="subject-1", 
                content="test", emotion_tags=[], topics=[], people_mentioned=[],
                consent_level=ConsentLevel.PRIVATE, confidence_score=0.9
            )
            
        @staticmethod
        async def delete_memory(conn, memory_id, user_id):
            conn.execute_calls.append(("DELETE FROM memories WHERE id = $1 AND user_id = $2", (memory_id, user_id)))
            
    monkeypatch.setattr(memory_router.repositories, "get_memory", FakeRepo.get_memory)
    monkeypatch.setattr(memory_router.repositories, "delete_memory", FakeRepo.delete_memory)
    
    conn = FakeConnection()
    
    result = asyncio.run(memory_router.delete_memory("memory-1", {"sub": "user-1"}, conn))
    
    assert result["deleted"] is True
    
    assert len(conn.execute_calls) == 2
    assert "DELETE FROM mind_evidence" in conn.execute_calls[0][0]
    assert conn.execute_calls[0][1] == ("memory-1",)
    
    assert "DELETE FROM memories" in conn.execute_calls[1][0]
    assert conn.execute_calls[1][1] == ("memory-1", "user-1")


def test_delete_memory_external_service_isolation(monkeypatch):
    calls = []

    class FakePineconeService:
        def delete_vectors(self, namespace, filter=None, delete_all=False):
            calls.append(("pinecone", namespace, filter, delete_all))

    class FakeMemoryStorageService:
        async def delete_memory(self, user_id, memory_id):
            calls.append(("storage", user_id, memory_id))

    from app.services import memory_erasure_service
    monkeypatch.setattr(memory_erasure_service, "PineconeService", lambda: FakePineconeService())
    monkeypatch.setattr(memory_erasure_service, "MemoryStorageService", lambda: FakeMemoryStorageService())

    service = memory_erasure_service.MemoryErasureService()

    
    result = asyncio.run(service.erase_memory_external("user-1", "memory-1"))
    
    assert result["pinecone"] == "success"
    assert result["storage_memories"] == "success"
    
    # Must use filter, not delete_all
    assert ("pinecone", "user-1", {"memory_id": {"$eq": "memory-1"}}, False) in calls
    assert ("storage", "user-1", "memory-1") in calls

