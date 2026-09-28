import asyncio
import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock

import app.routers.auth as auth_router
import app.services.account_erasure_service as erasure_service_module
from app.config import get_settings

def test_delete_account_cleans_all_stores(monkeypatch):
    calls = []

    class FakeAccountErasureService:
        async def erase_account(self, user_id):
            calls.append(("erase_account", user_id))
            return {
                "pinecone": "success",
                "storage_audio": "success",
                "storage_memories": "success",
                "database": "success",
                "auth": "success"
            }

    monkeypatch.setattr(auth_router, "AccountErasureService", FakeAccountErasureService)
    
    settings = get_settings()
    monkeypatch.setattr(settings, "supabase_url", "https://example.supabase.co")
    monkeypatch.setattr(settings, "supabase_service_role_key", "fake-key")

    current_user = {"sub": "user-1"}

    asyncio.run(auth_router.delete_account(current_user))

    assert ("erase_account", "user-1") in calls

def test_delete_account_handles_partial_failure(monkeypatch):
    calls = []

    class FakeAccountErasureService:
        async def erase_account(self, user_id):
            calls.append(("erase_account", user_id))
            return {
                "pinecone": "failed",
                "storage_audio": "success",
                "storage_memories": "success",
                "database": "pending",
                "auth": "pending"
            }

    monkeypatch.setattr(auth_router, "AccountErasureService", FakeAccountErasureService)
    
    settings = get_settings()
    monkeypatch.setattr(settings, "supabase_url", "https://example.supabase.co")
    monkeypatch.setattr(settings, "supabase_service_role_key", "fake-key")

    current_user = {"sub": "user-1"}

    with pytest.raises(HTTPException) as error:
        asyncio.run(auth_router.delete_account(current_user))
        
    assert error.value.status_code == 502
    assert "Account erasure incomplete" in error.value.detail
    assert ("erase_account", "user-1") in calls


def test_account_erasure_service_flow(monkeypatch):
    calls = []

    class FakePinecone:
        def delete_vectors(self, namespace, **kwargs):
            calls.append(("pinecone", namespace, kwargs))

    class FakeAudioStorage:
        async def delete_all(self, user_id):
            calls.append(("audio", user_id))

    class FakeMemoryStorage:
        async def delete_all(self, user_id):
            calls.append(("memory_storage", user_id))

    class FakeResponse:
        status_code = 200
        text = "ok"

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
        async def delete(self, url, headers=None):
            calls.append(("supabase", url, headers))
            return FakeResponse()

    monkeypatch.setattr(erasure_service_module, "PineconeService", FakePinecone)
    monkeypatch.setattr(erasure_service_module, "SessionAudioStorageService", FakeAudioStorage)
    monkeypatch.setattr(erasure_service_module, "MemoryStorageService", FakeMemoryStorage)
    monkeypatch.setattr(erasure_service_module.httpx, "AsyncClient", FakeAsyncClient)
    
    settings = get_settings()
    monkeypatch.setattr(settings, "supabase_url", "https://example.supabase.co")
    monkeypatch.setattr(settings, "supabase_service_role_key", "fake-key")

    service = erasure_service_module.AccountErasureService()
    status = asyncio.run(service.erase_account("user-1"))

    assert status == {
        "pinecone": "success",
        "storage_audio": "success",
        "storage_memories": "success",
        "database": "success",
        "auth": "success"
    }
    assert ("pinecone", "user-1", {"delete_all": True}) in calls
    assert ("audio", "user-1") in calls
    assert ("memory_storage", "user-1") in calls
    assert calls[3][0] == "supabase"

def test_account_erasure_service_halts_on_pinecone_failure(monkeypatch):
    calls = []

    class FailingPinecone:
        def delete_vectors(self, namespace, **kwargs):
            raise Exception("Pinecone failure")

    class FakeAudioStorage:
        async def delete_all(self, user_id):
            calls.append(("audio", user_id))

    class FakeMemoryStorage:
        async def delete_all(self, user_id):
            calls.append(("memory_storage", user_id))

    monkeypatch.setattr(erasure_service_module, "PineconeService", FailingPinecone)
    monkeypatch.setattr(erasure_service_module, "SessionAudioStorageService", FakeAudioStorage)
    monkeypatch.setattr(erasure_service_module, "MemoryStorageService", FakeMemoryStorage)

    service = erasure_service_module.AccountErasureService()
    status = asyncio.run(service.erase_account("user-1"))

    assert status["pinecone"] == "failed"
    assert status["auth"] == "pending"
    assert ("audio", "user-1") in calls
