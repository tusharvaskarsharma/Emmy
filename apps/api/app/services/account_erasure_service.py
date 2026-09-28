import logging
import httpx
from typing import Dict, Any

from app.config import get_settings
from app.services.pinecone_service import PineconeService
from app.services.session_audio_storage_service import SessionAudioStorageService
from app.services.memory_storage_service import MemoryStorageService

logger = logging.getLogger(__name__)

class AccountErasureService:
    def __init__(self):
        self.settings = get_settings()
        self.pinecone_service = PineconeService()
        self.audio_storage = SessionAudioStorageService()
        self.memory_storage = MemoryStorageService()
        
    async def erase_account(self, user_id: str) -> Dict[str, Any]:
        """
        Coordinates full, idempotent deletion of an account across all durable stores.
        Returns a status dictionary detailing what was cleaned up.
        """
        status = {
            "pinecone": "pending",
            "storage_audio": "pending",
            "storage_memories": "pending",
            "database": "pending",
            "auth": "pending"
        }

        # 1. Pinecone Vectors (Idempotent)
        try:
            # We use namespace=user_id for vector isolation per user.
            self.pinecone_service.delete_vectors(namespace=user_id, delete_all=True)
            status["pinecone"] = "success"
        except Exception as e:
            logger.error(f"Pinecone erasure failed for user {user_id}: {e}")
            status["pinecone"] = "failed"
            # We do NOT raise here; partial failure must be reported.

        # 2. Storage Audio (Idempotent by prefix)
        try:
            await self.audio_storage.delete_all(user_id)
            status["storage_audio"] = "success"
        except Exception as e:
            logger.error(f"Audio storage erasure failed for user {user_id}: {e}")
            status["storage_audio"] = "failed"

        # 3. Storage Memories (Idempotent by prefix)
        try:
            await self.memory_storage.delete_all(user_id)
            status["storage_memories"] = "success"
        except Exception as e:
            logger.error(f"Memory storage erasure failed for user {user_id}: {e}")
            status["storage_memories"] = "failed"

        # If any external cleanup failed, we halt before deleting the DB/Auth user.
        # This keeps the user alive so they can safely retry the deletion later.
        if "failed" in status.values():
            return status

        # 4. Supabase Auth (which cascades DB)
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.delete(
                    f"{self.settings.supabase_url.rstrip('/')}/auth/v1/admin/users/{user_id}", 
                    headers={"apikey": self.settings.supabase_service_role_key, "Authorization": f"Bearer {self.settings.supabase_service_role_key}"}
                )
            if response.status_code in (200, 204, 404):
                # 404 means the user is already deleted, which satisfies idempotency
                status["auth"] = "success"
                status["database"] = "success"
            else:
                logger.error(f"Supabase Auth deletion failed for user {user_id} with status {response.status_code}: {response.text}")
                status["auth"] = "failed"
                status["database"] = "pending"
        except Exception as e:
            logger.error(f"Supabase Auth API error for user {user_id}: {e}")
            status["auth"] = "failed"
            status["database"] = "pending"
            
        return status
