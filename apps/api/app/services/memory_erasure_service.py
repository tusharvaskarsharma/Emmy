import logging
from typing import Dict, Any

from app.services.pinecone_service import PineconeService
from app.services.memory_storage_service import MemoryStorageService

logger = logging.getLogger(__name__)

class MemoryErasureService:
    def __init__(self):
        self.pinecone_service = PineconeService()
        self.memory_storage = MemoryStorageService()

    async def erase_memory_external(self, user_id: str, memory_id: str) -> Dict[str, Any]:
        """
        Coordinates idempotent deletion of external artifacts for a single memory.
        Returns a status dictionary.
        Does NOT delete from the database.
        """
        status = {
            "pinecone": "pending",
            "storage_memories": "pending",
        }

        # 1. Pinecone Vectors (Idempotent by metadata filter)
        try:
            self.pinecone_service.delete_vectors(
                namespace=user_id, 
                filter={"memory_id": {"$eq": memory_id}}
            )
            status["pinecone"] = "success"
        except Exception as e:
            logger.error(f"Pinecone memory erasure failed for user {user_id}, memory {memory_id}: {e}")
            status["pinecone"] = "failed"

        # 2. Storage Memories
        try:
            await self.memory_storage.delete_memory(user_id, memory_id)
            status["storage_memories"] = "success"
        except Exception as e:
            logger.error(f"Memory storage erasure failed for user {user_id}, memory {memory_id}: {e}")
            status["storage_memories"] = "failed"

        return status
