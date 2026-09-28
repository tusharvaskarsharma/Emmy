"""Private Supabase Storage support for audio submitted by interview sessions."""

from __future__ import annotations

import httpx

from app.config import get_settings


class SessionAudioStorageService:
    """Stores recordings privately; clients never receive a provider service key or public URL."""

    bucket = "echo-session-audio"

    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.supabase_url
        self.headers = {
            "apikey": settings.supabase_service_role_key,
            "Authorization": f"Bearer {settings.supabase_service_role_key}",
        }

    async def _ensure_bucket(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            f"{self.base_url}/storage/v1/bucket",
            headers={**self.headers, "Content-Type": "application/json"},
            json={"id": self.bucket, "name": self.bucket, "public": False},
        )
        if response.status_code not in (200, 201, 400, 409):
            response.raise_for_status()

    @staticmethod
    def _path(user_id: str, session_id: str, suffix: str) -> str:
        safe_suffix = suffix if suffix in {"webm", "m4a", "wav", "mp3", "ogg"} else "webm"
        return f"{user_id}/{session_id}/recording.{safe_suffix}"

    async def upload(self, user_id: str, session_id: str, content: bytes, content_type: str) -> str:
        suffix = content_type.split("/")[-1].split(";")[0].replace("x-", "")
        path = self._path(user_id, session_id, suffix)
        async with httpx.AsyncClient(timeout=90) as client:
            await self._ensure_bucket(client)
            response = await client.post(
                f"{self.base_url}/storage/v1/object/{self.bucket}/{path}",
                headers={**self.headers, "Content-Type": content_type, "x-upsert": "true"},
                content=content,
            )
            response.raise_for_status()
        # Persist an opaque storage URI. The worker resolves it using its own
        # service-role credential; there is no public or expiring URL in the DB.
        return f"supabase://{self.bucket}/{path}"

    async def download(self, storage_uri: str) -> bytes:
        prefix = f"supabase://{self.bucket}/"
        if not storage_uri.startswith(prefix):
            raise ValueError("Unexpected session-audio storage URI")
        path = storage_uri.removeprefix(prefix)
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.get(
                f"{self.base_url}/storage/v1/object/{self.bucket}/{path}",
                headers=self.headers,
            )
            response.raise_for_status()
            return response.content

    async def delete(self, storage_uri: str) -> None:
        """Permanently remove one private session recording from Storage."""
        prefix = f"supabase://{self.bucket}/"
        if not storage_uri.startswith(prefix):
            raise ValueError("Unexpected session-audio storage URI")
        path = storage_uri.removeprefix(prefix)
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.delete(
                f"{self.base_url}/storage/v1/object/{self.bucket}",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"prefixes": [path]},
            )
            # A recording can have already been removed manually.  That still
            # satisfies this erasure request and keeps retrying safe.
            if response.status_code != 404:
                response.raise_for_status()

    async def delete_all(self, user_id: str) -> None:
        """Permanently remove all private session recordings for a user."""
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{self.base_url}/storage/v1/object/list/{self.bucket}",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"prefix": f"{user_id}/", "limit": 1000},
            )
            if response.status_code == 404:
                return
            response.raise_for_status()
            
            paths_to_delete = []
            for entry in response.json():
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not name:
                    continue
                
                # Check if it's a file directly under user_id/
                if entry.get("id"): 
                    paths_to_delete.append(f"{user_id}/{name}")
                else:
                    # It's a folder (session_id)
                    folder_res = await client.post(
                        f"{self.base_url}/storage/v1/object/list/{self.bucket}",
                        headers={**self.headers, "Content-Type": "application/json"},
                        json={"prefix": f"{user_id}/{name}/", "limit": 100},
                    )
                    if folder_res.is_success:
                        for sub_entry in folder_res.json():
                            if isinstance(sub_entry, dict) and sub_entry.get("name"):
                                paths_to_delete.append(f"{user_id}/{name}/{sub_entry['name']}")
            
            if not paths_to_delete:
                return
                
            for i in range(0, len(paths_to_delete), 100):
                batch = paths_to_delete[i:i+100]
                deletion = await client.delete(
                    f"{self.base_url}/storage/v1/object/{self.bucket}",
                    headers={**self.headers, "Content-Type": "application/json"},
                    json={"prefixes": batch},
                )
                deletion.raise_for_status()
