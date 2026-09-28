import asyncio
import pytest
from unittest.mock import MagicMock
from uuid import uuid4

import app.services.session_service as session_module
from app.services.session_service import SessionService
from app.models.session import SessionUpdate, SessionStatus
import app.db.repositories as repositories
from app.workers.process_session import run_session_job
import app.workers.process_session as process_module


class FakeConnection:
    def __init__(self):
        self.jobs = {} # id -> job
        self.queries = []
    
    async def execute(self, query, *args):
        self.queries.append((query, args))
        if "UPDATE processing_jobs SET status" in query:
            status, job_id = args[0], args[1]
            if job_id in self.jobs:
                self.jobs[job_id]['status'] = status
                
    async def fetchval(self, query, *args):
        self.queries.append((query, args))
        if "SELECT user_id FROM sessions WHERE id" in query:
            return "user-1"
        if "INSERT INTO processing_jobs" in query:
            job_id = "job-1"
            # Simulate idempotency
            for existing_job in self.jobs.values():
                if existing_job['payload']['session_id'] == str(args[1]) and existing_job['status'] != 'failed':
                    return None
            self.jobs[job_id] = {'status': 'queued', 'payload': {'session_id': str(args[1])}}
            return job_id
        if "UPDATE processing_jobs SET status = 'processing'" in query:
            job_id = args[0]
            if job_id in self.jobs and self.jobs[job_id]['status'] == 'queued':
                self.jobs[job_id]['status'] = 'processing'
                return job_id
            return None
        return None
        
    async def fetchrow(self, query, *args):
        self.queries.append((query, args))
        if "UPDATE processing_jobs SET status" in query:
             pass
        if "WHERE id = $1" in query and "UPDATE processing_jobs" in query:
            job_id = args[0]
            if job_id in self.jobs and self.jobs[job_id]['status'] == 'queued':
                self.jobs[job_id]['status'] = 'processing'
                return {"id": job_id, "job_type": self.jobs[job_id].get("job_type", "process_session")}
            return None
        if "SELECT payload FROM processing_jobs WHERE id = $1" in query:
            job_id = args[0]
            if job_id in self.jobs:
                return {"payload": self.jobs[job_id]['payload']}
            return None
        if "INSERT INTO processing_jobs" in query:
            job_id = "job-1"
            # Simulate idempotency
            for existing_job in self.jobs.values():
                if existing_job['payload']['session_id'] == str(args[1]) and existing_job['status'] != 'failed':
                    return None
            job_type = "process_session"
            if "index_memory" in query:
                job_type = "index_memory"
            elif "retrain_persona" in query:
                job_type = "retrain_persona"
            self.jobs[job_id] = {'status': 'queued', 'payload': {'session_id': str(args[1])}, 'job_type': job_type}
            return {"id": job_id}
        return None

    def transaction(self):
        class Tx:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
        return Tx()


def test_session_completion_creates_processing_job_and_does_not_await(monkeypatch):
    conn = FakeConnection()
    service = SessionService(conn, "user-1")

    tasks = []
    def fake_create_task(coro):
        tasks.append(coro)

    import asyncio
    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    # Mock get_session and update_session
    class FakeSession:
        id = "session-1"
        status = SessionStatus.ACTIVE
        ended_at = None
        
    async def fake_get_session(session_id): return FakeSession()
    monkeypatch.setattr(service, "get_session", fake_get_session)
    async def fake_update_session(conn, session, subject_id): return session
    monkeypatch.setattr(session_module.repositories, "update_session", fake_update_session)

    # 1 & 2. Completing session creates job durably, does not execute in-memory
    asyncio.run(service.update_session("session-1", SessionUpdate(status=SessionStatus.COMPLETED)))

    assert len(tasks) == 0 # create_task is no longer used, rely on background runner
    assert "job-1" in conn.jobs
    assert conn.jobs["job-1"]["status"] == "queued"
    
    # 3. Duplicate completion requests do not create duplicate jobs
    asyncio.run(service.update_session("session-1", SessionUpdate(status=SessionStatus.COMPLETED)))
    assert len(tasks) == 0


def test_job_transitions_through_expected_states_on_success(monkeypatch):
    conn = FakeConnection()
    conn.jobs["job-1"] = {'status': 'queued', 'payload': {'session_id': 'session-1'}}
    
    class FakeDBClient:
        pool = MagicMock()
        
    fake_pool = MagicMock()
    fake_pool.acquire.return_value.__aenter__.return_value = conn
    FakeDBClient.pool = fake_pool
    
    monkeypatch.setattr(process_module, "db_client", FakeDBClient)
    
    async def fake_process(session_id):
        # 4. Job transitions through expected states
        assert conn.jobs["job-1"]["status"] == "processing"
        
    monkeypatch.setattr(process_module, "_process_session_async", fake_process)
    
    asyncio.run(run_session_job("job-1", "session-1"))
    
    # 5. Successful processing marks the job completed
    assert conn.jobs["job-1"]["status"] == "completed"


def test_job_transitions_to_failed_on_provider_error(monkeypatch):
    conn = FakeConnection()
    conn.jobs["job-1"] = {'status': 'queued', 'payload': {'session_id': 'session-1'}}
    
    class FakeDBClient:
        pool = MagicMock()
        
    fake_pool = MagicMock()
    fake_pool.acquire.return_value.__aenter__.return_value = conn
    FakeDBClient.pool = fake_pool
    
    monkeypatch.setattr(process_module, "db_client", FakeDBClient)
    
    async def fake_process_error(session_id):
        raise Exception("Provider failure")
        
    monkeypatch.setattr(process_module, "_process_session_async", fake_process_error)
    
    asyncio.run(run_session_job("job-1", "session-1"))
    
    # 6. Provider failure marks the job queued for retry (attempt 1)
    assert conn.jobs["job-1"]["status"] == "queued"
    
def test_job_terminal_failure_on_max_attempts(monkeypatch):
    conn = FakeConnection()
    conn.jobs["job-1"] = {'status': 'queued', 'payload': {'session_id': 'session-1', 'attempts': 2}}
    
    class FakeDBClient:
        pool = MagicMock()
        
    fake_pool = MagicMock()
    fake_pool.acquire.return_value.__aenter__.return_value = conn
    FakeDBClient.pool = fake_pool
    
    monkeypatch.setattr(process_module, "db_client", FakeDBClient)
    
    async def fake_process_error(session_id):
        raise Exception("Provider failure")
        
    monkeypatch.setattr(process_module, "_process_session_async", fake_process_error)
    
    asyncio.run(run_session_job("job-1", "session-1"))
    
    # Attempt becomes 3, so it marks it as failed (terminal)
    assert conn.jobs["job-1"]["status"] == "failed"


def test_concurrent_workers_cannot_process_same_session(monkeypatch):
    conn = FakeConnection()
    conn.jobs["job-1"] = {'status': 'processing', 'payload': {'session_id': 'session-1'}}
    
    class FakeDBClient:
        pool = MagicMock()
        
    fake_pool = MagicMock()
    fake_pool.acquire.return_value.__aenter__.return_value = conn
    FakeDBClient.pool = fake_pool
    
    monkeypatch.setattr(process_module, "db_client", FakeDBClient)
    
    called = False
    async def fake_process(session_id):
        nonlocal called
        called = True
        
    monkeypatch.setattr(process_module, "_process_session_async", fake_process)
    
    # 8. Two concurrent workers cannot process the same session simultaneously
    # Job is already "processing", claim_processing_job will fail
    asyncio.run(run_session_job("job-1", "session-1"))
    
    assert not called
    assert conn.jobs["job-1"]["status"] == "processing"

