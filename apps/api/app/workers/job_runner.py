import asyncio
import logging
from app.db.client import db_client
from app.workers.process_session import run_session_job

logger = logging.getLogger(__name__)

_runner_running = False

async def start_job_runner(poll_interval: int = 15, lease_minutes: int = 5):
    """
    Periodically polls the processing_jobs table to claim and run queued or stale jobs.
    """
    global _runner_running
    _runner_running = True
    logger.info("Starting background job runner loop.")
    
    while _runner_running:
        try:
            if db_client.pool:
                async with db_client.pool.acquire() as conn:
                    from app.db import repositories
                    claim_result = await repositories.claim_processing_job(conn, lease_minutes=lease_minutes)
                    
                    if claim_result[0]:
                        job_id, job_type = claim_result
                        row = await conn.fetchrow("SELECT payload FROM processing_jobs WHERE id = $1", job_id)
                        payload = row['payload']
                        if isinstance(payload, str):
                            import json
                            payload = json.loads(payload)
                        
                        logger.info(f"Runner claimed job {job_id} of type {job_type}")
                        
                        if job_type == 'process_session':
                            session_id = payload.get('session_id')
                            await run_session_job(job_id, str(session_id), already_claimed=True)
                        elif job_type == 'index_memory':
                            session_id = payload.get('session_id')
                            from app.workers.process_session import run_index_memory_job
                            await run_index_memory_job(job_id, str(session_id), already_claimed=True)
                        elif job_type == 'retrain_persona':
                            subject_id = payload.get('subject_id')
                            from app.workers.process_session import run_retrain_persona_job
                            await run_retrain_persona_job(job_id, str(subject_id), already_claimed=True)
                        else:
                            logger.error(f"Unknown job_type {job_type} for job {job_id}")
                            await repositories.complete_processing_job(conn, job_id, "failed")
                        
                        # Immediately try to claim another job without sleeping
                        continue

        except Exception:
            logger.exception("Error in job runner loop")
            
        await asyncio.sleep(poll_interval)

async def stop_job_runner(task: asyncio.Task):
    global _runner_running
    _runner_running = False
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
