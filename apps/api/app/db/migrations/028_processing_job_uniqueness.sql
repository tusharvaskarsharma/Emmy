-- Ensure idempotent background jobs by tracking processing intent uniquely
CREATE UNIQUE INDEX IF NOT EXISTS idx_processing_jobs_session_processing 
ON processing_jobs (user_id, job_type, (payload->>'session_id')) 
WHERE job_type = 'process_session';
