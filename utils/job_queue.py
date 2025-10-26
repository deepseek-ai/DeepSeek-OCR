"""
Job Queue System for Batch Processing
Handles job persistence, status tracking, and resume functionality
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum


class JobStatus(str, Enum):
    """Job status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobQueue:
    """SQLite-based job queue for batch processing"""

    def __init__(self, db_path: str = "jobs.db"):
        """Initialize job queue with SQLite database"""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                total_files INTEGER NOT NULL,
                processed_files INTEGER DEFAULT 0,
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT
            )
        """)

        # Job files table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_path TEXT,
                status TEXT NOT NULL,
                result_path TEXT,
                error_message TEXT,
                processed_at TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs(job_id)
            )
        """)

        # Job results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                file_id INTEGER NOT NULL,
                output_format TEXT NOT NULL,
                output_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs(job_id),
                FOREIGN KEY (file_id) REFERENCES job_files(file_id)
            )
        """)

        conn.commit()
        conn.close()

    def create_job(self, name: str, files: List[str], config: Dict[str, Any]) -> str:
        """Create a new batch job"""
        import uuid
        job_id = str(uuid.uuid4())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert job
        cursor.execute("""
            INSERT INTO jobs (job_id, name, status, total_files, config)
            VALUES (?, ?, ?, ?, ?)
        """, (job_id, name, JobStatus.PENDING, len(files), json.dumps(config)))

        # Insert files
        for filename in files:
            cursor.execute("""
                INSERT INTO job_files (job_id, filename, status)
                VALUES (?, ?, ?)
            """, (job_id, filename, JobStatus.PENDING))

        conn.commit()
        conn.close()

        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job details"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()

        conn.close()

        if row:
            return dict(row)
        return None

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get all jobs"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
        rows = cursor.fetchall()

        conn.close()

        return [dict(row) for row in rows]

    def get_pending_files(self, job_id: str) -> List[Dict[str, Any]]:
        """Get pending files for a job"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM job_files
            WHERE job_id = ? AND status = ?
            ORDER BY file_id
        """, (job_id, JobStatus.PENDING))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def update_job_status(self, job_id: str, status: JobStatus, error_message: Optional[str] = None):
        """Update job status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp_field = None
        if status == JobStatus.PROCESSING:
            timestamp_field = "started_at"
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            timestamp_field = "completed_at"

        if timestamp_field:
            cursor.execute(f"""
                UPDATE jobs
                SET status = ?, {timestamp_field} = CURRENT_TIMESTAMP, error_message = ?
                WHERE job_id = ?
            """, (status, error_message, job_id))
        else:
            cursor.execute("""
                UPDATE jobs
                SET status = ?, error_message = ?
                WHERE job_id = ?
            """, (status, error_message, job_id))

        conn.commit()
        conn.close()

    def update_file_status(self, file_id: int, status: JobStatus,
                          result_path: Optional[str] = None,
                          error_message: Optional[str] = None):
        """Update file processing status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE job_files
            SET status = ?, result_path = ?, error_message = ?, processed_at = CURRENT_TIMESTAMP
            WHERE file_id = ?
        """, (status, result_path, error_message, file_id))

        # Update job processed count
        cursor.execute("""
            UPDATE jobs
            SET processed_files = (
                SELECT COUNT(*) FROM job_files
                WHERE job_id = (SELECT job_id FROM job_files WHERE file_id = ?)
                AND status IN (?, ?)
            )
            WHERE job_id = (SELECT job_id FROM job_files WHERE file_id = ?)
        """, (file_id, JobStatus.COMPLETED, JobStatus.FAILED, file_id))

        conn.commit()
        conn.close()

    def add_result(self, job_id: str, file_id: int, output_format: str, output_path: str):
        """Add a result output for a processed file"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO job_results (job_id, file_id, output_format, output_path)
            VALUES (?, ?, ?, ?)
        """, (job_id, file_id, output_format, output_path))

        conn.commit()
        conn.close()

    def get_job_results(self, job_id: str) -> List[Dict[str, Any]]:
        """Get all results for a job"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT jr.*, jf.filename
            FROM job_results jr
            JOIN job_files jf ON jr.file_id = jf.file_id
            WHERE jr.job_id = ?
            ORDER BY jr.created_at
        """, (job_id,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_job_progress(self, job_id: str) -> Dict[str, Any]:
        """Get job progress summary"""
        job = self.get_job(job_id)
        if not job:
            return {}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM job_files
            WHERE job_id = ?
            GROUP BY status
        """, (job_id,))

        status_counts = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()

        return {
            "job_id": job_id,
            "name": job["name"],
            "status": job["status"],
            "total_files": job["total_files"],
            "processed_files": job["processed_files"],
            "pending": status_counts.get(JobStatus.PENDING, 0),
            "processing": status_counts.get(JobStatus.PROCESSING, 0),
            "completed": status_counts.get(JobStatus.COMPLETED, 0),
            "failed": status_counts.get(JobStatus.FAILED, 0),
            "progress": (job["processed_files"] / job["total_files"] * 100) if job["total_files"] > 0 else 0
        }

    def delete_job(self, job_id: str):
        """Delete a job and all associated data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM job_results WHERE job_id = ?", (job_id,))
        cursor.execute("DELETE FROM job_files WHERE job_id = ?", (job_id,))
        cursor.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

        conn.commit()
        conn.close()

    def cancel_job(self, job_id: str):
        """Cancel a job"""
        self.update_job_status(job_id, JobStatus.CANCELLED)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Cancel all pending files
        cursor.execute("""
            UPDATE job_files
            SET status = ?
            WHERE job_id = ? AND status = ?
        """, (JobStatus.CANCELLED, job_id, JobStatus.PENDING))

        conn.commit()
        conn.close()
