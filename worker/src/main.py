#!/usr/bin/env python3
"""
Tile Generation Worker
Processes jobs from the queue using the complete 7-stage pipeline.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

from .core.job_processor import JobProcessor

def main():
    """Main worker entry point - processes jobs from the queue or a specific job file."""
    import argparse

    parser = argparse.ArgumentParser(description="Tile Generation Worker")
    parser.add_argument("--job-file", type=str, help="Process a specific job file")
    parser.add_argument("--job-stdin", action="store_true", help="Read job from stdin")
    args = parser.parse_args()

    print("🔧 Tile Generation Worker starting...")

    # Note: Jobs directory is created by CLI in project root

    if args.job_stdin:
        # Process job from stdin
        process_job_from_stdin()
    elif args.job_file:
        # Process specific job file
        process_specific_job(args.job_file)
    else:
        # Start processing job queue
        process_job_queue()



def process_job_queue():
    """Process jobs from the queue continuously."""
    queue_dir = Path("jobs/queue")
    processor = JobProcessor()
    
    print(f"👀 Watching job queue: {queue_dir}")
    print("   Press Ctrl+C to stop")
    
    try:
        while True:
            # Check for new job files
            job_files = list(queue_dir.glob("*.json"))
            
            if job_files:
                for job_file in job_files:
                    process_single_job(job_file, processor)
            
            # Wait before checking again
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n👋 Worker shutting down")
        sys.exit(0)

def process_single_job(job_file: Path, processor: JobProcessor):
    """Process a single job file."""
    job_id = job_file.stem
    
    try:
        print(f"\n🚀 Processing job: {job_id}")
        
        # Load job specification
        with open(job_file, 'r') as f:
            job_spec = json.load(f)
        
        # Update job status
        update_job_status(job_id, "processing", "Job started")
        
        # Process the job through the complete pipeline
        result = processor.process_job(job_spec)
        
        if result["success"]:
            print(f"✅ Job completed: {job_id}")
            
            # Move to completed
            completed_dir = Path("jobs/completed")
            completed_dir.mkdir(exist_ok=True)
            job_file.rename(completed_dir / job_file.name)
            
            # Update status
            update_job_status(job_id, "completed", "Job completed successfully")
            
        else:
            print(f"❌ Job failed: {job_id}")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            
            # Move to failed
            failed_dir = Path("jobs/failed")
            failed_dir.mkdir(exist_ok=True)
            job_file.rename(failed_dir / job_file.name)
            
            # Update status
            update_job_status(job_id, "failed", result.get('error', 'Unknown error'))
            
    except Exception as e:
        print(f"💥 Error processing {job_id}: {str(e)}")
        
        # Move to failed
        failed_dir = Path("jobs/failed")
        failed_dir.mkdir(exist_ok=True)
        job_file.rename(failed_dir / job_file.name)
        
        # Update status
        update_job_status(job_id, "failed", f"Processing error: {str(e)}")

def update_job_status(job_id: str, status: str, message: str):
    """Update job status file."""
    # Use project root jobs directory, not worker-relative
    status_file = Path("../jobs/status") / f"{job_id}.json"
    
    status_data = {
        "job_id": job_id,
        "status": status,
        "message": message,
        "timestamp": time.time()
    }
    
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)

def process_specific_job(job_file_path: str):
    """Process a specific job file directly."""
    job_file = Path(job_file_path)

    if not job_file.exists():
        print(f"❌ Job file not found: {job_file_path}")
        sys.exit(1)

    processor = JobProcessor()

    try:
        print(f"🚀 Processing job file: {job_file.name}")

        # Load job specification
        with open(job_file, 'r') as f:
            job_spec = json.load(f)

        job_id = job_spec.get('id', job_file.stem)

        # Update job status
        update_job_status(job_id, "processing", "Job started")

        # Process the job through the complete pipeline
        result = processor.process_job(job_spec)

        if result["success"]:
            print(f"✅ Job completed successfully: {job_id}")
            update_job_status(job_id, "completed", "Job completed successfully")
        else:
            print(f"❌ Job failed: {job_id}")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            update_job_status(job_id, "failed", result.get('error', 'Unknown error'))
            sys.exit(1)

    except Exception as e:
        print(f"💥 Error processing job: {str(e)}")
        update_job_status(job_id, "failed", f"Processing error: {str(e)}")
        sys.exit(1)

def process_job_from_stdin():
    """Process job data received from stdin."""
    try:
        print("📥 Reading job data from stdin...")

        # Read job spec from stdin
        job_data = sys.stdin.read()
        job_spec = json.loads(job_data)

        job_id = job_spec.get('id', 'stdin-job')
        print(f"🚀 Processing job: {job_id}")

        processor = JobProcessor()

        # Update job status
        update_job_status(job_id, "processing", "Job started")

        # Process the job through the complete pipeline
        result = processor.process_job(job_spec)

        if result["success"]:
            print(f"✅ Job completed successfully: {job_id}")
            update_job_status(job_id, "completed", "Job completed successfully")
        else:
            print(f"❌ Job failed: {job_id}")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            update_job_status(job_id, "failed", result.get('error', 'Unknown error'))
            sys.exit(1)

    except Exception as e:
        print(f"💥 Error processing job from stdin: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
