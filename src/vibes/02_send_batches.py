# send_batches.py
import argparse
import json
import os
import time
from pathlib import Path
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="redacted",
    azure_endpoint="https://prate-m1kg1tki-southindia.openai.azure.com/",
    api_version="2025-03-01-preview",
)


def upload_file(path):
    return client.files.create(file=open(path, "rb"), purpose="batch")


def create_batch_job(file_id, endpoint):
    return client.batches.create(
        input_file_id=file_id, endpoint=endpoint, completion_window="24h"
    )


def get_active_batches(log_file):
    """
    Get list of batch IDs that are currently active (not completed, failed, or cancelled).
    Returns dict mapping batch_id to jsonl_file.
    """
    if not os.path.exists(log_file):
        return {}
    
    active_batches = {}
    with open(log_file, "r") as lf:
        for line in lf:
            if not line.strip():
                continue
            entry = json.loads(line)
            batch_id = entry["batch"]["id"]
            jsonl_file = entry["jsonl_file"]
            
            # Check current status
            try:
                batch = client.batches.retrieve(batch_id)
                if batch.status in ("validating", "in_progress", "finalizing"):
                    active_batches[batch_id] = jsonl_file
            except Exception as e:
                print(f"Warning: Could not retrieve batch {batch_id}: {e}")
    
    return active_batches


def wait_for_batch_completion(batch_ids, out_dir, poll_interval=30, min_completions=1):
    """
    Wait for at least min_completions batches to complete and fetch their results.
    Returns number of batches completed.
    """
    print(f"\nWaiting for at least {min_completions} batch(es) to complete...")
    print(f"Monitoring {len(batch_ids)} active batches")
    
    completed_count = 0
    remaining_batches = set(batch_ids)
    
    while completed_count < min_completions and remaining_batches:
        print(f"\nPolling {len(remaining_batches)} remaining batches...")
        
        for batch_id in list(remaining_batches):
            try:
                batch = client.batches.retrieve(batch_id)
                print(f"Batch {batch_id}: {batch.status}")
                
                if batch.status == "completed":
                    save_batch_results(batch, out_dir)
                    remaining_batches.remove(batch_id)
                    completed_count += 1
                    
                elif batch.status in ("failed", "cancelled", "expired"):
                    print(f"✗ Batch {batch_id} ended with status: {batch.status}")
                    remaining_batches.remove(batch_id)
                    completed_count += 1  # Count as "resolved" to continue
                    
            except Exception as e:
                print(f"Error checking batch {batch_id}: {e}")
        
        if completed_count < min_completions and remaining_batches:
            print(f"Completed: {completed_count}/{min_completions}. Waiting {poll_interval}s...")
            time.sleep(poll_interval)
    
    print(f"\n✓ {completed_count} batch(es) completed/resolved")
    return completed_count


def save_batch_results(batch, out_dir):
    """Helper to download and save results for a single batch object."""
    out_path = os.path.join(out_dir, f"{batch.id}.jsonl")
    
    if not os.path.exists(out_path):
        # Download results
        if batch.output_file_id:
            out_content = client.files.content(batch.output_file_id)
            with open(out_path, "w") as f:
                f.write(out_content.text)
            print(f"✓ Downloaded results for batch {batch.id}")
        else:
            print(f"✗ No output file for batch {batch.id}")
        
        # Download errors if present
        if batch.error_file_id:
            err_content = client.files.content(batch.error_file_id)
            with open(f"{out_path}.errors.jsonl", "w") as f:
                f.write(err_content.text)
            print(f"  Downloaded errors for batch {batch.id}")
    else:
        print(f"  Results already exist for batch {batch.id}")


def download_missing_results(log_file, out_dir):
    """
    Iterates through the log file and downloads any missing output files.
    """
    print(f"\n{'='*60}")
    print("Final Phase: Checking logs for missing output files...")
    
    if not os.path.exists(log_file):
        print("No log file found.")
        return

    processed_ids = set()
    
    # Read all batch IDs from log
    with open(log_file, "r") as lf:
        for line in lf:
            if not line.strip(): continue
            try:
                entry = json.loads(line)
                batch_id = entry["batch"]["id"]
                processed_ids.add(batch_id)
            except:
                continue
    
    print(f"Checking {len(processed_ids)} historical batches...")
    
    for batch_id in processed_ids:
        # Check if file already exists
        out_path = os.path.join(out_dir, f"{batch_id}.jsonl")
        
        if not os.path.exists(out_path):
            try:
                # Retrieve batch status
                batch = client.batches.retrieve(batch_id)
                
                if batch.status == "completed":
                    print(f"Found completed batch {batch_id} missing local file. Downloading...")
                    save_batch_results(batch, out_dir)
                elif batch.status in ("failed", "cancelled", "expired"):
                    # Optional: Create a placeholder or error file so we don't check it again?
                    # For now, just logging it.
                    print(f"Batch {batch_id} is {batch.status} (no output to download).")
                else:
                    print(f"Batch {batch_id} is still {batch.status}.")
            except Exception as e:
                print(f"Could not retrieve/download batch {batch_id}: {e}")

    print("✓ Final check complete.")


def is_enqueue_limit_error(error):
    """
    Check if the error is related to enqueue token limits.
    """
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in [
        "enqueue",
        "queue",
        "rate limit",
        "rate_limit",
        "too many requests",
        "quota",
        "token_limit_exceeded",
        "token limit exceeded"
    ])


def process_batches(
    jsonl_dir,
    endpoint="/v1/responses",
    log_file="/data/joy/logs/batches_created.jsonl",
    out_dir="batch_results",
    limit=None,
    poll_interval=60,
    max_retries=40,
    wait_for_completions=2
):
    """
    Callable function to process batches using arguments directly.
    """
    # Make sure paths exist
    os.makedirs(jsonl_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Load already processed files
    processed = set()
    if os.path.exists(log_file):
        with open(log_file, "r") as lf:
            for line in lf:
                if line.strip():
                    try:
                        processed.add(json.loads(line)["jsonl_file"])
                    except Exception:
                        pass

    # Get list of files to process
    jsonl_files = sorted(Path(jsonl_dir).glob("*.jsonl"))
    if limit is not None:
        jsonl_files = jsonl_files[:limit]
    
    print(f"Found {len(jsonl_files)} JSONL files")
    print(f"Already processed: {len(processed)}")
    
    count = 0
    retry_count = 0
    
    for path in jsonl_files:
        if str(path) in processed:
            print(f"Skipping {path} (already processed)")
            continue

        # Try to submit the batch
        submitted = False
        while not submitted and retry_count < max_retries:
            try:
                print(f"\n{'='*60}")
                print(f"Uploading {path}…")
                uploaded = upload_file(path)
                print(f"Uploaded: {uploaded.id}")
                
                print(f"Creating batch job for {path}…")
                batch = create_batch_job(uploaded.id, endpoint)
                print(f"✓ Batch created: {batch.id} (status: {batch.status})")
                
                # Log the batch
                with open(log_file, "a") as lf:
                    lf.write(
                        json.dumps({"jsonl_file": str(path), "batch": batch.model_dump()})
                        + "\n"
                    )
                
                count += 1
                submitted = True
                retry_count = 0  # Reset retry count on success
                
            except Exception as e:
                # Check if this is an enqueue limit error (regardless of exception type)
                if is_enqueue_limit_error(e):
                    print(f"\n⚠ Enqueue limit reached: {e}, tried {retry_count}/{max_retries} times")
                    retry_count += 1
                    
                    if retry_count >= max_retries:
                        print(f"✗ Max retries ({max_retries}) reached. Stopping submission loop.")
                        break
                    
                    # Get active batches
                    active_batches = get_active_batches(log_file)
                    
                    if not active_batches:
                        print("No active batches found. Waiting before retry...")
                        time.sleep(poll_interval)
                        continue
                    
                    print(f"Active batches: {len(active_batches)}")
                    
                    # Wait for some batches to complete
                    completed = wait_for_batch_completion(
                        list(active_batches.keys()),
                        out_dir,
                        poll_interval,
                        wait_for_completions
                    )
                    
                    if completed > 0:
                        print(f"\n✓ Freed up queue space. Retrying submission...")
                    else:
                        print(f"\n⚠ No batches completed. Waiting before retry...")
                        time.sleep(poll_interval)
                        
                else:
                    # Not an enqueue error - this is a real problem
                    print(f"✗ Error processing {path}: {e}")
                    print("This is not an enqueue limit error. Skipping this file...")
                    break  # Skip to next file

    print(f"\n{'='*60}")
    print(f"✓ Successfully submitted {count} new batch(es)")
    
    # REQUIREMENT 1: Iterate log file and download missing output files
    download_missing_results(log_file, out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_dir", required=True)
    parser.add_argument("--endpoint", default="/v1/responses")
    parser.add_argument("--log_file", default="/data/joy/logs/batches_created.jsonl")
    parser.add_argument("--out_dir", default="batch_results", help="Directory to store batch results")
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of batch files to process"
    )
    parser.add_argument(
        "--poll_interval", type=int, default=60, help="Seconds to wait between status checks"
    )
    parser.add_argument(
        "--max_retries", type=int, default=40, help="Maximum number of times to retry after enqueue limit"
    )
    parser.add_argument(
        "--wait_for_completions", type=int, default=2, help="Number of batches to wait for when hitting enqueue limit"
    )
    args = parser.parse_args()

    # Call the processing function with parsed arguments
    process_batches(
        jsonl_dir=args.jsonl_dir,
        endpoint=args.endpoint,
        log_file=args.log_file,
        out_dir=args.out_dir,
        limit=args.limit,
        poll_interval=args.poll_interval,
        max_retries=args.max_retries,
        wait_for_completions=args.wait_for_completions
    )


if __name__ == "__main__":
    main()