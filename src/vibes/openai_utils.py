import json
import os
import time
from pathlib import Path
from openai import AzureOpenAI

class AzureBatchHandler:
    def __init__(
        self, 
        api_key, 
        endpoint, 
        api_version="2025-03-01-preview",
        log_file="batches_created.jsonl",
        out_dir="batch_results"
    ):
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self.log_file = log_file
        self.out_dir = out_dir
        
        os.makedirs(os.path.dirname(self.log_file) or '.', exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

    def generate_response(self, prompt, model_name="gpt-5", max_tokens=1000, temperature=0.7):
        """
        Processes a single request immediately.
        Usage: handler.generate_response("Your prompt here")
        """
        # print(prompt)
        try:
            response = self.client.responses.create(
                model=model_name,
                input=[{"role": "user", "content": prompt}],
                # max_completion_tokens=max_tokens
            )
            return response.model_dump_json()
        except Exception as e:
            print(f"Error in single call: {e}")
            # return response.model_dump_json()

    def format_prompts_to_jsonl(self, prompt_dict, model_name, output_path):
        """Converts {id: prompt} or {id: {'text': ..., 'image_base64': ...}} to JSONL."""
        with open(output_path, "w") as f:
            print(f"opened {f}")
            for custom_id, data in prompt_dict.items():
                if isinstance(data, dict) and "image_base64" in data:
                    content = [
                        {"type": "text", "text": data.get("text", "")},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{data['image_base64']}"}
                        }
                    ]
                else:
                    content = data

                entry = {
                    "custom_id": str(custom_id),
                    "method": "POST",
                    "url": "/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": 1000
                    }
                }
                f.write(json.dumps(entry) + "\n")
        return output_path

    def process_existing_jsonl_dir(self, jsonl_dir, limit=None, poll_interval=60):
        """
        Reads a directory of pre-formatted JSONL files and submits them.
        Skips files already found in the log_file.
        """
        # 1. Load already processed files from log
        processed_files = set()
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as lf:
                for line in lf:
                    if line.strip():
                        try:
                            processed_files.add(json.loads(line)["jsonl_file"])
                        except: continue

        # 2. Find all .jsonl files in dir
        jsonl_files = sorted([str(p) for p in Path(jsonl_dir).glob("*.jsonl")])
        if limit:
            jsonl_files = jsonl_files[:limit]

        print(f"Found {len(jsonl_files)} files. {len(processed_files)} already in logs.")

        # 3. Submit them using the existing robust logic
        submitted_count = 0
        for file_path in jsonl_files:
            if file_path in processed_files:
                print(f"Skipping {file_path} (already logged)")
                continue
            
            print(f"\nSubmitting existing file: {file_path}")
            self.run_batch_process(file_path, poll_interval=poll_interval)
            submitted_count += 1
            
        return submitted_count

    def upload_file(self, path):
        return self.client.files.create(file=open(path, "rb"), purpose="batch")

    def create_batch_job(self, file_id, endpoint="/v1/chat/completions"):
        return self.client.batches.create(
            input_file_id=file_id, endpoint=endpoint, completion_window="24h"
        )

    def get_active_batches(self):
        if not os.path.exists(self.log_file): return {}
        active = {}
        with open(self.log_file, "r") as lf:
            for line in lf:
                if not line.strip(): continue
                entry = json.loads(line)
                bid = entry["batch"]["id"]
                try:
                    batch = self.client.batches.retrieve(bid)
                    if batch.status in ("validating", "in_progress", "finalizing"):
                        active[bid] = entry["jsonl_file"]
                except: continue
        return active

    def save_batch_results(self, batch):
        out_path = os.path.join(self.out_dir, f"{batch.id}.jsonl")
        if not os.path.exists(out_path):
            if batch.output_file_id:
                content = self.client.files.content(batch.output_file_id)
                with open(out_path, "w") as f: f.write(content.text)
                print(f"✓ Saved results for {batch.id}")
            if batch.error_file_id:
                err = self.client.files.content(batch.error_file_id)
                with open(f"{out_path}.errors.jsonl", "w") as f: f.write(err.text)
        else:
            print(f"  Results exist for {batch.id}")

    def run_batch_process(self, jsonl_path, max_retries=40, poll_interval=60):
        retry_count = 0
        while retry_count < max_retries:
            try:
                uploaded = self.upload_file(jsonl_path)
                batch = self.create_batch_job(uploaded.id)
                with open(self.log_file, "a") as lf:
                    lf.write(json.dumps({"jsonl_file": str(jsonl_path), "batch": batch.model_dump()}) + "\n")
                return batch
            except Exception as e:
                if any(k in str(e).lower() for k in ["enqueue", "rate_limit", "quota"]):
                    retry_count += 1
                    print(f"Queue full. Waiting for capacity...")
                    active = self.get_active_batches()
                    if active:
                        # Wait for at least one to finish to make room
                        self.wait_for_completion(list(active.keys()), poll_interval, 1)
                    else:
                        time.sleep(poll_interval)
                else: raise e

    def wait_for_completion(self, batch_ids, poll_interval=60, min_completions=1):
        completed_count = 0
        remaining = set(batch_ids)
        while completed_count < min_completions and remaining:
            for bid in list(remaining):
                batch = self.client.batches.retrieve(bid)
                if batch.status == "completed":
                    self.save_batch_results(batch)
                    remaining.remove(bid)
                    completed_count += 1
                elif batch.status in ("failed", "cancelled", "expired"):
                    remaining.remove(bid)
                    completed_count += 1
            if completed_count < min_completions and remaining:
                time.sleep(poll_interval)

    def wait_for_all_and_download(self, poll_interval=60):
        while True:
            active = self.get_active_batches()
            if not active: break
            self.wait_for_completion(list(active.keys()), poll_interval, 1)