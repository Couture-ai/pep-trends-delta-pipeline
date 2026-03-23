# fetch_batches.py
import argparse
import json
import os
import time

from openai import AzureOpenAI

client = AzureOpenAI(
    api_key="NONE",
    azure_endpoint="NONE",
    api_version="NONE",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", default="batches_created.jsonl")
    parser.add_argument("--out_dir", default="batch_results")
    parser.add_argument("--poll_interval", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.log_file) as lf:
        entries = [json.loads(line) for line in lf if line.strip()]

    for e in entries:
        bid = e["batch"]["id"]
        out_path = os.path.join(args.out_dir, f"{bid}.jsonl")

        if os.path.exists(out_path):
            print(f"Results already downloaded for batch {bid}")
            continue

        print(f"Polling status for batch {bid}…")
        while True:
            batch = client.batches.retrieve(bid)
            print(f"Status: {batch.status}")

            if batch.status == "completed":
                if batch.output_file_id:
                    # Download success output
                    out_content = client.files.content(batch.output_file_id)
                    print(out_content.content)
                    with open(out_path, "w") as f:
                        f.write(out_content.text)

                    print(f"Downloaded outputs for batch {bid}")
                else:
                    print(f"No output file found for batch {bid}. Some error occurred.")

                # Optionally handle errors file similarly
                print(f"Error file ID: {batch.error_file_id}")
                if batch.error_file_id:
                    err_content = client.files.content(batch.error_file_id)
                    with open(f"{out_path}.errors.jsonl", "w") as f:
                        f.write(err_content.text)

                break
            elif batch.status in ("failed", "cancelled"):
                print(
                    f"Batch {bid} ended with status {batch.status}. More details: \n{batch.model_dump()}"
                )
                break

            elif batch.status in ("in_progress", "finalizing", "validating"):
                time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
