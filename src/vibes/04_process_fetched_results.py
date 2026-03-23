from multiprocessing import process
import pandas as pd
import numpy as np
import os
import json
import argparse
from textwrap import dedent
from tqdm import tqdm
from datetime import datetime

import os
import math
import pandas as pd

def save_df_in_parquet_parts(df: pd.DataFrame, out_dir: str, n_parts: int) -> None:
    """
    Partition a pandas DataFrame into `n_parts` row-wise segments and
    write each segment as an individual Parquet file inside `out_dir`.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    out_dir : str
        Path to the directory where the parquet parts should be written.
    n_parts : int
        Number of parquet parts to create.

    Notes
    -----
    The final part may contain fewer rows if the total row count
    is not exactly divisible by `n_parts`.
    """

    os.makedirs(out_dir, exist_ok=True)
    total_rows = len(df)
    if total_rows == 0 or n_parts < 1:
        return

    part_size = math.ceil(total_rows / n_parts)
    print(f"writing responses to output dir: {out_dir}")
    for i in tqdm(range(n_parts)):
        start = i * part_size
        end = min(start + part_size, total_rows)
        if start >= total_rows:
            break

        part = df.iloc[start:end]
        part_path = os.path.join(out_dir, f"part_{i:05d}.parquet")
        part.to_parquet(part_path, index=False)

def add_product_info(processed_df, metadata_path):
    metadata = pd.read_csv(metadata_path)
    processed_df["product_id"] = processed_df["custom_id"].apply(lambda x: x.split("/")[-1].replace(".jpg", ""))
    processed_df = processed_df.merge(metadata[["product_id", "category"]], on="product_id", how="left")
    print(processed_df.info())
    return processed_df

def flatten_product_tags(data_list):
    """
    Convert nested product tag data into a flat pandas DataFrame.
    
    Args:
        data_list: List of dictionaries with the structure shown in your example
    
    Returns:
        pandas DataFrame with columns: custom_id, type, name, confidence, stv_list
    """
    rows = []
    print("Processing loaded responses")
    for item in tqdm(data_list):
        custom_id = item['custom_id']
        
        # # Convert shop_the_vibe list to a simple list of names
        # try:
        #     stv_list = [stv['name'] for stv in item['content']['shop_the_vibe']]
        # except:
        #     stv_list = []
        # Flatten each product tag
        
        
        # add stv rows separately
        try:
            for tag in item["content"]["shop_the_vibe"]:
                rows.append({
                        'custom_id': custom_id,
                        'type': "shop_the_vibe",
                        'name': tag['name'],
                        'confidence': tag['confidence'],
                        # 'stv_list': stv_list  # Same list for all tags from this product
                    })
        except:
            pass

        try:
            for tag in item['content']['product_tags']:
                rows.append({
                    'custom_id': custom_id,
                    'type': tag['type'],
                    'name': tag['name'],
                    'confidence': tag['confidence'],
                    # 'stv_list': stv_list  # Same list for all tags from this product
                })
        except:
            continue
    
    return pd.DataFrame(rows)

def process_response_files(response_dir):
    processed_files = []
    results = []
    num_files = 0
    file_list = os.listdir(response_dir)
    valid_files = [f for f in file_list if f.endswith(".jsonl") and not f.endswith("errors.jsonl")]
    print("loading responses")
    for file in tqdm(valid_files):
        if file.endswith(".jsonl") and not file.endswith("errors.jsonl"):
            num_files += 1
            processed_files.append(file)
            inp_file = f"{response_dir}/{file}"
            with open(inp_file, "r") as file:
                for line in file:
                    if line.strip():  # Skip empty lines
                        data = json.loads(line)

                        # Extract the custom_id and content
                        custom_id = data.get("custom_id", "")

                        # Get the content from the response
                        if "response" in data and "body" in data["response"]:
                            choices = data["response"]["body"].get("choices", [])
                            if choices and len(choices) > 0:
                                content = choices[0].get("message", {}).get("content", "")

                                # Parse the content as JSON if it's a JSON string
                                try:
                                    parsed_content = json.loads(content)
                                    result = {
                                        "custom_id": custom_id,
                                        "content": parsed_content,
                                    }
                                except json.JSONDecodeError:
                                    # If content is not JSON, keep it as string
                                    result = {"custom_id": custom_id, "content": content}

                                results.append(result)
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Extract content from batch results JSONL file"
    )
    parser.add_argument(
        "--response_dir", required=True, help="Path to the batch results JSONL file"
    )
    parser.add_argument(
        "--processed_dataframe_dir", required=True, help="Path for the output JSON file"
    )
    parser.add_argument(
        "--product_metadata_path", required=True, help="Path for the csv containing product_id and category."
    )

    args = parser.parse_args()

    # Check if batch file exists
    if not os.path.exists(args.response_dir):
        print(f"Error: Batch file not found: {args.response_dir}")
        return

    # Extract content
    results = process_response_files(args.response_dir)
    processed_df = flatten_product_tags(results)
    processed_df = processed_df[processed_df["confidence"]>0.7]
    processed_df_with_category = add_product_info(processed_df, args.product_metadata_path)
    save_df_in_parquet_parts(processed_df_with_category, args.processed_dataframe_dir, 100)

if __name__ == "__main__":
    main()