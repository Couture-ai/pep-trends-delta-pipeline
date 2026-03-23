import argparse
import base64
import json
import mimetypes
import os
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
import time
from typing import Dict, Literal, Optional

import pandas as pd
from PIL import Image, UnidentifiedImageError

from prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    TREND_TAGS_SCRORE_PROMPT,
    USER_INSTRUCTION_TEMPLATE,
    BASELINE_TAGS,
    category_prompt_map,
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
UNKNOWN_MIME_LOG = f"unknown_mime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
BAD_IMAGE_LOG = f"bad_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


def to_data_url(path: str) -> Optional[str]:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/jpeg"
        with open(UNKNOWN_MIME_LOG, "a", encoding="utf-8") as f:
            f.write(path + "\n")

    try:
        img = Image.open(path).convert("RGB")
    except UnidentifiedImageError:
        with open(BAD_IMAGE_LOG, "a", encoding="utf-8") as f:
            f.write(path + "\n")
        return None
    except Exception as e:
        with open(BAD_IMAGE_LOG, "a", encoding="utf-8") as f:
            f.write(f"{path} | {repr(e)}\n")
        return None

    img = img.resize((384, 384))
    buffer = BytesIO()
    img.save(buffer, format=mime.split("/")[-1].upper() if "/" in mime else "JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_product_id(image_path: str) -> Optional[str]:
    """Extract product_id from image filename by removing the extension."""
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    return name_without_ext


def get_trending_tags(brick_name: str):
    with open("trending_tags.json", "r") as f:
        trend_tags_data = json.load(f)

    return {
        "trending_tags_india": trend_tags_data.get(brick_name, {}).get(
            "trending_tags_india", []
        ),
        "trending_tags_global": trend_tags_data.get(brick_name, {}).get(
            "trending_tags_global", []
        ),
    }


def make_line(
    image_path: str,
    brick_name: str,
    model: str,
    use_data_url: bool,
    task: Literal["trend_tags_score", "full_tagging"],
):
    if use_data_url:
        data_url = to_data_url(image_path)
        if data_url is None:
            return None  # skip bad image
        image_payload = {"type": "input_image", "image_url": f"{data_url}"}
        # image_payload = {"type": "image_url", "image_url": {"url": data_url}}

    else:
        image_payload = {"type": "input_image", "image_url": f"file://{image_path}"}

    image_id = os.path.abspath(image_path)

    if task == "full_tagging":
        # Create prompts with brick context using LangChain templates
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format()
        user_instruction = USER_INSTRUCTION_TEMPLATE.format(
            brick_name=brick_name if brick_name else "Unknown Brick",
            brick_specific_attributes = category_prompt_map[brick_name] if brick_name in category_prompt_map else BASELINE_TAGS
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_instruction},
                    image_payload,
                ],
            },
        ]
    # elif task == "trend_tags_score":
    #     user_instruction = TREND_TAGS_SCRORE_PROMPT.format(
    #         brick_name=brick_name if brick_name else "Unknown Brick",
    #         trend_tags=json.dumps(
    #             get_trending_tags(brick_name) if brick_name else {}, indent=2
    #         ),
    #     )

    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "input_text", "text": user_instruction},
    #                 image_payload,
    #             ],
    #         },
    #     ]

    return {
        "custom_id": image_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": messages,
            "reasoning":{"effort": "minimal"},
            # "response_format": {"type": "json_object"},
            "max_output_tokens": 15360,
            # "temperature": 0.3,
        },
    }


def is_batch_complete(parent_folder: str, batch_num: int) -> bool:
    """
    Check if a batch is complete by verifying that the next batch folder exists.
    A batch is considered complete if batch_{batch_num + 1} folder exists.
    """
    next_batch_folder = os.path.join(parent_folder, f"batch_{batch_num + 1}")
    return os.path.isdir(next_batch_folder)


def get_images_in_batch_folder(parent_folder: str, batch_num: int):
    """
    Get all images in a specific batch folder.
    """
    batch_folder = os.path.join(parent_folder, f"batch_{batch_num}")
    if not os.path.isdir(batch_folder):
        return []
    
    images = []
    for ext in SUPPORTED_EXTS:
        images.extend(Path(batch_folder).glob(f"*{ext}"))
        images.extend(Path(batch_folder).glob(f"*{ext.upper()}"))
    
    return [str(img) for img in images]


def get_complete_batches(parent_folder: str, max_batch: Optional[int] = None):
    """
    Get list of complete batch numbers by checking folder structure.
    Returns list of batch numbers that are complete (i.e., next batch folder exists).
    """
    if not os.path.isdir(parent_folder):
        print(f"Warning: Parent folder {parent_folder} does not exist")
        return []
    
    # Find all batch folders
    batch_folders = []
    for item in os.listdir(parent_folder):
        if item.startswith("batch_") and os.path.isdir(os.path.join(parent_folder, item)):
            try:
                batch_num = int(item.replace("batch_", ""))
                batch_folders.append(batch_num)
            except ValueError:
                continue
    
    batch_folders.sort()
    
    if not batch_folders:
        print("No batch folders found in parent directory")
        return []
    
    # Determine complete batches (all except the last one, unless max_batch is specified)
    complete_batches = []
    for batch_num in batch_folders:
        if max_batch is not None and batch_num > max_batch:
            print(f"max_batch_num \t {batch_num}")
            break
        if max_batch is not None:
            complete_batches.append(batch_num)
        elif is_batch_complete(parent_folder, batch_num):
            complete_batches.append(batch_num)
    
    return complete_batches


def make_batch_jsonl(
    dataframe_path: str,
    parent_folder: str,
    model: str = "gpt-5",
    out_dir: str = "jsonl_shards",
    use_data_url: bool = True,
    task: Literal["trend_tags_score", "full_tagging"] = "full_tagging",
    max_batch: Optional[int] = None,
):
    """
    Callable function that executes the original script logic.
    Parameters mirror the CLI arguments that were formerly parsed inside main().
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load dataframe
    df = pd.read_csv(dataframe_path)
    print(f"Loaded dataframe with {len(df)} rows")
    
    # Verify required columns
    required_cols = ['image_path', 'category', 'batch_number']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Dataframe must contain '{col}' column. Found columns: {df.columns.tolist()}")

    # Get complete batches from folder structure
    complete_batches = get_complete_batches(parent_folder, max_batch)
    print(f"Found {len(complete_batches)} complete batches: {complete_batches}")
    
    if not complete_batches:
        print("No complete batches to process. Exiting.")
        return

    total_processed = 0
    batches_processed = 0


    print(len(complete_batches), complete_batches[-1])
    time.sleep(10)
    # Process each complete batch
    for batch_num in complete_batches:
        print(f"\n{'='*60}")
        print(batch_num)
        if os.path.exists(os.path.join(out_dir, f"batch_{batch_num:03d}.jsonl")):
            print(f"skipping batch {batch_num}")
            continue
        else:
            print(f"Processing batch {batch_num}")
            print(f"{'='*60}")
        
        # Filter dataframe for this batch
        batch_df = df[df['batch_number'] == batch_num].copy()
        
        if len(batch_df) == 0:
            print(f"No rows in dataframe for batch {batch_num}, skipping...")
            continue
        
        # Get actual images in the batch folder
        batch_images = get_images_in_batch_folder(parent_folder, batch_num)
        print(f"Found {len(batch_images)} images in batch_{batch_num} folder")
        
        # Create a set of image filenames for quick lookup
        image_filenames = {os.path.basename(img) for img in batch_images}
        
        # Filter batch_df to only include rows where the image file actually exists
        batch_df['image_filename'] = batch_df['image_path'].apply(os.path.basename)
        batch_df = batch_df[batch_df['image_filename'].isin(image_filenames)]
        
        print(f"Dataframe has {len(batch_df)} rows with downloaded images for batch {batch_num}")
        
        if len(batch_df) == 0:
            print(f"No matching images found for batch {batch_num}, skipping...")
            continue
        # if len(batch_df) <1180:
        #     print("too many images missing, skipping")
        #     break
        
        # Create JSONL file for this batch
        output_file = os.path.join(out_dir, f"batch_{batch_num:03d}.jsonl")
        
        count = 0
        with open(output_file, "w", encoding="utf-8") as out_f:
            for _, row in batch_df.iterrows():
                image_filename = row['image_filename']
                # Construct full path to image in batch folder
                image_path = os.path.join(parent_folder, f"batch_{batch_num}", image_filename)
                
                # Verify image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue
                
                brick_name = row['category']
                
                line = make_line(image_path, brick_name, model, use_data_url, task)
                if line is None:
                    continue
                    
                out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
                count += 1
        
        print(f"Wrote {count} requests to {output_file}")
        total_processed += count
        batches_processed += 1

    print(f"\n{'='*60}")
    print(f"Summary: Processed {batches_processed} batches with {total_processed} total requests")
    print(f"{'='*60}")
    
    if os.path.exists(UNKNOWN_MIME_LOG):
        print(f"Unknown MIME types logged to {UNKNOWN_MIME_LOG}")
    if os.path.exists(BAD_IMAGE_LOG):
        print(f"Bad/corrupted images logged to {BAD_IMAGE_LOG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataframe_path", required=True, help="Path to CSV with image_path, category, and batch_number columns")
    ap.add_argument("--parent_folder", required=True, help="Parent folder containing batch_N subfolders with images")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--out_dir", default="jsonl_shards")
    ap.add_argument("--use_data_url", action="store_true")
    ap.add_argument(
        "--task", choices=["trend_tags_score", "full_tagging"], default="full_tagging"
    )
    ap.add_argument("--max_batch", type=int, default=None, help="Maximum batch number to process")
    args = ap.parse_args()

    main(
        dataframe_path=args.dataframe_path,
        parent_folder=args.parent_folder,
        model=args.model,
        out_dir=args.out_dir,
        use_data_url=args.use_data_url,
        task=args.task,
        max_batch=args.max_batch,
    )
