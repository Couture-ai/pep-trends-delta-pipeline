# # image_downloader.py
# from pathlib import Path
# from multiprocessing.pool import ThreadPool
# import pandas as pd
# from typing import Tuple, Dict, Any, Optional
# import requests

# # Module-level variable consumed by the worker to exactly preserve original download logic.
# IMAGE_PATH: Optional[Path] = None


# def download_single_image(row_tuple: Tuple[int, str, int, str]) -> str:
#     """
#     IDENTICAL download logic to the original script.
#     Accepts a tuple: (idx, productid, batch_number, image_url)
#     Returns one of: 'success', 'skipped', or 'failed: <message>'.
#     """
#     # Unpack arguments exactly as in the original
#     idx, productid, batch_number, image_url = row_tuple

#     # Use module-level IMAGE_PATH (set by download_images) to preserve original behavior.
#     global IMAGE_PATH
#     if IMAGE_PATH is None:
#         return f"failed: IMAGE_PATH not set"

#     batch_folder = IMAGE_PATH / f"batch_{batch_number}"
#     file_path = batch_folder / f"{productid}.jpg"

#     # Skip if already exists (identical behaviour)
#     if file_path.exists():
#         return 'skipped'

#     # Create batch folder (identical behaviour)
#     batch_folder.mkdir(parents=True, exist_ok=True)

#     try:
#         # Exactly the same HTTP call and timeout as in the original script
#         response = requests.get(image_url, timeout=10)
#         if response.status_code == 200:
#             with open(file_path, 'wb') as f:
#                 f.write(response.content)
#             return 'success'
#         else:
#             return f'failed: HTTP {response.status_code}'
#     except Exception as e:
#         return f'failed: {str(e)}'


# def download_images(
#     dataframe_path: str | Path,
#     image_dir: str | Path,
#     image_url_col: str = "img-473Wx593H_string",
#     product_id_col: str = "productid",
#     batch_col: str = "batch_number",
#     batch_size: int = 1000,
#     num_threads: int = 50,
#     progress_interval: int = 1000,
# ) -> Tuple[Dict[str, Any], pd.DataFrame]:
#     """
#     Read a dataframe and download images into batch subfolders using the original
#     download logic unchanged.

#     Parameters
#     ----------
#     dataframe_path : str | Path
#         Path to the dataframe file. Supports parquet and csv heuristically.
#     image_dir : str | Path
#         Destination directory; batch subfolders are created as `batch_<n>`.
#     image_url_col : str
#         Column name that contains image URLs. (required)
#     product_id_col : str
#         Column name used to name files (required)
#     batch_col : str
#         Column name for batch number. If absent, batch numbers will be assigned.
#     batch_size : int
#         When batch_col is absent, assign batch_size rows per batch (default 1000).
#     num_threads : int
#         Number of threads for ThreadPool (default 50).
#     progress_interval : int
#         Print progress every `progress_interval` items (default 1000).

#     Returns
#     -------
#     summary : dict
#         Counters for total, processed, success, skipped, failed
#     status_df : pandas.DataFrame
#         Original dataframe augmented with `download_status` (and `download_error` where applicable).
#     """

#     # Resolve and create base image directory
#     dataframe_path = Path(dataframe_path).expanduser()
#     image_path = Path(image_dir).expanduser()
#     image_path.mkdir(parents=True, exist_ok=True)

#     # Load dataframe: prefer parquet if extension suggests, otherwise attempt both.
#     if not dataframe_path.exists():
#         raise FileNotFoundError(f"Dataframe file not found: {dataframe_path}")

#     suffix = dataframe_path.suffix.lower()
#     if suffix in (".parquet", ".parq"):
#         df = pd.read_parquet(dataframe_path)
#     elif suffix in (".csv", ".txt"):
#         df = pd.read_csv(dataframe_path)
#     else:
#         # Best-effort: try parquet then csv
#         try:
#             df = pd.read_parquet(dataframe_path)
#         except Exception:
#             df = pd.read_csv(dataframe_path)

#     # Validate required columns
#     missing = [c for c in (product_id_col, image_url_col) if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing required columns in dataframe: {missing}")

#     # Ensure deterministic index for batch assignment
#     df = df.reset_index(drop=True)

#     # If batch column missing, assign batches deterministically (batch numbers start at 1)
#     if batch_col not in df.columns:
#         if batch_size <= 0:
#             raise ValueError("batch_size must be a positive integer when batch column is absent.")
#         df[batch_col] = (df.index // int(batch_size)) + 1
#     else:
#         # Attempt to coerce to integer; if it fails, fall back to deterministic assignment
#         try:
#             df[batch_col] = df[batch_col].astype(int)
#         except Exception:
#             df[batch_col] = (df.index // int(batch_size)) + 1

#     # Prepare tuples for the worker (must match worker unpacking)
#     data_tuples = [
#         (int(idx), row[product_id_col], int(row[batch_col]), row[image_url_col])
#         for idx, row in df.iterrows()
#     ]

#     # Set module-level IMAGE_PATH so the top-level worker uses it without changing logic
#     global IMAGE_PATH
#     IMAGE_PATH = image_path

#     # Counters (mirroring the original script's behaviour)
#     total_processed = 0
#     success_count = 0
#     skipped_count = 0
#     failed_count = 0
#     total = len(data_tuples)

#     # Prepare per-row status containers
#     status_series = pd.Series(index=df.index, dtype="object")
#     error_series = pd.Series(index=df.index, dtype="object")

#     print(f"Downloading {total} images with {num_threads} threads...")
#     print(f"Progress updates every {progress_interval} images\n")

#     # Use ThreadPool and the identical-imap_unordered call as the original script
#     with ThreadPool(processes=int(num_threads)) as pool:
#         for result in pool.imap_unordered(download_single_image, data_tuples):
#             total_processed += 1

#             # result is one of: 'success', 'skipped', or 'failed: ...'
#             if result == 'success':
#                 success_count += 1
#             elif result == 'skipped':
#                 skipped_count += 1
#             else:
#                 failed_count += 1

#             # Infer index from data_tuples order: find mapping by idx
#             # (we could map idx->position by construction; use small dict for speed)
#             # Build mapping once outside loop would be more efficient; for clarity do it now:
#             # Create a lightweight mapping from idx to tuple position
#             # (Note: idx is the first element of each tuple)
#             # To avoid O(n^2) lookup here, produce mapping prior to pool loop.
#             # But since this is an orchestration wrapper and intended for typical usage,
#             # construct the mapping once below and reuse.

#             # (We'll lazily create mapping on first use)
#             if '___idx_map' not in locals():
#                 idx_map = {t[0]: pos for pos, t in enumerate(data_tuples)}
#                 # create lists of indices to feed back into the df
#                 tuple_indices = [t[0] for t in data_tuples]

#             # We need the idx associated with this result; but pool.imap_unordered
#             # returns only the worker return value, so we must instead run the worker
#             # returns without idx (original code did not capture per-row status)
#             # To attach per-row status here without altering download logic we re-run with knowledge:
#             # The original download_single_image returns only status string; to preserve the identical
#             # worker signature while still mapping status back we require that data_tuples and
#             # results iterate in the same order — they do not when unordered. Therefore, rather than
#             # attempt to maintain exact per-row alignment here, we will record summary counters
#             # exactly and attach status naively as None for per-row entries (preserving original logic).
#             # This keeps the download logic truly unchanged; per-row status would require changing worker signature.
#             #
#             # For now, do not attempt to map result to specific df row (to avoid changing the worker).
#             # Instead we only maintain the aggregate counters exactly as original script did.

#             if (total_processed % int(progress_interval) == 0):
#                 print(f"Processed: {total_processed}/{total} | Success: {success_count} | Skipped: {skipped_count} | Failed: {failed_count}")

#     # Final progress print (as original script)
#     print(f"\n{'='*60}")
#     print(f"Download Complete!")
#     print(f"Total Processed: {total_processed}")
#     print(f"Success: {success_count}")
#     print(f"Skipped: {skipped_count}")
#     print(f"Failed: {failed_count}")
#     print(f"{'='*60}")

#     # Construct summary
#     summary = {
#         "total": total,
#         "processed": total_processed,
#         "success": success_count,
#         "skipped": skipped_count,
#         "failed": failed_count,
#     }

#     # The original script did not record per-row statuses. To avoid altering the internal
#     # download logic, we attach an empty download_status column (None) so callers can opt-in
#     # to post-process logging if desired.
#     df = df.copy()
#     df["download_status"] = None
#     df["download_error"] = None

#     return summary, df


# image_downloader.py
from pathlib import Path
from multiprocessing.pool import ThreadPool
import pandas as pd
import requests
import threading
from typing import Tuple, Dict, Any, Optional

# --- Unmodified download worker (textually equivalent to original script) ---
def download_single_image(row_tuple: Tuple[int, str, int, str]) -> str:
    """
    IDENTICAL download logic to the original script.
    Accepts a tuple: (idx, productid, batch_number, image_url)
    Returns one of: 'success', 'skipped', or 'failed: <message>'.
    """
    idx, productid, batch_number, image_url = row_tuple

    # IMAGE_PATH will be set by download_images before pool starts
    global IMAGE_PATH
    if IMAGE_PATH is None:
        return f"failed: IMAGE_PATH not set"

    batch_folder = IMAGE_PATH / f"batch_{batch_number}"
    file_path = batch_folder / f"{productid}.jpg"

    # Skip if already exists
    if file_path.exists():
        return 'skipped'

    # Create batch folder
    batch_folder.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return 'success'
        else:
            return f'failed: HTTP {response.status_code}'
    except Exception as e:
        return f'failed: {str(e)}'


# Module-level IMAGE_PATH used by the unchanged worker
IMAGE_PATH: Optional[Path] = None


# --- Wrapper function to be imported/used by callers ---
def download_images(
    dataframe_path: str | Path,
    image_dir: str | Path,
    image_url_col: str = "img-473Wx593H_string",
    product_id_col: str = "productid",
    batch_col: str = "batch_number",
    batch_size: int = 1000,
    num_threads: int = 50,
    progress_interval: int = 1000,
    category_col: str = "l1l3category_en_string_mv"
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Read a dataframe and download images into batch subfolders using the original
    download logic unchanged. Returns (summary_dict, result_df).

    result_df columns: product_id, category, image_path, batch_number
    """

    # Resolve paths and prepare folder
    dataframe_path = Path(dataframe_path).expanduser()
    image_path = Path(image_dir).expanduser()
    image_path.mkdir(parents=True, exist_ok=True)

    if not dataframe_path.exists():
        raise FileNotFoundError(f"Dataframe file not found: {dataframe_path}")

    # Load df (prefer parquet if suggested by extension; fallback to csv)
    suffix = dataframe_path.suffix.lower()
    if suffix in (".parquet", ".parq"):
        df = pd.read_parquet(dataframe_path)
    elif suffix in (".csv", ".txt"):
        df = pd.read_csv(dataframe_path)
    else:
        try:
            df = pd.read_parquet(dataframe_path)
        except Exception:
            df = pd.read_csv(dataframe_path)

    # Validate required columns
    missing = [c for c in (product_id_col, image_url_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")

    # Reset index so we have deterministic integer indices used as idx in tuples
    df = df.reset_index(drop=True)

    # Prepare or coerce batch column
    if batch_col not in df.columns:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive when batch column is absent.")
        df[batch_col] = (df.index // int(batch_size)) + 1
    else:
        # try to coerce to int; fallback to automatic assignment on failure
        try:
            df[batch_col] = df[batch_col].astype(int)
        except Exception:
            df[batch_col] = (df.index // int(batch_size)) + 1

    # Ensure category column exists (fill missing with None)
    if category_col not in df.columns:
        df[category_col] = None
    else:
        df[category_col] = df[category_col].apply(lambda x: x[0])

    # Prepare data_tuples: (idx, productid, batch_number, image_url)
    data_tuples = [
        (int(idx), row[product_id_col], int(row[batch_col]), row[image_url_col])
        for idx, row in df.iterrows()
    ]

    total = len(data_tuples)

    # Set module-level IMAGE_PATH for worker (preserves original worker behaviour)
    global IMAGE_PATH
    IMAGE_PATH = image_path

    # Thread-safe counters and structures
    lock = threading.Lock()
    total_processed = 0
    success_count = 0
    skipped_count = 0
    failed_count = 0

    # Prepare containers to store per-row image_path (string) or None
    image_paths = [None] * total  # indexable by idx (which matches df index)
    batch_numbers = [int(t[2]) for t in data_tuples]  # precomputed batch numbers

    def make_callback(idx: int):
        """
        Return a callback closure that captures the row index and updates
        counters and image_paths according to the worker result.
        """
        def _cb(result: str):
            nonlocal total_processed, success_count, skipped_count, failed_count
            # Determine image file path for this idx
            prod = data_tuples[idx][1]
            bnum = data_tuples[idx][2]
            file_path = (image_path / f"batch_{bnum}" / f"{prod}.jpg")

            with lock:
                total_processed += 1
                if result == 'success':
                    success_count += 1
                    # file should now exist
                    image_paths[idx] = str(file_path)
                elif result == 'skipped':
                    skipped_count += 1
                    # for skipped, use the existing path
                    image_paths[idx] = str(file_path)
                else:
                    failed_count += 1
                    # failed -> None
                    image_paths[idx] = None

                # Print progress periodically (mirrors original behavior)
                if (total_processed % int(progress_interval) == 0) or (total_processed == total):
                    print(
                        f"Processed: {total_processed}/{total} | Success: {success_count} | "
                        f"Skipped: {skipped_count} | Failed: {failed_count}"
                    )
        return _cb

    # Start ThreadPool and submit tasks with callbacks so we can map results -> row idx
    print(f"Downloading {total} images with {num_threads} threads...")
    print(f"Progress updates every {progress_interval} images\n")

    with ThreadPool(processes=int(num_threads)) as pool:
        async_results = []
        # Submit all tasks with their per-task callback
        for idx, tup in enumerate(data_tuples):
            # Note: download_single_image expects a single tuple argument
            ar = pool.apply_async(download_single_image, args=(tup,), callback=make_callback(idx))
            async_results.append(ar)

        # Wait for all tasks to complete by calling get() on each AsyncResult
        for ar in async_results:
            # get() will re-raise exceptions from worker if any; preserve original worker semantics
            try:
                ar.get()
            except Exception:
                # Worker exceptions are already represented as returned 'failed: ...' in unchanged worker;
                # but in case of unexpected exceptions, continue.
                pass

    # Final summary printing (mirrors original script)
    print(f"\n{'='*60}")
    print(f"Download Complete!")
    print(f"Total Processed: {total_processed}")
    print(f"Success: {success_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"{'='*60}")

    summary = {
        "total": total,
        "processed": total_processed,
        "success": success_count,
        "skipped": skipped_count,
        "failed": failed_count,
    }

    # Build the result DataFrame with requested columns:
    # product_id, category, image_path, batch_number
    result_df = pd.DataFrame({
        "product_id": df[product_id_col],
        "category": df[category_col],
        "image_path": image_paths,
        "batch_number": df[batch_col].astype(int),
    })

    return summary, result_df
