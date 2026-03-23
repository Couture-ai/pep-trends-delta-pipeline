#!/usr/bin/env python3
"""
cluster_labeling_pipeline.py

Author: Generated (assistant, revised)
Date: 2025-11-21

This module implements an asynchronous, high-throughput pipeline to generate concise
cluster labels for product-tag groups. Configuration parameters are declared at the
top of the file for convenience; modify these variables directly to control behaviour.

Revision summary:
- The last cluster (final non-empty line) of each input file is NOT submitted to the
  model. Instead, it is recorded under the label "noise" in the JSON output for that file.
- When a file contains a single cluster, that cluster is treated as "noise" and no model
  calls are made for that file.
- All other behaviour is unchanged from the prior version.

Operational summary:
- A local HTTP-compatible vLLM (e.g., GPT-OSS-120B served at an internal base_url)
  accepts requests at the `/chat/completions` endpoint with payload shape:
    {"model": "<model_id>", "messages": [{"role": "user", "content": "<prompt>"}]}

- The pipeline constructs a compact, deterministic prompt per cluster, submits
  prompts in parallel using httpx.AsyncClient, constrains concurrency via an
  asyncio.Semaphore, and writes a JSON summary mapping labels to cluster lists.

- The prompt enforces label constraints (lower-case, no punctuation, ≤4 words).
  A conservative post-processing stage enforces these constraints in the event
  that model outputs diverge.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from tqdm import tqdm

# ---------------------------
# Configuration (edit here)
# ---------------------------

# Directory containing cluster files. Each file should contain one cluster per line,
# with tags in a line separated by `SEP`.
INPUT_DIR: str = "/data/pep/tags_standardisation_output/clusters"
# Directory where JSON outputs will be written.
OUTPUT_DIR: str = "/data/pep/tags_standardisation_output/clusters_with_vllm_labels_v2_new"
# Glob pattern to select files in INPUT_DIR.
PATTERN: str = "*.txt"
# Base URL for the locally hosted vLLM service (no trailing slash ideally).
BASE_URL: str = "http://10.172.148.69:8000/v1"
# Model identifier as understood by the local service.
MODEL: str = "gpt-oss-120b"
# Maximum number of concurrent HTTP requests to the model service.
CONCURRENCY: int = 16
# Per-request timeout in seconds.
TIMEOUT: float = 30.0
# Separator used in cluster files to delimit tags on a line.
SEP: str = "|||"
# Optional Authorization header value, e.g. "Bearer TOKEN". Set to None if not needed.
AUTH_HEADER: Optional[str] = None

# ---------------------------
# Prompt construction
# ---------------------------

def make_label_prompt(tags: List[str]) -> str:
    """
    Construct the label-generation prompt for a set of tags.

    The prompt instructs the model to return only a concise label:
      - lower-case
      - no punctuation
      - maximum of 4 words

    The function truncates the list to 50 tags to limit prompt length for throughput.
    """
    truncated = tags[:50]
    suffix = "\n... (truncated)" if len(tags) > 50 else ""
    prompt = (
        "You are a concise label generator. Output only a short label (maximum 4 words), "
        "lower-case, without punctuation.\n"
        "Given the following product tag strings, produce one concise label summarizing "
        "only the common aspects across the group. The output must contain only the label.\n\n"
        "Tags:\n" +
        "\n".join(truncated) +
        suffix
    )
    return prompt


# ---------------------------
# File I/O utilities
# ---------------------------

def read_cluster_file(path: str, sep: str = "|||") -> List[List[str]]:
    """
    Read a cluster file where each line contains tags separated by `sep`.
    Returns a list of clusters, each cluster is a list of tag strings.

    Blank lines are ignored.
    """
    clusters: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(sep) if p.strip()]
            if parts:
                clusters.append(parts)
            else:
                # If a line exists but no tokens after splitting, skip (defensive).
                continue
    return clusters


def write_json_atomic(obj: Any, out_path: str) -> None:
    """
    Write JSON to file atomically (write to temp file then rename).
    """
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)


# ---------------------------
# HTTP / Async inference
# ---------------------------

async def _fetch_single(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    timeout: float = 30.0,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Submit a single prompt to the model endpoint and defensively extract the text.

    Return dict with keys:
      - "response": str | None
      - "error": dict | None
    """
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    headers = headers or {}
    try:
        async with semaphore:
            resp = await client.post(endpoint, json=payload, timeout=timeout, headers=headers)
        if resp.status_code < 200 or resp.status_code >= 300:
            try:
                text = await resp.aread()
                text_decoded = text.decode(errors="ignore")
            except Exception:
                text_decoded = resp.text
            return {
                "response": None,
                "error": {
                    "type": "http_error",
                    "status_code": resp.status_code,
                    "text": text_decoded,
                },
            }
        data = resp.json()
        choices = data.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                if "message" in first and isinstance(first["message"], dict):
                    return {"response": first["message"].get("content"), "error": None}
                if "text" in first:
                    return {"response": first.get("text"), "error": None}
        return {"response": None, "error": {"type": "unexpected_shape", "raw": data}}
    except Exception as exc:
        raise exc
        # return {"response": None, "error": {"type": "exception", "repr": repr(exc), "str": str(exc)}}


async def parallel_prompts_to_responses(
    prompts: List[str],
    *,
    base_url: str = "http://10.172.148.69:8000/v1",
    model: str = "openai/gpt-oss-120b",
    concurrency: int = 16,
    timeout: float = 30.0,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Submit a list of prompts in parallel to the /chat/completions endpoint.

    Returns a list aligned with prompts, each entry being the dict as returned by `_fetch_single`.
    """
    endpoint = base_url.rstrip("/") + "/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(
                _fetch_single(client, endpoint, model, p, semaphore, timeout, headers)
            )
            for p in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    return results


# ---------------------------
# Post-processing and normalization
# ---------------------------

_LABEL_SAFE_RE = re.compile(r"[a-z0-9\s]+")

def sanitize_label(raw: Optional[str]) -> str:
    """
    Post-process a model output to enforce label constraints:
      - lower-case
      - remove punctuation (non alphanumeric + spaces)
      - collapse whitespace
      - trim to at most 4 words
      - fallback to 'unlabeled' if empty or unsafe
    """
    if not raw:
        return "unlabeled"
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return "unlabeled"
    words = s.split()
    words = words[:4]
    candidate = " ".join(words)
    if not _LABEL_SAFE_RE.fullmatch(candidate):
        return "unlabeled"
    return candidate


# ---------------------------
# High-level pipeline: per-file processing (with final-cluster-as-noise)
# ---------------------------

async def process_cluster_file(
    input_path: str,
    output_dir: str,
    *,
    base_url: str = "http://10.172.148.69:8000/v1",
    model: str = "openai/gpt-oss-120b",
    concurrency: int = 16,
    timeout: float = 30.0,
    headers: Optional[Dict[str, str]] = None,
    sep: str = "|||"
) -> str:
    """
    Read clusters from `input_path`, treat the last cluster as 'noise' (no model call),
    generate labels for remaining clusters using the configured model endpoint, and
    write a JSON summary to `output_dir/<basename>.json`.

    JSON schema:
      {
        "<label_1>": [ [tagA, tagB, ...], ... ],
        "noise": [ [tagX, ...] ],
        "unlabeled": [ ... ]
      }
    """
    os.makedirs(output_dir, exist_ok=True)
    clusters = read_cluster_file(input_path, sep=sep)
    out_basename = os.path.basename(input_path) + ".json"
    out_path = os.path.join(output_dir, out_basename)

    # -------- SKIP IF OUTPUT ALREADY EXISTS --------
    if os.path.exists(out_path) and "error" not in json.load(open(out_path, 'r')):
    # if os.path.exists(out_path):
        print(f"[skip] {out_basename} already processed!!")
        return out_path
    # print("not processed yet")
    # If no clusters, write empty mapping and return early
    if not clusters:
        write_json_atomic({}, out_path)
        return out_path

    # Separate last cluster as 'noise' (do not submit to model)
    if len(clusters) == 1:
        # Single cluster file: treat the sole cluster as noise
        summary: Dict[str, List[List[str]]] = {}
        summary.setdefault("noise", []).append(clusters[0])
        write_json_atomic(summary, out_path)
        return out_path

    # More than one cluster: last cluster reserved as noise, process rest
    clusters_to_label = clusters[:-1]
    noise_cluster = clusters[-1]

    # Build prompts for clusters_to_label (one prompt per cluster)
    prompts = [make_label_prompt(c) for c in clusters_to_label]

    # Submit prompts in parallel (if any)
    responses = []
    if prompts:
        responses = await parallel_prompts_to_responses(
            prompts,
            base_url=base_url,
            model=model,
            concurrency=concurrency,
            timeout=timeout,
            headers=headers,
        )

    # Map labels to clusters; include 'noise' cluster explicitly
    summary: Dict[str, List[List[str]]] = {}
    # add noise cluster
    summary.setdefault("noise", []).append(noise_cluster)

    # iterate over labeled clusters and their corresponding responses
    for cluster_tags, resp in zip(clusters_to_label, responses):
        if resp.get("error") is not None:
            # On any error, record the diagnostic in stdout and mark this cluster as 'unlabeled'
            print(f"Error for file {input_path}: {resp}")
            # label = "unlabeled"
            exit(0)
        else:
            raw = resp.get("response") or ""
            label = sanitize_label(raw)
            if not label:
                label = "unlabeled"
        summary.setdefault(label, []).append(cluster_tags)

    # Write JSON output
    write_json_atomic(summary, out_path)
    return out_path


async def process_all_cluster_files(
    input_dir: str,
    output_dir: str,
    *,
    pattern: str = "*.txt",
    base_url: str = "http://10.172.148.69:8000/v1",
    model: str = "gpt-oss-120b",
    concurrency: int = 16,
    timeout: float = 30.0,
    headers: Optional[Dict[str, str]] = None,
    sep: str = "|||"
) -> List[str]:
    """
    Process every file matching `pattern` under `input_dir` sequentially.
    Each file is processed in turn; within each file the prompts are submitted
    in parallel using the concurrency limit.

    Returns the list of output paths written.
    """
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    # Optional filtering preserved from original (user previously had a file filter).
    # If you want to process all files, remove the next line.
    # files = [file for file in files if "Men_-_Shirts_shop_the_vibe_chunk" in file]
    # print(files)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {input_dir}")

    output_paths: List[str] = []
    for fpath in tqdm(files, desc="Processing cluster files", unit="file"):
        try:
            out = await process_cluster_file(
                fpath,
                output_dir,
                base_url=base_url,
                model=model,
                concurrency=concurrency,
                timeout=timeout,
                headers=headers,
                sep=sep,
            )
            output_paths.append(out)
        except Exception as exc:
            diagnostic = {"error": {"type": "exception", "repr": repr(exc), "str": str(exc)}}
            out_basename = os.path.basename(fpath) + ".json"
            diag_path = os.path.join(output_dir, out_basename)
            write_json_atomic(diagnostic, diag_path)
            output_paths.append(diag_path)
    return output_paths


# ---------------------------
# Entrypoint
# ---------------------------

def main() -> None:
    """
    Driver which uses the top-of-file configuration variables. This function
    orchestrates the end-to-end workflow and prints a brief completion summary.
    """
    headers = None
    if AUTH_HEADER:
        headers = {"Authorization": AUTH_HEADER}

    try:
        asyncio.run(
            process_all_cluster_files(
                INPUT_DIR,
                OUTPUT_DIR,
                pattern=PATTERN,
                base_url=BASE_URL,
                model=MODEL,
                concurrency=CONCURRENCY,
                timeout=TIMEOUT,
                headers=headers,
                sep=SEP,
            )
        )
        print(f"Completed processing files. Outputs written to: {OUTPUT_DIR}")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as exc:
        print(f"Fatal error: {exc!r}")


if __name__ == "__main__":
    main()
