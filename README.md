# PEP Trends Curation Pipeline

## Overview
This repository contains the PEP Trends Curation Pipeline used to:
- curate weekly, category-specific trend hashtags using an LLM,
- normalize/merge curated trend outputs into a single dataset,
- generate embeddings for catalog attributes and trends,
- pre-filter candidate products using similarity + category thresholds, and
- produce a final trend → product mapping suitable for downstream consumption.

## Project Structure
```
pep-trends-delta-pipeline/
├── config.json
├── requirements.txt
├── readme.md
├── data/
│   └── trending_tags_thresholds_v3.csv
└── src/
    ├── 000_scrape_product_trends.py
    ├── 001_process_raw_trends.py
    ├── 003_embed_for_prefilter.py
    ├── 004_pre-filter_with_marqo.py
    ├── 005_trend_product_mapping.py
    └── openai_utils.py
```

## Prerequisites
- Python 3.x
- (Recommended) Conda / virtualenv
- GPU(s) for embedding generation and similarity steps (003/004) depending on dataset size
- Access to Azure OpenAI (for step 000)

## Installation
Create a fresh environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration
Update `config.json` with environment-specific paths and parameters.

> Note: Several scripts also contain hard-coded defaults such as `UNIVERSAL_CONFIG_PATH = "/app/notebooks/pep_dev_ready/config.json"`. In enterprise deployments, prefer standardizing this to use the repo-level `config.json` (or a single runtime config path) to avoid configuration drift.

### Key Config Fields (high level)
- Trend scraping
  - `raw_trends_path`: directory where per-category trend JSONs are written
  - `category_list`: list of catalog categories to curate trends for
- Trend processing
  - `processed_trends_path`: consolidated trends CSV output
- Pre-filtering (candidate generation)
  - `pre_filter_output_parent_dir`: output parent directory for prefilter artifacts
  - `pre_fitler_tmp` / `pre_filter_tmp`: temporary working directory
  - `model_id`, `local_model_path`, `model_weights_path`: embedding model configuration
  - `num_gpus`, `batch_size`: runtime sizing
  - `catalog_token_embeddings_path`: directory name used to store catalog token embeddings
  - `pre_filter_final_output_file`: final candidates file name/path
  - `cateogory_wise_threshold_file`: category-wise similarity thresholds CSV
- Final mapping
  - `catalog_metadata_processed`: processed catalog metadata input
  - `final_trends_output_dir`: final outputs directory
  - `start_date`, `end_date`: trend validity window

## Pipeline Runbook
Run the scripts in `src/` in numeric order.

### 1) Scrape Trends (LLM)
**Script:** `src/000_scrape_product_trends.py`

**Purpose:**
Generates weekly trend hashtags per category by calling an LLM and writing category-wise JSON outputs.

**Config Inputs:**
- `raw_trends_path`
- `category_list`

**Operational Notes:**
- This script requires Azure OpenAI credentials.
- Ensure secrets are injected via environment variables or a secure secret manager.
- Avoid committing API keys or endpoints into source control.

---

### 2) Process Raw Trend JSONs
**Script:** `src/001_process_raw_trends.py`

**Purpose:**
Parses the JSONs from step 1 and produces a consolidated, cleaned trends CSV.

**Config Inputs:**
- `raw_trends_path` (output of step 1)
- `processed_trends_path`

---

### 3) Embed Catalog Attributes (Pre-filter Preparation)
**Script:** `src/003_embed_for_prefilter.py`

**Purpose:**
Generates vector embeddings for catalog attribute tokens/values to enable efficient similarity search.

**Config Inputs (typical):**
- `catalog_metadata_processed`
- `model_id`, `local_model_path`, `model_weights_path`
- `num_gpus`, `batch_size`
- `pre_fitler_tmp` / `pre_filter_tmp`
- `catalog_token_embeddings_path`

**Operational Notes:**
- Requires enough temporary disk (multi-GB depending on catalog size).
- For GPU runs, tune `batch_size` to avoid OOM.

---

### 4) Pre-filter Candidate Products (Similarity + Thresholds)
**Script:** `src/004_pre-filter_with_marqo.py`

**Purpose:**
Embeds trends and computes cosine similarity against catalog attribute embeddings to generate a candidate set of products per trend. Applies category-wise thresholds.

**Config Inputs:**
- `raw_trends_path`
- `processed_trends_path`
- `cateogory_wise_threshold_file`
- `pre_filter_output_parent_dir`
- `pre_filter_final_output_file`

---

### 5) Final Trend → Product Mapping
**Script:** `src/005_trend_product_mapping.py`

**Purpose:**
Builds final product descriptions, embeds products and trends, ranks by similarity, applies thresholds, and writes final mapping artifacts.

**Config Inputs:**
- `catalog_metadata_processed`
- `processed_trends_path`
- `pre_filter_output_parent_dir`, `pre_filter_final_output_file`
- `cateogory_wise_threshold_file`
- `final_trends_output_dir`
- `start_date`, `end_date`

## Outputs (Typical)
- Category-wise raw trend JSONs (step 1)
- Consolidated trends CSV (step 2)
- Catalog token embeddings + similarity artifacts (steps 3–4)
- Final trend metadata + trend-product mapping for downstream systems (step 5)

## Security & Compliance
- Do not commit credentials.
- Prefer environment variables / secret managers for Azure OpenAI settings.
- Validate that outputs do not contain sensitive or disallowed content before publishing.
