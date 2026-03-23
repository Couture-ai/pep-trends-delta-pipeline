import os
import re
import math
import time
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------
# Utility functions (unchanged)
# ---------------------------
def _sanitize_for_filename(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:200]


def save_clusters_line_format_with_noise(tags, labels, filepath, separator="|||", append=False):
    mode = "a" if append else "w"
    cluster_dict = {}
    for tag, lbl in zip(tags, labels):
        cluster_dict.setdefault(lbl, []).append(tag)

    non_noise_labels = sorted(k for k in cluster_dict.keys() if k != -1)

    with open(filepath, mode, encoding="utf-8") as f:
        for lbl in non_noise_labels:
            cluster = cluster_dict.get(lbl, [])
            if cluster:
                f.write(separator.join(cluster) + "\n")
        noise_cluster = cluster_dict.get(-1, [])
        if noise_cluster:
            f.write(separator.join(noise_cluster) + "\n")


def load_clusters_line_format(filepath, separator="|||"):
    clusters = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            clusters.append(line.split(separator))
    return clusters


# ---------------------------
# New clustering pipeline: TF-IDF (char ngrams) + cosine distance
# ---------------------------
def run_dbscan_clustering_tfidf_cosine(
    tags,
    eps=0.075,
    min_samples=2,
    ngram_range=(4, 7),
    analyzer="char_wb",
    max_features=65536,
    dtype=np.float32,
):
    """
    Compute pairwise cosine distance on character n-gram TF-IDF vectors and run DBSCAN.

    Parameters
    ----------
    tags : List[str]
        Input list of strings.
    eps : float
        DBSCAN eps (distance). Cosine-distance = 1 - cosine-similarity.
    min_samples : int
        DBSCAN min_samples.
    ngram_range : tuple
        Character n-gram range for TF-IDF.
    analyzer : str
        Analyzer for TfidfVectorizer (char_wb recommended for short strings).
    max_features : int
        Max features for TF-IDF (keeps memory bounded).
    dtype : numpy dtype
        dtype for the resulting distance matrix (helps reduce memory footprint).

    Returns
    -------
    labels : np.ndarray
        Cluster labels from DBSCAN.
    """
    n = len(tags)
    if n == 0:
        return np.array([], dtype=int)

    t0 = time.time()
    logger.info("Vectorizing %d tags with TF-IDF (analyzer=%s, ngram_range=%s, max_features=%d)",
                n, analyzer, ngram_range, max_features)

    # Vectorize: use sparse matrix; char_wb is robust for short tokens
    vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, max_features=max_features)
    X = vect.fit_transform(tags)  # sparse (n_samples, n_features)
    logger.info("TF-IDF matrix shape: %s, nnz=%d", X.shape, X.nnz)

    t_vec = time.time()
    logger.info("Computing cosine similarity (parallel)...")
    # cosine_similarity supports sparse input and n_jobs; returns dense array by default
    # We request a dense output because DBSCAN with precomputed metric expects an array-like.
    # For n ~ 7000 the resulting dense matrix (7000x7000) is manageable (~~400MB for float64).
    # We cast to dtype at the end to reduce memory footprint if needed.
    # similarity = cosine_similarity(X, X, dense_output=True, n_jobs=-1)
    # distance_matrix = (1.0 - similarity).astype(dtype, copy=False))
    # Compute cosine distance directly (parallelized)
    distance_matrix = pairwise_distances(X, X, metric="cosine", n_jobs=-1).astype(dtype, copy=False)
    # Ensure diagonal is exactly 0 (cosine_distance( x,x ) should be 0 but numerical noise can appear)
    np.fill_diagonal(distance_matrix, 0.0)
    t_sim = time.time()
    logger.info("Cosine similarity computed in %.2fs (vec step %.2fs)", t_sim - t_vec, t_vec - t0)

    # Distance matrix = 1 - similarity; clamp numerical noise to [0, 2]
    # distance_matrix = (1.0 - similarity).astype(dtype, copy=False)
    # optional: ensure diagonal is zero
    # np.fill_diagonal(distance_matrix, 0.0)

    t_dist = time.time()
    logger.info("Distance matrix ready (dtype=%s). Shape: %s; building/clamping took %.2fs",
                distance_matrix.dtype, distance_matrix.shape, t_dist - t_sim)

    # Run DBSCAN: using precomputed distances. Use n_jobs=-1 to parallelize where available.
    logger.info("Running DBSCAN (eps=%.4f, min_samples=%d)...", eps, min_samples)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)
    labels = db.fit_predict(distance_matrix)

    t_db = time.time()
    logger.info("DBSCAN finished in %.2fs (total pipeline %.2fs)", t_db - t_dist, t_db - t0)

    return labels


# ---------------------------
# Main: load dataset and run per (category,attribute) using chunking
# ---------------------------
def main():
    # --- CONFIG ---
    parquet_path = "/data/pep/tags_for_clustering_r3"
    output_dir = "/data/pep/tags_standardisation_output/clusters_layer_3_cosine"
    tag_column = "tag_std_l1"
    type_column = "tag_type"
    category_column = "category"
    sample_mode = False

    # clustering hyperparameters (tune as needed)
    eps_val = 0.15           # cosine-distance threshold (1 - cosine_sim). Tune empirically.
    min_samples_val = 2
    ngram_range = (4, 6)     # captures local substrings and approximate order-independence
    max_features = 65536     # bounding the TF-IDF vocabulary
    dtype = np.float32       # reduce memory footprint of distance matrix

    # chunking policy (unchanged)
    CHUNK_SIZE = 7000
    THRESHOLD_FOR_SPLIT = 8000

    os.makedirs(output_dir, exist_ok=True)

    # Attempt to use HDFS filesystem if available; otherwise fall back to local
    try:
        os.environ["LIBHDFS_OPTS"] = (
            "-Djava.security.krb5.conf=/home/jioapp/aditya/jiomart_cluster/krb5.conf"
        )
        fs = pa.hdfs.connect(host='10.144.96.170', port=8020, kerb_ticket="/home/jioapp/aditya/jiomart_cluster/krb5cc_154046")
        dataset = pq.ParquetDataset(parquet_path, filesystem=fs)
    except Exception:
        logger.warning("HDFS connection failed or not available; falling back to local ParquetDataset.")
        dataset = pq.ParquetDataset(parquet_path)

    data = dataset.read().to_pandas()

    assert category_column in data.columns and type_column in data.columns and tag_column in data.columns
    print(f"loaded dataset with {len(data)} rows, which has {data[tag_column].nunique()} tags for clustering")
    if sample_mode:
        pairs = [['Women - Dresses', 'shop the vibe']]
    else:
        pairs = data[[category_column, type_column]].drop_duplicates().values.tolist()

    for cat, attr in pairs:
        mask = (data[category_column] == cat) & (data[type_column] == attr)
        tags = sorted(data.loc[mask, tag_column].dropna().unique().tolist())
        n = len(tags)
        if n == 0:
            continue

        # Decide chunking
        if n > THRESHOLD_FOR_SPLIT:
            n_chunks = math.ceil(n / CHUNK_SIZE)
            chunks = [tags[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] for i in range(n_chunks)]
        else:
            chunks = [tags]

        cat_s = _sanitize_for_filename(cat)
        attr_s = _sanitize_for_filename(attr)

        for idx, chunk_tags in enumerate(chunks):
            chunk_num = f"{idx:02d}" if len(chunks) > 1 else "00"
            out_fname = f"{cat_s}_{attr_s}_chunk_{chunk_num}.txt"
            out_path = os.path.join(output_dir, out_fname)

            logger.info("Processing category=%s | attribute=%s -> chunk %s/%d (n_tags=%d)",
                        cat, attr, chunk_num, len(chunks), len(chunk_tags))

            # Run optimized TF-IDF cosine-distance + DBSCAN
            labels = run_dbscan_clustering_tfidf_cosine(
                chunk_tags,
                eps=eps_val,
                min_samples=min_samples_val,
                ngram_range=ngram_range,
                analyzer="char_wb",
                max_features=max_features,
                dtype=dtype,
            )

            save_clusters_line_format_with_noise(chunk_tags, labels, out_path, separator="|||", append=False)
            logger.info("Saved %d tags -> %s", len(chunk_tags), out_path)

    logger.info("All done.")


if __name__ == "__main__":
    main()
