import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings
import json
warnings.filterwarnings("ignore")
UNIVERSAL_CONFIG_PATH = "/app/notebooks/pep_dev_ready/config.json"

def efficient_product_description(df):
    """Generates condensed product descriptions from metadata."""
    # 1. Memory Optimization: Convert 'type' to Category
    if not isinstance(df['standardize_type_updated'].dtype, pd.CategoricalDtype):
        df['standardize_type_updated'] = df['standardize_type_updated'].astype('category')
    print("Converted type to category")
    
    # 2. First Aggregation: Group by ID and Type
    df_grouped = (
        df.groupby(['product_id', 'standardize_type_updated'], sort=False, observed=True)['normalised_name']
        .agg(lambda x: ", ".join(dict.fromkeys(x.astype(str))))
        .reset_index()
    )
    print("Grouped data")

    # 3. Vectorized String Formatting
    df_grouped['formatted'] = (
        df_grouped['standardize_type_updated'].astype(str) + 
        " values are " + 
        df_grouped['normalised_name']
    )
    print("String formatted")
    
    # 4. Final Aggregation
    df_final = (
        df_grouped.groupby('product_id', sort=False)['formatted']
        .agg(lambda x: ". ".join(x) + ".")
        .reset_index()
    )
    print("Aggregating complete")
    
    df_final.columns = ['product_id', 'product_description']
    return df_final


def main():

    def load_universal_config(config_path=UNIVERSAL_CONFIG_PATH):

        
    
        with open("/app/notebooks/pep_dev_ready/config.json") as f:
            config = json.load(f)
        
        path_trend_product_map = f'{config["pre_filter_output_parent_dir"]}/{config["pre_filter_final_output_file"]}'
        path_product_metadata = config["catalog_metadata_processed"]
        path_embedding_model = config["ajio_embedding_model_path"]
        path_processed_tags = config["processed_trends_path"]
        dir_output_path = config["final_trends_output_dir"]
        start_date = config["start_date"]
        end_date = config["end_date"]
        
        print("Universal Config Loaded Successfully!!")

        return path_trend_product_map, path_product_metadata, path_embedding_model, path_processed_tags, dir_output_path
    
    # ==========================================
    # 1. Define Input and Output Paths
    # ==========================================
    # path_trend_product_map = "/data/pep/multitags_v3/merged_output/trending_tags_multitags.parquet"
    # path_product_metadata = "/data/pep/solrPlusTagsTitleData"
    # path_embedding_model = "/data/models/ajio_mpnet_v6_ep3/"
    # path_processed_tags = "../misc/processed_tags_multitags_v3_combined.csv"
    path_trend_product_map, path_product_metadata, path_embedding_model, path_processed_tags, dir_output_path, start_date, end_date = load_universal_config()
    
    # Ensure outputs directory exists
    os.makedirs(dir_output_path, exist_ok=True)
    
    output_final_tag_path = f"{dir_output_path}/final_tag_df.parquet" # Or swap to final_tag_df_path of your choice
    output_ranking_path = f"{dir_output_path}/top_1k_trends_trends_v3_new_ranking.parquet"

    # ==========================================
    # 2. Data Loading & Initial Filtering
    # ==========================================
    print("Loading datasets...")
    trend_product_map = pd.read_parquet(path_trend_product_map)
    product_metadata = pd.read_parquet(path_product_metadata)

    base_threshold = -0.5
    filtered_trend_product_map = trend_product_map[trend_product_map["confidence"] > base_threshold]
    
    # Sort and take top 5000 per tag and category
    filtered_trend_product_map_top_1k = trend_product_map.sort_values(
        by=['tag', 'category', 'confidence'], 
        ascending=[True, True, False]
    ).groupby(['tag', 'category']).head(5000)

    # ==========================================
    # 3. Attribute Filtering & Descriptions
    # ==========================================
    valid_tags = ['style', 'color', 'cut', 'neckline', 'sleeve length', 'length', 'pattern',
                  'seasonality', 'trim', 'hemline', 'sleeve type', 'closure', 'embroidery']
    valid_catalog_attributes = ['brand', 'occasion', 'rise', 'embellishment', 
                                'cartoon / movie character', 'lining', 'upper material', 'slit']
    title_type = ["product title"]
    valid_attributes = set(valid_tags + valid_catalog_attributes + title_type)

    # Filter product metadata based on valid products and valid attributes
    df = product_metadata[product_metadata["product_id"].isin(filtered_trend_product_map_top_1k["product_id"])].copy()
    df = df[df["standardize_type_updated"].isin(valid_attributes)]

    print(f"Generating descriptions for {df['product_id'].nunique()} unique products...")
    descriptions_df = efficient_product_description(df)

    # ==========================================
    # 4. Generate Product Embeddings
    # ==========================================
    print(f"Loading SentenceTransformer model from {path_embedding_model}...")
    model = SentenceTransformer(path_embedding_model)
    
    print("Encoding product descriptions...")
    descriptions_df["embeddings"] = model.encode(
        descriptions_df["product_description"].to_list(), 
        device=["cuda:0", "cuda:1", "cuda:2", "cuda:3"], # Adjust GPU devices if needed
        show_progress_bar=True
    ).tolist()

    # Map product_id to embeddings
    product_embeddings = descriptions_df.set_index("product_id")["embeddings"].to_dict()

    # ==========================================
    # 5. Process Tags & Compute Similarities
    # ==========================================
    print("Processing tag embeddings...")
    tag_df = pd.read_csv(path_processed_tags)
    
    # Optional: Filter based on any constraints you had (e.g. valid_tag_df logic)
    valid_tag_df = tag_df.copy() # Assuming preprocessing applies here

    # Format the final_tag_df output
    if "preprocessed_tag" in valid_tag_df.columns:
        final_tag_df = valid_tag_df[["preprocessed_tag", "category", "region", "embedding"]].rename(columns={
            "preprocessed_tag": "tag_value",
            "region": "tag_type",
            "embedding": "vector",
        })
    else:
        final_tag_df = valid_tag_df
        
    
    # ==========================================
    # 6. Reranking using Cosine Similarity
    # ==========================================
    print("Computing cosine similarities and reranking...")
    
    # Join product vectors with tag vectors
    rank_df = filtered_trend_product_map_top_1k.merge(
        descriptions_df[['product_id', 'embeddings']], 
        on='product_id', 
        how='inner'
    ).rename(columns={"embeddings": "p_vec"})
    
    # Filter non-null vectors
    rank_df = rank_df[rank_df["p_vec"].notna()]
    
    # Convert tag embeddings strings back to arrays if needed, and map them
    # Assuming valid_tag_df has 'tag' and 'embedding' columns representing vectors
    tag_vector_map = valid_tag_df.set_index('tag')['embedding'].to_dict()
    
    # Helper to calculate cosine similarity
    def cosine_sim(vec1, vec2):
        if not isinstance(vec1, (list, np.ndarray)) or not isinstance(vec2, (list, np.ndarray)):
            return 0.0
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        # Assuming vectors are not strictly normalized
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # Apply similarity calculation
    rank_df['t_vec'] = rank_df['tag'].map(tag_vector_map)
    rank_df['cosine_similarity'] = rank_df.apply(lambda row: cosine_sim(row['t_vec'], row['p_vec']), axis=1)

    # Sort by tag, category, and newly computed cosine similarity
    df_sorted = rank_df.sort_values(
        by=['tag', 'category', 'cosine_similarity'], 
        ascending=[True, True, False]
    )

    # Keep top 1000 records per tag/category
    top_thousand_df = df_sorted.groupby(['tag', 'category']).head(1000).reset_index(drop=True)

    top_thousand_df["start_date"]= start_date
    top_thousand_df["end_date"] = end_date
    top_thousand_df["gender"]=top_thousand_df["category"].apply(lambda x: x.split(" - ")[0].lower())

    # Clean up output formatting (remove raw vector columns)
    output_df = top_thousand_df.drop(columns=['p_vec', 't_vec'], errors='ignore')

    # ==========================================
    # 7. Final Saving
    # ==========================================
    print(f"Saving final ranking to {output_ranking_path}...")
    output_df.to_parquet(output_ranking_path, index=False)
    print("Pipeline Execution Complete!")

    print(f"Saving final_tag_df to {output_final_tag_path}...")
    final_tag_df = final_tag_df[final_tag_df["preprocessed_tag"].isin(output_df["tag"])]
    final_tag_df.to_parquet(output_final_tag_path, index=False)


if __name__ == "__main__":
    main()