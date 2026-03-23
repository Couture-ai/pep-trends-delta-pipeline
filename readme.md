# PEP Trends Curation Pipeline

## 1. Setup
In a new conda environment, install the prerequisite libraries using the following command:
`pip install -r requirements.txt`


## 2. Config Setup

Update required paths and parameters in config.json. Each  parameter and its function is described in the specific tasks below.

## 3. Pipeline

Run each of the files present in SRC in order. The function and parameters of each file are explained below. The pipeline requires for input the following paramters: the raw metadata of the products to be processed in the form of processed SOLR data, local paths for the models used for prefiltering and final mapping, and a static file with category wise thresholds for minimum product-trend similarity.

### 3.1 000_scrape_product_trends.py

Input Parameters:
"raw_trends_path": Directory for dumping processed LLM responses after parsing with trend jsons for each category.
"catgory_list": list of categories to curate trends for.

ENSURE THAT OPENAI API KEY AND AZURE ENDPOINT URL ARE CONFIGURED CORRECTLY IN THIS FILE. THIS WILL NOT WORK OTHERWISE.

This file queries an online LLM with an API key and curates trends for each category based on the prompt provided, and parses the response and dumps the output in category wise jsons in the otuput dir provided in config.

### 3.2 001_process_raw_trends.py

Input Parameters:
"raw_trends_path": output of 000
"processed_trends_path": Output path for dumping merged, cleaned and processed CSV of all trends for all categories scraped in step 0.

This task simply parses the JSONs generated in step 000 and combines their results in a single consolidated CSV file for easier processing.

### 3.3 002_extract_catalog_metadata

Input Parameters:
"catalog_metadata_raw": processed parquet of live catalog SOLR data for the catalog.
"additional_tag_metadata_path": path to processed product and vibe tags data if curated for this catalog.
"catalog_metadata_processed": output path to dump processed metadata as required for subsequent steps.

This task takes in processed SOLR data and the product and vibe tags data for the products for which trend to product mapping is required, and filters and reformats it to retain only the required attribtues and tags for this process.

### 3.4 003_embed_for_prefilter.py

Input Parameters:
"pre_filter_output_parent_dir": output directory for the prefiltering output files,
"pre_fitler_tmp": temporary working directory for prefiltering, which will be cleared post processing. This directory needs at least 5 GB of free space to ensure correct execution.
"model_id": huggingface ID of the model for prefiltering, by default using `Marqo/marqo-ecommerce-embeddings-L`.
"local_model_path": directory where the model is downloaded. If empty or absent, the script will download the model automatically and save it to this path, which will require internet access. If empty and there is no internet access, this script will fail.
"model_weights_path": path to pytorch weights of the downloaded model.
"num_gpus": number of GPUs available for processing the embedding models. Higher is better.
"batch_size": batch size for embedding generation and similarity processing (in next step). Defaults to 512, recommended to set it to the highest that is possible without the GPU running out of memory. 
"catalog_token_embeddings_path": directory name for for storing category-level catalog embeddings after generation. This will be stored inside the temp dir. If this step crashes due to any reason, the task can be triggered again and will resume for only whatever categories are missing in this folder.

This task is used to generate vector embeddings for each of the unique attribute values present in all the catagories for the catalogue and storing it in a temp dir for further processing. The embeddings are generated at the attribute level, i.e. each unique attributevalue in the processed catalog metadata will have a corresponding embedding.

### 3.5 004_prefilter_with_marqo.py

Input Parameters:
In addition to the inputs from 003, additional inputs:
"raw_trends_path": output of 000
"processed_trends_path": output of 001
"pre_filter_final_output_file": final output of candidate products for each trend in the dataset,
"cateogory_wise_threshold_file":threshold for each category, which is the minimum score that a product needs to have to be considered a candidate.

In this step, embedding is generated for each trend using  the tag string, and its cosine similarity is calculated with each attribute value in the catalog. Then, for each trend the highest scoring attribtue values that are above specified thresholds are considered as "positive" markers for the trend, and any product that has any on of these positive attribtue values is associated as a candidate for their trend. The highest cosine similarity between a trend and an attribute value of the product is defined as the confidence of the product belonging to that trend. This initial candidate product set for trends is further refined in the next  step.

### 3.6 005_trend_product_mapping.py

Input Parameters:
"pre_filter_output_parent_dir", "pre_filter_final_output_file": location of output of 004.
"catalog_metadata_processed": output of 002
"processed_trends_path": output of 001
"final_trends_output_dir": output dir for final trend product mapping and trend metadata.
"cateogory_wise_threshold_file":hreshold for each category, which is the minimum score that a candidate needs to have to be considered a final mapped product. ( same file as 004 )
"start_date": starting date for trend to be considered valid,
"end_date": last date for trend to be considered valid.


This step takes in the candidate set from step 1, and uses a sentence embedding model to create a single vector for each product in the set of candidates by combining their attribtutes into a single product description. It then creates a single vector for each trend by using the description of the trend from the processed trends csv and ranks each candidate product according to cosine similarity, filters them according to thresholds and outputs the final trend metadata ( including vectors) and the trend-product mapping that can be uploaded to  the console.