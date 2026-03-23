import pandas as pd
import json
import os
import glob
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
import itertools
import gc
import shutil

# --- NEW SPAcY IMPORTS (CLEANED) ---
import spacy
from typing import Set
from spacy.symbols import ORTH, LEMMA # <-- Removed invisible space
# --- END NEW IMPORTS ---

# Set this environment variable to mitigate memory fragmentation issues
# This helps with the CUDA Out of Memory error
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
UNIVERSAL_CONFIG_PATH = "/app/notebooks/pep_dev_ready/config.json"

# =================================================================
# 1. CONFIGURATION AND PATHS - NO CHANGE
# =================================================================

# --- Model Config ---
# MODEL_ID = "Marqo/marqo-ecommerce-embeddings-L"
# LOCAL_MODEL_PATH = "/data/models/marqo-ecommerce-L"
# CONFIG_FILE_PATH = os.path.join(LOCAL_MODEL_PATH, "config.json")
# MODEL_WEIGHTS_PATH = "/data/models/marqo-ecomm-embeddings-l.pt"


# --- Tag Data Config ---
# TAG_JSON_INPUT_DIR = '/data/pep/global_and_india_tags' 
# TAG_JSON_INPUT_DIR = '/app/notebooks/misc/trends_v4_updated_description'
# TEMP_OUTPUT_ROOT = "/data/pep/tmp/similarity_temp_data"
# TEMP_OUTPUT_ROOT = "/data/pep/tmp/similarity_temp_data_multitags_v4"
# PROCESSED_TAGS_FILE_PATH="/app/notebooks/misc/processed_tags.csv"
# PROCESSED_TAGS_FILE_PATH="/app/notebooks/misc/processed_tags_trends_v4_updated_descriptions.csv"
# LOCAL_TAG_EMBEDDING_FILE = TEMP_OUTPUT_ROOT + '/all_tags_with_embeddings.parquet'

# --- Item Data Config (Similarity) ---
# THESE DIRECTORIES REMAIN THE ROOT. INPUT/OUTPUT WILL USE CATEGORY SUBFOLDERS.
# INPUT_DIR_EMBEDDINGS = "/data/pep/solrPlusTagsTitleData_with_embeddings"
# OUTPUT_DIR_SIMILARITY = "/data/pep/solrPlusTagsTitleData_with_SIMILARITY"
# OUTPUT_DIR_SIMILARITY = "/data/pep/trends_v4_updated_description/solrPlusTagsTitleData_with_SIMILARITY_multitags"
# PROCESSED_OUTPUT_DIR = "/data/pep/solrPlusTagsTitleData_PROCESSED_OUTPUT"
# PROCESSED_OUTPUT_DIR = "/data/pep/trends_v4_updated_description/solrPlusTagsTitleData_PROCESSED_OUTPUT_multitags"
# COMBINED_OUTPUT_FILE_PATH = "/data/pep/merged_output/trending_tags.parquet"
# COMBINED_OUTPUT_FILE_PATH = "/data/pep/trends_v4_updated_description/merged_output/trending_tags_multitags.parquet"
# FINAL_FILTERED_OUTPUT_FILE_PATH="/data/pep/merged_output/trending_tags_filtered.parquet"
# FINAL_FILTERED_OUTPUT_FILE_PATH="/data/pep/trends_v4_updated_description/merged_output/trending_tags_filtered_multitags.parquet"
# CONFIDENCE_THRESHOLD = 0.0
# THRESHOLD_CSV_FILE_PATH="/app/notebooks/misc/trending_tags_tresholds_test.csv"


# --- Model Config ---
MODEL_ID = ""
LOCAL_MODEL_PATH = ""
CONFIG_FILE_PATH = ""
MODEL_WEIGHTS_PATH = ""


# --- Tag Data Config ---
TAG_JSON_INPUT_DIR = ""
TEMP_OUTPUT_ROOT = ""
PROCESSED_TAGS_FILE_PATH=""
LOCAL_TAG_EMBEDDING_FILE = ""

# --- Item Data Config (Similarity) ---
# THESE DIRECTORIES REMAIN THE ROOT. INPUT/OUTPUT WILL USE CATEGORY SUBFOLDERS.
INPUT_DIR_EMBEDDINGS = ""
OUTPUT_DIR_SIMILARITY = ""
PROCESSED_OUTPUT_DIR = ""
COMBINED_OUTPUT_FILE_PATH = ""
FINAL_FILTERED_OUTPUT_FILE_PATH=""
CONFIDENCE_THRESHOLD = 0.0
THRESHOLD_CSV_FILE_PATH="/app/notebooks/misc/trending_tags_tresholds_test.csv"



# --- DDP Config (No change) ---
NUM_GPUS = -1
BATCH_SIZE = -1
# -----------------------------------------------------------------



def load_universal_config(config_path=UNIVERSAL_CONFIG_PATH):

    global MODEL_ID, LOCAL_MODEL_PATH, CONFIG_FILE_PATH, MODEL_WEIGHTS_PATH, TEMP_OUTPUT_ROOT,PROCESSED_OUTPUT_DIR, PROCESSED_TAGS_FILE_PATH, LOCAL_TAG_EMBEDDING_FILE, NUM_GPUS, TOTAL_WORKERS, BATCH_SIZE

    with open("/app/notebooks/pep_dev_ready/config.json") as f:
        config = json.load(f)
    
    MODEL_ID = config["model_id"]
    LOCAL_MODEL_PATH = config["local_model_path"]
    CONFIG_FILE_PATH = os.path.join(LOCAL_MODEL_PATH, "config.json")
    MODEL_WEIGHTS_PATH = config["model_weights_path"]
    NUM_GPUS = int(config["num_gpus"])
    BATCH_SIZE = config["batch_size"]

    TAG_JSON_INPUT_DIR = config["raw_trends_path"]
    TEMP_OUTPUT_ROOT = config["pre_fitler_tmp"]
    PROCESSED_TAGS_FILE_PATH = config["processed_trends_path"]
    LOCAL_TAG_EMBEDDING_FILE = TEMP_OUTPUT_ROOT + '/all_tags_with_embeddings.parquet'

    
    INPUT_DIR_EMBEDDINGS = config["catalog_token_embeddings_path"]
    
    OUTPUT_DIR_SIMILARITY = f"{config['pre_filter_output_parent_dir']}/solrPlusTagsTitleData_with_SIMILARITY_multitags"
    PROCESSED_OUTPUT_DIR = f"{config['pre_filter_output_parent_dir']}/solrPlusTagsTitleData_PROCESSED_OUTPUT_multitags"
    COMBINED_OUTPUT_FILE_PATH = f"{config['pre_filter_output_parent_dir']}/{config['pre_filter_final_output_file']}"
    FINAL_FILTERED_OUTPUT_FILE_PATH = f"{config['pre_filter_output_parent_dir']}/trending_tags_filtered_multitags.parquet"
    THRESHOLD_CSV_FILE_PATH = config["cateogory_wise_threshold_file"]

    return



# =================================================================
# NEW: OFFLINE MODEL LOADING HELPER (MODIFIED)
# =================================================================

def load_model_offline(local_path, device):
    """
    Loads model and tokenizer strictly from the local path,
    then loads custom fine-tuned weights.
    """
    config_file = os.path.join(local_path, "config.json")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"❌ Required configuration not found at {config_file}. "
            "Model files must be downloaded and saved locally in that directory first."
        )
    
    # 1. Load Architecture
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    model = AutoModel.from_pretrained(local_path, local_files_only=True).to(device)
    print("Successfully loaded the tokenizer and the model")
    
    # 2. Load Custom Weights
    if os.path.exists(MODEL_WEIGHTS_PATH):
        try:
            state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
            # Load state dict, ignoring non-matching keys if strict=False is necessary
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Custom weights loaded from {MODEL_WEIGHTS_PATH}.")
        except Exception as e:
            # Note: If weights were saved as `model.text_model.state_dict()`, 
            # this loading might fail (KeyError). You might need custom logic here.
            print(f"⚠️ Rank {dist.get_rank()}: Failed to load custom weights directly. Using base weights. Error: {e}")
    else:
        print(f"⚠️ Rank {dist.get_rank()}: Custom weights file not found at {MODEL_WEIGHTS_PATH}. Using base architecture weights.")
    
    return model, tokenizer


# =================================================================
# 2. HELPER FUNCTIONS (PREPROCESSING) - UNCHANGED
# =================================================================

# Helper dictionary for number-to-text conversion (can be expanded)
NUM_TO_WORD = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', 
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', 
    '10': 'ten', '1st': 'first', '2nd': 'second', '3rd': 'third',
    'one': 'one', 'two': 'two', 'three': 'three', 'four': 'four', 'five': 'five',
    'six': 'six', 'seven': 'seven', 'eight': 'eight', 'nine': 'nine', 'ten': 'ten'
}
# Set of all final number words that must be protected
PROTECTED_NUMBER_WORDS = set(NUM_TO_WORD.values())

# Words that are short, common, but semantically critical (lemmatized form)
PROTECTED_SEMANTIC_WORDS = {
    # Original words to keep
    'no', 'not', 'mini', 'maxi', 'mid', 'high', 'low', 'free', 'above', 'below', 
    'front', 'back', 'a', 'side', 'sole', 'up', 'down', 'on', 'off', 'through', 
    'and', 'or', 'full', 'half', 'micro', 'piece', 't-shirt', 'hem', 'dress', 
    'skirt', 'pant', 'shoe', 'jacket', 'kurta', 'sneaker', 'coat', 'tee', 
    'frock', 'sweatshirt', 'jean', 'jegging','hot', 'cold',
    
    # **Added words for fashion context**
    
    # Common Prepositions/Adjectives for Fit/Style
    'slim', 'regular', 'loose', 'wide', 'skinny', 'straight', 'cropped', 
    'over', 'under', 'short', 'long', 'cap', 'cowl', 'polo', 'v-neck', 'round', 
    'square', 'boat', 'deep', 'open', 'closed', 'zip', 'button', 'tie', 'faux', 
    'real', 'pure', 'blend', 'classic', 'modern', 'vintage', 'new', 'old',
    'big', 'small', 'plus', 'petite', 'tall', 'basic', 'print', 'solid', 
    'stripe', 'check', 'dot', 'floral', 'graphic', 'hood', 'collar', 'sleeve', 
    'pocket', 'waist', 'ankle', 'calf', 'knee', 'boot', 'chunky', 'platform',
    
    # Specific Garment/Accessory Parts or Types
    'top', 'bottom', 'set', 'pair', 'blouse', 'shirt', 'trouser', 'legging', 
    'jumpsuit', 'romper', 'swimsuit', 'bikini', 'bra', 'brief', 'sock', 'stocking', 
    'scarf', 'shawl', 'hat', 'cap', 'beanie', 'glove', 'mitten', 'belt', 'bag', 
    'clutch', 'tote', 'crossbody', 'earring', 'necklace', 'bracelet', 'ring',
    
    # Materials (crucial modifiers)
    'cotton', 'silk', 'denim', 'wool', 'linen', 'leather', 'suede', 'lace', 
    'knit', 'woven', 'jersey', 'velvet', 'chiffon', 'polyester', 'nylon'
}

# --- NEW ADDITION: Explicit phrase replacements ---
EXPLICIT_PHRASE_REPLACEMENTS = {
    # Existing entries (Preserved)
    'one piece': 'one-piece',
    'two piece': 'two-piece',
    'three piece': 'three-piece',
    '4 piece': 'four-piece',  # Note: Keeping '4 piece' for common usage
    '5 piece': 'five-piece',
    'a line': 'a-line',
    'v neck': 'v-neck',
    'u neck': 'u-neck',
    'indo western': 'indo-western',
    'low top': 'low-top',
    'mid top': 'mid-top',
    'high top': 'high-top',
    
    # **New Additions for Fashion Catalog Accuracy**
    
    # Necklines and Sleeves
    'crew neck': 'crew-neck',
    'scoop neck': 'scoop-neck',
    'boat neck': 'boat-neck',
    'cowl neck': 'cowl-neck',
    'polo neck': 'polo-neck',
    'sweet heart': 'sweetheart',
    'off shoulder': 'off-shoulder',
    'one shoulder': 'one-shoulder',
    'cold shoulder': 'cold-shoulder',
    'cap sleeve': 'cap-sleeve',
    'half sleeve': 'half-sleeve',
    'full sleeve': 'full-sleeve',
    'three fourth': 'three-fourth',
    
    # Fits, Cuts, and Rises
    'regular fit': 'regular-fit',
    'slim fit': 'slim-fit',
    'skinny fit': 'skinny-fit',
    'loose fit': 'loose-fit',
    'boyfriend fit': 'boyfriend-fit',
    'mom fit': 'mom-fit',
    'high waist': 'high-waist',
    'mid rise': 'mid-rise',
    'low rise': 'low-rise',
    'boot cut': 'boot-cut',
    'straight cut': 'straight-cut',
    
    # Clothing and Accessory Types
    'cross body': 'cross-body',
    'beach wear': 'beachwear',
    'foot wear': 'footwear',
    'sleep wear': 'sleepwear',
    'lounge wear': 'loungewear',
    'maxi dress': 'maxi-dress',
    'mini dress': 'mini-dress',
    'pencil skirt': 'pencil-skirt',
    'ball gown': 'ball-gown'
}
# ---------------------------------------------------


def convert_numbers_to_text(word: str) -> str:
    """Converts a digit, ordinal, or number word string to its word representation."""
    return NUM_TO_WORD.get(word, word) 

def add_decade_exceptions(nlp_obj: spacy.Language):
    """Adds specific decade terms as special case tokens to the SpaCy tokenizer."""
    
    decades_to_preserve = [f'{d}s' for d in range(10, 100, 10)] 
    decades_to_preserve.extend(['2000s', '20s', '3D', '2D'])
    print(decades_to_preserve)

    for token_text in decades_to_preserve:
        nlp_obj.tokenizer.add_special_case(
            token_text, 
            [{ORTH: token_text}] 
        )
    print(f"✅ SpaCy Tokenizer exceptions added (ORTH only) for {len(decades_to_preserve)} terms.")

# --- SpaCy Customization Patch ---
def apply_spacy_patches(nlp_obj: spacy.Language):
    """Applies custom lemmatization rules for 'me' and explicitly sets the LEMMA for decades."""
    
    if nlp_obj.vocab.lookups.has_table("lemma_lookup"):
        lemma_lookup = nlp_obj.vocab.lookups.get_table("lemma_lookup")
    else:
        lemma_lookup = {}

    lemma_lookup['me'] = 'me' 
    
    decades_to_preserve = [f'{d}s' for d in range(10, 100, 10)] 
    decades_to_preserve.extend(['2000s', '20s', '3D', '2D'])

    for term in decades_to_preserve:
        lemma_lookup[term] = term 
    
    nlp_obj.vocab.lookups.set_table("lemma_lookup", lemma_lookup)
    print("✅ SpaCy lemma lookup set for 'me' and preserved decades.")


class SpacyLemmatizer:
    """Wrapper class using SpaCy's lemmatization, exposing the internal nlp object."""
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner', 'tok2vec'])
            add_decade_exceptions(self.nlp)
            apply_spacy_patches(self.nlp)

        except OSError:
            print("❌ SpaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'.")
            raise

    def stem(self, word: str) -> str:
        """Mimics the .stem() method using SpaCy's lemmatizer."""
        doc = self.nlp(word)
        return doc[0].lemma_

def load_data_from_json_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file path was not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_category_stop_set(df: pd.DataFrame, lemmatizer_obj: SpacyLemmatizer) -> set:
    """Uses lemmatization to create a set of stop words from category names."""
    category_texts = df['category'].astype(str).str.lower().tolist()
    all_category_words = list(itertools.chain.from_iterable(text.split() for text in category_texts))
    
    # Use the aliased .stem() method for word-by-word lemmatization
    lemmatized_category_words = [lemmatizer_obj.stem(word) for word in all_category_words]
    
    # Custom stop words (expanded)
    custom_words_to_add = ['t', 's', 'shirt', 'hem', 'set', 'dress', 'gown', 'kurta', 'top', 'bottom', 'frock']
    lemmatized_custom_words = [lemmatizer_obj.stem(word) for word in custom_words_to_add]
    
    final_stop_set = set(lemmatized_category_words)
    final_stop_set.update(lemmatized_custom_words)
    return final_stop_set

# --- MODIFIED create_embedding_text (FIXED piece rule) ---
def create_embedding_text(df: pd.DataFrame, lemmatizer_obj: SpacyLemmatizer) -> pd.DataFrame:
    """
    Applies SpaCy lemmatization, deduplication, and all custom rules to create the embedding text.
    """
    # 1. Access the internal nlp object and pre-calculate stop word sets
    nlp_obj = lemmatizer_obj.nlp
    category_stop_set = create_category_stop_set(df, lemmatizer_obj)
    spacy_stop_words = nlp_obj.Defaults.stop_words
    
    def preprocess_tag(tag: str) -> str:
        # A. Initial cleaning and custom replacements
        tag = str(tag)
        
        # FIX: Replace underscores with hyphens
        text = tag.replace('_', '-').lower()
        
        # Rule 1: & to be converted to and
        text = text.replace('&', 'and')
        
        # --- NEW/UPDATED RULE: Explicitly handle X piece phrases (e.g., "two piece") ---
        # Replace the phrase with a single hyphenated token to preserve it fully.
        for k, v in EXPLICIT_PHRASE_REPLACEMENTS.items():
            text = text.replace(k, v)


        # Step 3: Pass the raw, modified text directly to SpaCy
        doc = nlp_obj(text)
        
        final_lemmas = []
        for i, token in enumerate(doc):
            token_text = token.text
            
            # 1. Determine the lemma (or number conversion)
            is_number_like = token_text.isdigit() or \
                             token_text.endswith(('st', 'nd', 'rd', 'th')) or \
                             token_text in NUM_TO_WORD
            
            if is_number_like:
                lemma = convert_numbers_to_text(token_text)
            else:
                # Use standard SpaCy lemmatization for non-numbers
                lemma = token.lemma_

            # 2. Determine if the word should be protected
            # Check against protected number set (which contains converted words like 'two')
            is_protected_number = lemma in PROTECTED_NUMBER_WORDS
            
            # Check against the new protected semantic word set ('no', 'mini', etc.)
            is_protected_semantic = lemma in PROTECTED_SEMANTIC_WORDS
            
            # 3. Apply standard stop word checks
            is_stop_word = token_text in spacy_stop_words or lemma in spacy_stop_words
            is_category_stop = lemma in category_stop_set

            # 4. Final Inclusion Check
            # Rule 2: Keep 'and'/'or' unless at the end
            if token_text in ('and', 'or') and i == len(doc) - 1:
                continue 
            
            # Decide to keep the token if:
            # a) It's a protected number OR protected semantic word (overrides stop lists)
            # b) It's not punctuation/space AND NOT in any stop list.
            
            should_keep = is_protected_number or is_protected_semantic or \
                          (not is_stop_word and not is_category_stop)

            if not token.is_punct and not token.is_space and should_keep:
                
                final_lemmas.append(lemma)
            
        # D. Deduplicate and Final Join
        unique_lemmatized_words = list(dict.fromkeys(final_lemmas))
        return " ".join(unique_lemmatized_words)

    def make_embedding_text(row):
        attribute_value = str(row['normalised_name']).strip().lower()
        attribute_name  = str(row['standardize_type_updated']).strip().lower()
    
        if attribute_value == 'sleeveless':
            return row['normalised_name']
        elif attribute_name == 'shop the vibe':
            return row['normalised_name']
        elif attribute_name == 'product title':
            return row['normalised_name']
        elif attribute_name == 'seasonality':
            return row['normalised_name']
        elif attribute_name == 'season':
            return row['normalised_name']
        elif attribute_name == 'functional':
            return row['normalised_name']
        elif attribute_name == 'sustainability':
            return row['normalised_name']
        else:
            return f"{row['normalised_name']} {row['standardize_type_updated']}"

    df['embedding_text_raw'] = df.apply(make_embedding_text, axis=1)
    
    # 2. Create the raw text column 
    # df['embedding_text_raw'] = df.apply(
    #    lambda row: row['normalised_name']
    #    if str(row['standardize_type_updated']).strip().lower() in ['shop the vibe', 'product title','seasonality','functional']
    #    else f"{row['normalised_name']} {row['standardize_type_updated']}",
    #    axis=1
    # )
    
    # 3. Apply the preprocessing function
    tqdm.pandas(desc="Lemmatizing & Filtering Tags with SpaCy")
    df['embedding_text'] = df['embedding_text_raw'].progress_apply(preprocess_tag)

    # 4. Clean up
    df = df.drop(columns=['embedding_text_raw'])
    
    return df

def create_tag_embedding_text(df: pd.DataFrame, lemmatizer_obj: SpacyLemmatizer) -> pd.DataFrame:
    """Applies simplified lemmatization and deduplication to the 'tag' column."""
    def preprocess_tag(tag: str) -> str:
        text = tag.replace('_', ' ').lower()
        words = text.split()
        
        # 1. Apply number-to-word conversion
        converted_words = [convert_numbers_to_text(word) for word in words]

        # 2. Lemmatize words
        lemmatized_words = [lemmatizer_obj.stem(word) for word in converted_words]
        
        # 3. Deduplicate
        unique_lemmatized_words = list(dict.fromkeys(lemmatized_words))
        return " ".join(unique_lemmatized_words)

    df['preprocessed_tag'] = df['tag'].apply(preprocess_tag)
    df['tag_embedding_text'] = df['preprocessed_tag']
    return df

def generate_embeddings(texts, model, tokenizer, device, rank, gpu_id, input_file_path):
    """
    Tokenizes and generates embeddings for a list of texts in batches.
    Uses model.text_model for forward pass.
    """
    embeddings_list = []
    model.eval()
    num_texts = len(texts)
    
    # ... (DEBUG prints remain unchanged) ...
    if rank == 0:
        print(f"| R{rank} DEBUG | Starting embedding generation for file: {os.path.basename(input_file_path)}")
        print(f"| R{rank} DEBUG | Total texts to embed: {num_texts}")
        if any(t is None for t in texts[:min(5, num_texts)]):
             print(f"| R{rank} DEBUG | WARNING: First {min(5, num_texts)} texts contain None or invalid values.")


    num_batches = (num_texts + BATCH_SIZE - 1) // BATCH_SIZE
    filename = os.path.basename(input_file_path)

    batch_iterator = tqdm(
        range(num_batches), 
        desc=f"R{rank} (GPU{gpu_id}): Embed {num_texts} texts for {filename}", 
        disable=(rank != 0)
    )

    for batch_index in batch_iterator:
        i = batch_index * BATCH_SIZE
        j = i + BATCH_SIZE
        batch_texts = texts[i:j]
        
        safe_batch_texts = [str(t) for t in batch_texts if t is not None]

        if not safe_batch_texts:
            if rank == 0:
                 print(f"| R{rank} DEBUG | WARNING: Batch {batch_index} is empty or contains only None values. Skipping.")
            continue


        try:
            encoded_input = tokenizer(
                safe_batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            ).to(device)
            
            if rank == 0 and batch_index == 0:
                print(f"| R{rank} DEBUG | Input IDs shape: {encoded_input['input_ids'].shape}")

        except Exception as e:
            if rank == 0:
                print(f"\n| R{rank} DEBUG | ❌ CRITICAL TOKENIZATION FAILURE at batch {batch_index}: {e}")
            raise e
        
        try:
            with torch.no_grad():
                # 🔑 CRITICAL MODIFICATION: Use model.text_model here
                # We assume model.text_model is the core transformer (e.g., BERT, ELECTRA)
                # and returns a ModelOutput, whose first element is the last hidden state.
                # If model is *not* a Marqo model (AutoModel generic wrapper), this will throw AttributeError.
                
                # Check for existence of text_model first (optional but safer)
                if hasattr(model, 'text_model') and model.text_model is not None:
                    model_output = model.text_model(**encoded_input)
                else:
                    # Fallback to direct model call if text_model doesn't exist/is None
                    model_output = model(**encoded_input)
                    if rank == 0 and batch_index == 0:
                        print("| R0 DEBUG | WARNING: 'text_model' not found on AutoModel. Using direct model(**encoded_input).")


            # 🔑 CRITICAL MODIFICATION: Extract the last hidden state (the sequence of token vectors)
            # This handles the case whether it's the full AutoModel output (outputs[0]) 
            # or the direct transformer output (which is often just the tensor).
            # The safer bet for a custom architecture is assuming the output is indexable and the first element is the hidden state.
            if isinstance(model_output, tuple):
                 token_embeddings = model_output[0]
            elif hasattr(model_output, 'last_hidden_state'):
                 token_embeddings = model_output.last_hidden_state
            else:
                 token_embeddings = model_output # Assume it's the tensor directly
            
            if rank == 0 and batch_index == 0:
                 print(f"| R{rank} DEBUG | Model output tensor shape: {token_embeddings.shape}")
                
        except Exception as e:
            if rank == 0:
                print(f"\n| R{rank} DEBUG | ❌ CRITICAL MODEL FORWARD PASS FAILURE at batch {batch_index}: {e}")
            raise e

        attention_mask = encoded_input['attention_mask']
        
        # --- Mean Pooling Calculation ---
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        mean_embeddings = sum_embeddings / sum_mask
        mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
        
        embeddings_list.append(mean_embeddings.cpu().numpy())
        
    return [item for sublist in embeddings_list for item in sublist]

def load_tag_embeddings(file_path):
    """Loads tag embeddings and prepares them for matrix multiplication."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Tag embeddings file not found at: {file_path}")
        
    tag_df = pd.read_parquet(file_path)
    # Ensure only unique tags are used for the matrix
    tag_df_unique = tag_df.drop_duplicates(subset=['tag'])
    tag_embeddings = np.stack(tag_df_unique['tag_embedding_vector'].to_numpy())
    tag_embeddings_T = tag_embeddings.T 
    tag_names = tag_df_unique['tag'].tolist()
    
    return tag_embeddings_T, tag_names
    
# =================================================================
# 3. DDP WORKER FUNCTION (SIMILARITY) - MODIFIED FOR ROBUST WEIGHTED SIMILARITY
# =================================================================

def similarity_worker(rank, world_size, all_input_files, temp_output_dir, target_category): 
    """
    Function executed by each DDP process for similarity calculation.
    Calculates weighted similarity: Cosine(Item, Tag) * LLM_Confidence.
    Includes a robust check for the LLM Confidence column.
    """
    # Define the expected confidence column name 
    # 🚨 CRITICAL: CHANGE THIS TO THE ACTUAL COLUMN NAME IN YOUR PARQUET FILES IF IT IS DIFFERENT!
    CONFIDENCE_COLUMN = 'confidence' 
    
    # 0. DDP Initialization
    gpu_id = rank
    device = torch.device(f"cuda:{gpu_id}")
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=f'env://')
    torch.cuda.set_device(gpu_id)

    # 1. Load Tag Embeddings
    if rank == 0:
        print(f"Rank {rank}: Loading Tag Embeddings from {LOCAL_TAG_EMBEDDING_FILE}...")
    try:
        tag_embeddings_T, tag_names = load_tag_embeddings(LOCAL_TAG_EMBEDDING_FILE)
        tag_embeddings_T_gpu = torch.from_numpy(tag_embeddings_T).float().to(device)
    except FileNotFoundError as e:
        if rank == 0: print(f"❌ Initialization Error: {e}")
        dist.destroy_process_group()
        return

    # 2. Data Distribution Logic
    num_files = len(all_input_files)
    my_files = all_input_files[rank:num_files:world_size]
    
    if rank == 0:
        print(f"Total files: {num_files}. Files assigned to Rank {rank}: {len(my_files)}")
    
    # 3. Process files
    processed_count = 0
    file_iterator = tqdm(my_files, desc=f"R{rank} (GPU{gpu_id}) processing files", disable=(rank != 0))

    for input_file_path in file_iterator:
        filename = os.path.basename(input_file_path)
        
        output_file_path = os.path.join(temp_output_dir, f"rank{rank}_{filename}") 

        try:
            # a. Read and Filter Data (using passed target_category)
            df = pd.read_parquet(input_file_path)
            
            df_filtered = df[df['category'] == target_category].copy()

            print(df_filtered.head(5))
            
            if df_filtered.empty:
                del df; gc.collect()
                continue
            
            # --- Robust LLM Confidence Extraction ---
            if 'text_embedding' not in df_filtered.columns:
                if rank == 0: 
                    print(f"\n❌ Error in R{rank}: Input file {filename} is missing 'text_embedding'. Skipping file.")
                del df, df_filtered; gc.collect()
                continue
                
            if CONFIDENCE_COLUMN not in df_filtered.columns:
                # Fallback: If column is missing, default confidence to 1.0 (standard cosine similarity)
                if rank == 0: 
                    print(f"\n⚠️ Warning in R{rank}: Column '{CONFIDENCE_COLUMN}' not found in {filename}. Defaulting confidence to 1.0.")
                    
                # Create a temporary column with all 1.0s
                llm_confidence_np = np.ones(len(df_filtered), dtype=np.float32)
            else:
                # Extract LLM Confidence. Convert to numpy float32
                llm_confidence_np = df_filtered[CONFIDENCE_COLUMN].to_numpy().astype(np.float32)
                
            # Convert to a GPU tensor and reshape for broadcasting (Nx1)
            llm_confidence_gpu = torch.from_numpy(llm_confidence_np).float().to(device).unsqueeze(1)
            
            # b. Calculate Cosine Similarity
            item_embeddings_np = np.stack(df_filtered['text_embedding'].to_numpy())
            item_embeddings_gpu = torch.from_numpy(item_embeddings_np).float().to(device)
            
            with torch.no_grad():
                # Cosine Similarity Matrix (N_items x N_tags)
                similarity_matrix_gpu = torch.matmul(item_embeddings_gpu, tag_embeddings_T_gpu)
                
                # c. Apply Weighting: Weighted Similarity = Cosine Similarity * LLM Confidence
                weighted_similarity_matrix_gpu = similarity_matrix_gpu * llm_confidence_gpu
            
            weighted_similarity_matrix_np = weighted_similarity_matrix_gpu.cpu().numpy()
            
            # d. Create Similarity DataFrame using the weighted scores
            similarity_df = pd.DataFrame(
                weighted_similarity_matrix_np,
                columns=[f'sim_{tag}' for tag in tag_names],
                index=df_filtered.index
            )
            
            # Merge with the filtered item data
            df_final = df_filtered.merge(similarity_df, left_index=True, right_index=True)
            
            # Drop the embedding column
            if 'text_embedding' in df_final.columns:
                df_final = df_final.drop(columns=['text_embedding'])
            
            # Save intermediate file to temp dir
            df_final.to_parquet(output_file_path, index=False)

            # f. Cleanup
            del df, df_filtered, df_final, item_embeddings_np, item_embeddings_gpu, similarity_matrix_gpu, weighted_similarity_matrix_gpu, weighted_similarity_matrix_np, similarity_df, llm_confidence_np, llm_confidence_gpu
            gc.collect()
            torch.cuda.empty_cache()
            processed_count += 1

        except Exception as e:
            if rank == 0:
                # Add more context to the general exception printout
                print(f"\n⚠️ Rank {rank} FAILED to process file {filename} with unexpected error: {e}")
                # For debugging purposes, you might want to uncomment the line below 
                # to see the traceback if the error is not a KeyError:
                # import traceback; traceback.print_exc()
            continue

    # 4. Final Cleanup
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"✅ Rank 0 finished. Processed {processed_count} files to temp dir.")

# =================================================================
# 4. STANDALONE TAG EMBEDDING GENERATION - NO FUNCTIONAL CHANGE
# =================================================================

def generate_tag_embeddings_standalone(tags_df_with_metadata: pd.DataFrame):
    """
    Loads model once on a single GPU (or CPU) and generates embeddings 
    for the tags DataFrame using SpaCy lemmatization.
    """
    print("\n--- Starting Tag Embedding Generation ---")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    lemmatizer_obj = SpacyLemmatizer() 
    lemmatizer_obj
    
    # Isolate unique tags for embedding generation
    unique_tags_for_embedding = tags_df_with_metadata[['tag', 'tag_embedding_text']].drop_duplicates().reset_index(drop=True)

    # 1. Check/Download Model
    if not os.path.exists(CONFIG_FILE_PATH):
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        print(f"⬇️ Downloading model and tokenizer for '{MODEL_ID}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            tokenizer.save_pretrained(LOCAL_MODEL_PATH)
            model = AutoModel.from_pretrained(MODEL_ID)
            model.save_pretrained(LOCAL_MODEL_PATH)
            print("✅ Download complete.")
        except Exception as e:
            print(f"❌ An error occurred during download: {e}")
            return tags_df_with_metadata
    else:
        print(f"✅ Model found at {LOCAL_MODEL_PATH}.")
    
    # try:
    #     tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    #     model = AutoModel.from_pretrained(LOCAL_MODEL_PATH).to(device)
    # except Exception as e:
    #     print(f"❌ ERROR loading model for standalone job: {e}")
    #     return tags_df_with_metadata

    try:
        model, tokenizer = load_model_offline(LOCAL_MODEL_PATH, device)
    except Exception as e:
        print(f"❌ ERROR loading model for standalone job: {e}")
        return tags_df_with_metadata



    # 2. Preprocess texts and Generate Embeddings
    # unique_tags_for_embedding = create_tag_embedding_text(unique_tags_for_embedding, lemmatizer_obj)
    
    texts_to_embed = unique_tags_for_embedding['tag_embedding_text'].tolist()
    print(f"Generating embeddings for {len(texts_to_embed)} unique tags on {device}...")
    
    embeddings = generate_embeddings(
        texts_to_embed, model, tokenizer, device, rank=0, gpu_id=0, input_file_path="TAG_DATA_FRAME.parquet"
    )

    # 3. Attach and Save
    unique_tags_for_embedding['tag_embedding_vector'] = list(embeddings)
    
    # Merge the vectors back to the original full tags_df (which contains 'region', 'category' etc.)
    tags_df_final = tags_df_with_metadata.merge(
        unique_tags_for_embedding[['tag', 'tag_embedding_vector', 'tag_embedding_text']],
        on='tag',
        how='left'
    )
    
    os.makedirs(os.path.dirname(LOCAL_TAG_EMBEDDING_FILE), exist_ok=True)
    tags_df_final.to_parquet(LOCAL_TAG_EMBEDDING_FILE, index=False)
    print(f"✅ Tag embeddings generated and saved to: {LOCAL_TAG_EMBEDDING_FILE}")
    
    del model, tokenizer, lemmatizer_obj; gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
        
    return tags_df_final

# =================================================================
# 5. POST-PROCESSING ANALYSIS FUNCTION - NO FUNCTIONAL CHANGE
# =================================================================

def post_processing_analysis(final_similarity_file: str, target_category: str): 
    """
    Performs the requested analysis on the single, aggregated similarity file.
    """
    print(f"\n--- Phase 4: Starting Post-Processing Analysis for {target_category} ---")
    
    if not os.path.exists(final_similarity_file):
        print(f"❌ Cannot start analysis: Final similarity file not found at {final_similarity_file}")
        return

    # --- Load Data ---
    df = pd.read_parquet(final_similarity_file)
    print(f"Loaded {df.shape[0]} rows for analysis.")
    
    # Load Tag Metadata (contains 'tag', 'region', 'tag_embedding_text', 'category')
    try:
        all_tags_df = pd.read_parquet(LOCAL_TAG_EMBEDDING_FILE).rename(columns={'tag_embedding_text_x':'tag_embedding_text'})
        print(all_tags_df.head())
    except FileNotFoundError:
        print(f"❌ Tag metadata file not found at {LOCAL_TAG_EMBEDDING_FILE}. Skipping analysis.")
        return
        
    # Filter the tag metadata to the current category 
    print(all_tags_df[all_tags_df['category'] == target_category])
    df_tag_meta = all_tags_df[all_tags_df['category'] == target_category][['tag', 'region', 'tag_embedding_text']].copy()
    
    # -----------------------------------------------------------
    # 2. Max Similarity Grouping and Melting
    # -----------------------------------------------------------
    
    sim_cols = [c for c in df.columns if c.startswith("sim_")]
    
    df_sim = (
        df[['product_id'] + sim_cols]
        .groupby("product_id")[sim_cols]
        .max()
        .reset_index()
    )
    
    print(f"\nGrouped to unique product_ids. Total unique products: {df_sim.shape[0]}")
    
    df_long = df_sim.melt(
        id_vars=["product_id"],
        value_vars=sim_cols,
        var_name="tag_raw",
        value_name="confidence"
    )

    # NOTE: The "confidence" column here now refers to the *weighted* similarity score.
    df_long = df_long[df_long["confidence"] > CONFIDENCE_THRESHOLD].copy()
    print(f"Filtered to {df_long.shape[0]} product-tag pairs with weighted confidence > {CONFIDENCE_THRESHOLD}.")

    df_long["tag"] = (
        df_long["tag_raw"]
        .str.replace("sim_", "", regex=False)
    )
    
    # Prepare tag metadata for merge (using only the unique entries for this category)
    df_tag_meta_unique = df_tag_meta.drop_duplicates(subset=['tag'])
    
    df_final = df_long.merge(
        df_tag_meta_unique[['tag', 'region', 'tag_embedding_text']], 
        on="tag",
        how="left"
    )

    df_final = df_final.drop(columns=["tag_raw"])
    df_final = df_final[['product_id', 'confidence', 'tag', 'region', 'tag_embedding_text']]
    
    print("\n--- Final Processed Data Head ---")
    print(df_final.head())

    # -----------------------------------------------------------
    # 3. Save Final Processed Output
    # -----------------------------------------------------------
    FINAL_PROCESSED_DIR = os.path.join(PROCESSED_OUTPUT_DIR, target_category)
    os.makedirs(FINAL_PROCESSED_DIR, exist_ok=True)
    
    FINAL_PROCESSED_FILE = os.path.join(
        FINAL_PROCESSED_DIR, 
        f"{target_category.replace(' ', '_')}_processed_tags.parquet"
    )
    
    df_final.to_parquet(FINAL_PROCESSED_FILE, index=False)
    print(f"\n✅ Final processed results saved to: {FINAL_PROCESSED_FILE}")

import pandas as pd
import glob
import os
from typing import List
def get_category_from_path(file_path: str, base_dir: str) -> str:
    """
    Extracts the direct subdirectory name between the base_dir and the filename.
    
    Example:
    file_path = "/data/pep/solrPlusTagsTitleData_PROCESSED_OUTPUT/category_a/file.parquet"
    base_dir = "/data/pep/solrPlusTagsTitleData_PROCESSED_OUTPUT/"
    Returns: "category_a"
    """
    # Normalize paths and ensure base_dir ends with a separator
    base_dir = os.path.join(base_dir, '') 
    
    # Get the part of the path relative to the base directory
    relative_path = os.path.relpath(file_path, base_dir)
    
    # The category is the first component of the relative path
    # which is the direct subdirectory name containing the .parquet file
    category = relative_path.split(os.sep)[0]
    
    # If the file is directly in the base_dir, the relative path might just be the filename
    if category.endswith('.parquet'):
         return "base_level" # Or any default category name you prefer for files in the root
         
    return category


# Aggregate all outputs into a single parquet
def aggregate_and_filter_parquets(source_dir: str, target_file: str, filter_col: str):
    """
    Recursively finds all Parquet files, reads them, filters for non-null
    region values, concatenates them, and writes the result to a new file.

    Args:
        source_dir (str): The root directory to search.
        target_file (str): The full path for the output Parquet file.
        filter_col (str): The column to check for non-null values.
    """
    # Create the full search pattern for all .parquet files in the directory and subdirectories
    search_pattern = os.path.join(source_dir, '**', '*.parquet')
    
    # Use glob.glob with recursive=True to find all matching file paths
    print(f"Searching for Parquet files in: {search_pattern}")
    parquet_files: List[str] = glob.glob(search_pattern, recursive=True)
    
    if not parquet_files:
        print("No Parquet files found. Exiting.")
        return

    print(f"Found {len(parquet_files)} Parquet files.")
    
    # List to hold the filtered DataFrames from each file
    filtered_data_frames: List[pd.DataFrame] = []
    
    # Process each file
    for i, file_path in enumerate(parquet_files):
        try:
            # 1. Read the Parquet file
            df = pd.read_parquet(file_path)

            # --- MODIFICATION START: Add 'category' column ---
            
            # Extract the category name
            category_name = get_category_from_path(file_path, source_dir)
            
            # Add the new 'category' column
            df['category'] = category_name
            
            # --- MODIFICATION END ---
            
            # Check if the filter column exists in the DataFrame
            if filter_col not in df.columns:
                print(f"Warning: Skipping file {file_path}. Column '{filter_col}' not found.")
                continue

            # 2. Apply the filter: region is not null
            initial_count = len(df)
            filtered_df = df[df[filter_col].notnull()]
            filtered_count = len(filtered_df)
            
            print(f"  [{i+1}/{len(parquet_files)}] Processed: {file_path} (Initial: {initial_count}, Filtered: {filtered_count})")
            
            # Add the filtered DataFrame to the list
            filtered_data_frames.append(filtered_df)
            
        except Exception as e:
            print(f"Error reading or processing file {file_path}: {e}")

    # 3. Take a union across each of them (Concatenate all filtered DataFrames)
    if not filtered_data_frames:
        print("No data remaining after filtering. Nothing to write.")
        return

    print("\nConcatenating all filtered dataframes...")
    final_df = pd.concat(filtered_data_frames, ignore_index=True)
    
    print(f"Total rows in final aggregated DataFrame: {len(final_df)}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(target_file), exist_ok=True)

    # 4. Write a single parquet to the target location
    print(f"Writing final DataFrame to: {target_file}")
    final_df.to_parquet(target_file, index=False)
    
    print("Process complete! Aggregated Parquet file successfully created.")


def drop_rows_below_confidence_threshold(input_df: pd.DataFrame, csv_file_path: str) -> pd.DataFrame:
    """
    Drops rows from the input DataFrame where the 'confidence' is less than 
    the category-specific threshold loaded from a CSV file.
    
    Args:
        input_df (pd.DataFrame): The DataFrame with 'confidence' and 'category' columns.
        csv_file_path (str): Path to the CSV file with 'category' and 'threshold' columns.
        
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    print("--- Applying Confidence Filter ---")

    if not os.path.exists(csv_file_path):
        print(f"Error: Threshold CSV not found at {csv_file_path}. Returning original DataFrame.")
        return input_df

    # 1. Read the thresholds
    threshold_df = pd.read_csv(csv_file_path)
    
    # Ensure the threshold column is numeric
    threshold_df['threshold'] = pd.to_numeric(threshold_df['threshold'], errors='coerce')
    threshold_df = threshold_df.dropna(subset=['threshold'])
    
    print("Loaded Thresholds:")
    print(threshold_df)
    
    # 2. Merge the thresholds into the input DataFrame
    # Note: Using how='left' ensures all original rows are kept, and missing thresholds are NaN
    df_merged = pd.merge(
        input_df, 
        threshold_df, 
        on='category', 
        how='left'
    )
    
    # Handle rows where category has no defined threshold (e.g., set threshold to 0 to keep all, or 1 to drop all)
    # Here, we set the threshold to a very low value (e.g., 0) so rows without a defined threshold are kept by default.
    df_merged['threshold'] = df_merged['threshold'].fillna(0.0)
    
    # 3. Apply the filter: keep rows where 'confidence' >= 'threshold'
    initial_count = len(df_merged)
    
    # The condition: confidence >= threshold
    filtered_df = df_merged[df_merged['confidence'] >= df_merged['threshold']].copy()
    
    final_count = len(filtered_df)
    
    print(f"\nInitial rows: {initial_count}")
    print(f"Final rows after filtering: {final_count}")
    print(f"Rows dropped: {initial_count - final_count}")
    print("Filtering complete.")
    print("---------------------------------")
    
    # Remove the temporary 'threshold' column before returning
    return filtered_df.drop(columns=['threshold'])


# =================================================================
# 6. MAIN EXECUTION FLOW - NO FUNCTIONAL CHANGE
# =================================================================

def main():
    load_universal_config()
    
    # Initial cleanup of global tag embedding file
    if os.path.exists(LOCAL_TAG_EMBEDDING_FILE):
        print(f"🧹 Initial cleanup: Deleting old tag embedding file: {LOCAL_TAG_EMBEDDING_FILE}")
        os.remove(LOCAL_TAG_EMBEDDING_FILE)
    
    # 1. Find all JSON files (to get the list of categories)
    tag_json_files = glob.glob(os.path.join(TAG_JSON_INPUT_DIR, "*.json"))
    
    if not tag_json_files:
        print(f"❌ No JSON files found in {TAG_JSON_INPUT_DIR}. Exiting.")
        return

    # 2. Load all tags across all categories to generate a single embeddings file
    print("--- PHASE 1: Loading ALL Tags and Generating/Loading Embeddings ---")
    
    all_tags_data = []
    category_list = []
    
    for tag_file_path in tag_json_files:
        tag_filename = os.path.basename(tag_file_path)
        category_name = tag_filename.replace('.json', '')
        category_list.append(category_name)
        
        try:
            data = load_data_from_json_file(tag_file_path)
            
            df_india = pd.DataFrame(data.get('trending_tags_india', []))
            df_india['region'] = 'india'
            df_global = pd.DataFrame(data.get('trending_tags_global', []))
            df_global['region'] = 'global'
            
            df_india['category'] = category_name
            df_global['category'] = category_name
            
            all_tags_data.append(pd.concat([df_india, df_global], ignore_index=True))
        except Exception as e:
            print(f"Skipping file {tag_filename}: {e}")
            continue

    if not all_tags_data:
        print("❌ No tag data loaded successfully. Exiting.")
        return

    # tags_df = pd.concat(all_tags_data, ignore_index=True)
    #tags_df.to_parquet(PROCESSED_TAGS_FILE_PATH, index=False)
    tags_df = pd.read_csv(PROCESSED_TAGS_FILE_PATH)
    print(tags_df.head())
    tags_df_with_embeddings = generate_tag_embeddings_standalone(tags_df) 
    print(tags_df.head())

    print(f"Total unique tags with metadata loaded/generated: {tags_df_with_embeddings.shape[0]}")
    
    # 3. Iterate through categories and run DDP Similarity
    print("\n--- Starting Iteration over Target Categories ---")

    for target_category in category_list:
        
        FINAL_PROCESSED_DIR_CHECK = os.path.join(PROCESSED_OUTPUT_DIR, target_category)
        if os.path.exists(FINAL_PROCESSED_DIR_CHECK):
            print(f"\n⏭️ Skipping category '{target_category}': Final processed output directory already exists at {FINAL_PROCESSED_DIR_CHECK}")
            continue

            
        print(f"\n\n=======================================================")
        print(f"🚀 Processing Target Category: {target_category}")
        print(f"=======================================================")

        # --- PHASE 2: DDP Similarity Calculation ---
        
        FINAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR_SIMILARITY, target_category)
        FINAL_OUTPUT_FILE = os.path.join(FINAL_OUTPUT_DIR, f"{target_category.replace(' ', '_')}_similarity_results.parquet")
        
        TEMP_WORKING_DIR = os.path.join(TEMP_OUTPUT_ROOT, f"{target_category.replace(' ', '_')}_temp")
        if os.path.exists(TEMP_WORKING_DIR):
            print(f"⚠️ Deleting existing temporary directory: {TEMP_WORKING_DIR}")
            shutil.rmtree(TEMP_WORKING_DIR)
        os.makedirs(TEMP_WORKING_DIR, exist_ok=True)
        
        os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True) 
        
        # 🔑 MODIFICATION: Input file search is now category-partitioned
        # category_input_dir = os.path.join(INPUT_DIR_EMBEDDINGS, target_category.replace(' ', '_'))
        category_input_dir = os.path.join(INPUT_DIR_EMBEDDINGS, target_category)
        all_input_files = glob.glob(os.path.join(category_input_dir, "*.parquet"))
        files_to_process = all_input_files

        if not files_to_process:
            print(f"✅ No item files found in category-partitioned directory: {category_input_dir}")
            if os.path.exists(TEMP_WORKING_DIR): shutil.rmtree(TEMP_WORKING_DIR)
            continue

        print(f"Output directory: {FINAL_OUTPUT_DIR}")
        print(f"Temp directory: {TEMP_WORKING_DIR}")
        print(f"Processing {len(files_to_process)} total item files with DDP.")

        try:
            mp.set_start_method('spawn', force=True) 
        except RuntimeError:
            pass 

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355' 
        world_size = NUM_GPUS

        try:
            mp.spawn(
                similarity_worker,
                args=(world_size, files_to_process, TEMP_WORKING_DIR, target_category), 
                nprocs=world_size,
                join=True
            )
        except Exception as e:
            print(f"🔥 DDP Spawn failed for {target_category}: {e}")
            if os.path.exists(TEMP_WORKING_DIR): shutil.rmtree(TEMP_WORKING_DIR)
            continue
            
        # --- PHASE 3: AGGREGATE AND SAVE ---
        print("\n--- Phase 3: Aggregating Results and Saving Single File ---")
        
        temp_files = glob.glob(os.path.join(TEMP_WORKING_DIR, "*.parquet"))
        
        if not temp_files:
            print("⚠️ No temporary data files were found after DDP completion. Skipping aggregation.")
            if os.path.exists(TEMP_WORKING_DIR): shutil.rmtree(TEMP_WORKING_DIR)
            continue

        all_results = [pd.read_parquet(f) for f in tqdm(temp_files, desc="Reading temp files")]
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_parquet(FINAL_OUTPUT_FILE, index=False)

        print(f"✅ Aggregation complete. Total rows: {final_df.shape[0]}.")
        print(f"✅ Single final Parquet file saved to: {FINAL_OUTPUT_FILE}")
        
        # --- PHASE 4: POST-PROCESSING ANALYSIS ---
        post_processing_analysis(FINAL_OUTPUT_FILE, target_category)

        # 5. Clean up the temporary directory for this category's intermediate results
        print(f"🧹 Deleting temporary directory: {TEMP_WORKING_DIR}")
        shutil.rmtree(TEMP_WORKING_DIR)
        
    # Final cleanup of global tag embedding file
    if os.path.exists(LOCAL_TAG_EMBEDDING_FILE):
        print(f"\n🧹 Final cleanup: Deleting tag embedding file after all categories: {LOCAL_TAG_EMBEDDING_FILE}")
        os.remove(LOCAL_TAG_EMBEDDING_FILE)

    
        
    print(f"\n✅ All categories processed. Now generating a single unified output file")
        # Ensure the source directory is an absolute path and ends with a slash for clarity
    if not os.path.isabs(PROCESSED_OUTPUT_DIR):
        print("Error: SOURCE_DIR must be an absolute path.")
    elif not os.path.exists(PROCESSED_OUTPUT_DIR):
        print(f"Error: Source directory not found at {SOURCE_DIR}. Please check the path.")
    else:
        FILTER_COLUMN = 'region'
        aggregate_and_filter_parquets(PROCESSED_OUTPUT_DIR, COMBINED_OUTPUT_FILE_PATH, FILTER_COLUMN)
        print(f"Merged output written to {COMBINED_OUTPUT_FILE_PATH}.")

    # Now post process the output
    output_df = pd.read_parquet(COMBINED_OUTPUT_FILE_PATH)
    output_df.head()
        
    
    # Drop flags whose confidence is below the tag specific threshold
    print("\nDropping flags below confidence threshold")
    filtered_df = drop_rows_below_confidence_threshold(output_df.copy(), THRESHOLD_CSV_FILE_PATH)
    
    filtered_df.head()
    
    tags_to_drop = ['square_toe_shape','abstract_motif_print','heritage_motif_print','mirror_work_embellishment','jacket_style','sharara_bottom']
    tags_to_drop_without_underscores = [tag.replace('_', ' ') for tag in tags_to_drop]
    tags_to_drop.extend(tags_to_drop_without_underscores)
    tags_to_drop
    
    print("\nDropping tags: ", tags_to_drop)
    filtered_df = filtered_df[~filtered_df['tag'].isin(tags_to_drop)]
    
    print("\nAdding tag descriptions")
    tags_df = pd.read_csv(PROCESSED_TAGS_FILE_PATH)
    tags_df.head()
    
    
    final_df = filtered_df.merge(tags_df[['tag_embedding_text','region','category','description']],
                     how='left',
                    on=['tag_embedding_text','region','category']
        )
    no_desc_tags = len( final_df[final_df['description'].isna() | (final_df['description'].str.strip() == '')] )
    print(f"{no_desc_tags} tags with no description found")
    
    
    print("\nFinal DataFrame Categories and Counts:")
    print(final_df['category'].value_counts())
    
    
    # write the final df
    final_df.to_parquet(FINAL_FILTERED_OUTPUT_FILE_PATH, index=False)

if __name__ == "__main__":
    main()