import pandas as pd
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
import re 
import json

# --- NEW SPAcY IMPORTS ---
import spacy
from typing import Set
from spacy.symbols import ORTH, LEMMA 
# --- END NEW IMPORTS ---

UNIVERSAL_CONFIG_PATH = "/app/notebooks/pep_dev_ready/config.json"

# Set this environment variable to mitigate memory fragmentation issues
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# =================================================================
# 1. CONFIGURATION AND PATHS
# =================================================================
# MODEL_ID = "Marqo/marqo-ecommerce-embeddings-L"
# LOCAL_MODEL_PATH = "/data/models/marqo-ecommerce-L"
# CONFIG_FILE_PATH = os.path.join(LOCAL_MODEL_PATH, "config.json")
# MODEL_WEIGHTS_PATH = "/data/models/marqo-ecomm-embeddings-l.pt"

# # These are now the root directories containing category subfolders
# INPUT_ROOT_DIR = "/data/pep/solrPlusTagsTitleData"
# OUTPUT_ROOT_DIR = "/data/pep/solrPlusTagsTitleData_with_embeddings"
# NUM_GPUS = 4 # Total number of GPUs available/to use
# TOTAL_WORKERS = NUM_GPUS # Each worker is a DDP rank
# BATCH_SIZE = 512

MODEL_ID = ""
LOCAL_MODEL_PATH = ""
CONFIG_FILE_PATH = ""
MODEL_WEIGHTS_PATH = ""

# These are now the root directories containing category subfolders
INPUT_ROOT_DIR = ""
OUTPUT_ROOT_DIR = ""
NUM_GPUS = 0 # Total number of GPUs available/to use
TOTAL_WORKERS = 0 # Each worker is a DDP rank
BATCH_SIZE = 0
# ----------------------------------------------------

def load_universal_config(config_path=UNIVERSAL_CONFIG_PATH):

    global MODEL_ID, LOCAL_MODEL_PATH, CONFIG_FILE_PATH, MODEL_WEIGHTS_PATH, INPUT_ROOT_DIR, NUM_GPUS, TOTAL_WORKERS, BATCH_SIZE, INPUT_ROOT_DIR, INPUT_ROOT_DIR

    with open("/app/notebooks/pep_dev_ready/config.json") as f:
        config = json.load(f)
    
    MODEL_ID = config["model_id"]
    LOCAL_MODEL_PATH = config["local_model_path"]
    CONFIG_FILE_PATH = os.path.join(LOCAL_MODEL_PATH, "config.json")
    MODEL_WEIGHTS_PATH = config["model_weights_path"]
    NUM_GPUS = int(config["num_gpus"])
    TOTAL_WORKERS = NUM_GPUS
    BATCH_SIZE = config["batch_size"]
    
    INPUT_ROOT_DIR = config["catalog_metadata_processed"]
    OUTPUT_ROOT_DIR = config["catalog_token_embeddings_path"]

    print("Universal Config Loaded Successfully!!")

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
    
    # 2. Load Custom Weights
    if os.path.exists(MODEL_WEIGHTS_PATH):
        try:
            state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
            # Load state dict, ignoring non-matching keys if strict=False is necessary
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Rank {dist.get_rank()}: Custom weights loaded from {MODEL_WEIGHTS_PATH}.")
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

# 🔑 MODIFICATION: create_category_stop_set now expects the category column to be present.
# It will be manually added in ddp_worker before this function is called.
def create_category_stop_set(df: pd.DataFrame, lemmatizer_obj: SpacyLemmatizer) -> Set[str]:
    """
    Creates a set of lemmatized category stop words using the SpaCy model.
    Assumes 'category' column has been added to the DataFrame in ddp_worker.
    """
    nlp_obj = lemmatizer_obj.nlp
    
    # This line previously failed because 'category' was missing. 
    # It now relies on ddp_worker ensuring the column exists.
    category_texts = df['category'].astype(str).str.lower().tolist() 
    all_category_words = list(itertools.chain.from_iterable(text.split() for text in category_texts))
    
    lemmatized_category_words = [
        token.lemma_ 
        for doc in nlp_obj.pipe(all_category_words, batch_size=500) 
        for token in doc if token.is_alpha
    ]
    
    custom_words_to_add = ['man','woman','child','kid','boy','girl','baby']
    
    lemmatized_custom_words = [
        token.lemma_ 
        for doc in nlp_obj.pipe(custom_words_to_add, batch_size=len(custom_words_to_add)) 
        for token in doc if token.is_alpha
    ]
    
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

# =================================================================
# 3. Processing Functions (MODIFIED for model.text_model)
# =================================================================

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

# =================================================================
# 4. THE DDP Worker/Rank Function (MODIFIED)
# =================================================================

def ddp_worker(rank, world_size, all_files_to_process, target_category):
    """
    Function executed by each DDP process (rank) for a specific category.
    Initializes DDP, loads model, and processes its assigned file chunk.
    """
    # 0. DDP Initialization (UNCHANGED)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=f'env://')
    gpu_id = rank
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    # 1. Model and Lemmatizer Setup (UNCHANGED)
    lemmatizer_obj = SpacyLemmatizer() 

    try:
        model, tokenizer = load_model_offline(LOCAL_MODEL_PATH, device)
        if rank == 0:
            print(f"Rank {rank}: Model loaded OFFLINE on {device}.")
    except Exception as e:
        print(f"Rank {rank} ERROR loading model: {e}")
        dist.destroy_process_group()
        return

    # 2. Data Distribution Logic (File Chunking) (UNCHANGED)
    num_files = len(all_files_to_process)
    my_files = all_files_to_process[rank:num_files:world_size]
    
    if rank == 0:
        print(f"Total files for '{target_category}': {num_files}. Files assigned to Rank {rank}: {len(my_files)}")
    
    # 3. Process files
    processed_count = 0
    
    for input_file_path in my_files:
        
        # --- File Processing ---
        try:
            df = pd.read_parquet(input_file_path)
            
            # 🔑 CRITICAL MODIFICATION: Manually add the 'category' column
            # The category value comes from the directory name passed as target_category
            if 'category' not in df.columns:
                df['category'] = target_category
            # Note: We keep the original error handling in case other files *do* have the column,
            # but the explicit addition bypasses the failure if it's missing.

            # Use Spacy-based embedding text creation
            df = create_embedding_text(df, lemmatizer_obj)
            print("FINISHED create_embedding_text")
            print(df.head())

            
            texts_to_embed = df['embedding_text'].tolist()
            print(f"{len(texts_to_embed)} texts to embed")
            
            # Pass the base model and device
            embeddings = generate_embeddings(texts_to_embed, model, tokenizer, device, rank, gpu_id, input_file_path)  

            df['text_embedding'] = list(embeddings)

            print("FINISHED generate_embeddings")
            print(df.head())

            
            # Save results to the category-partitioned folder
            filename = os.path.basename(input_file_path)
            
            # Output directory is partitioned by category (target_category already stripped of prefix)
            category_output_dir = os.path.join(OUTPUT_ROOT_DIR, target_category)
            os.makedirs(category_output_dir, exist_ok=True) 
            
            output_file_path = os.path.join(category_output_dir, filename)
            
            df.to_parquet(output_file_path, index=False)

            # Cleanup
            del df, texts_to_embed, embeddings
            gc.collect()
            torch.cuda.empty_cache()
            
            processed_count += 1

        except Exception as e:
            if rank == 0:
                print(f"\n⚠️ Rank {rank} FAILED to process file {os.path.basename(input_file_path)}: {e}")
            continue

    # 4. Final Cleanup and Synchronization (UNCHANGED)
    del model, tokenizer
    dist.barrier()
    dist.destroy_process_group()
    
    if rank == 0:
        print(f"\n✅ Rank {rank} finished. Processed {processed_count} files for {target_category}.")


# =================================================================
# 5. Main Execution Loop (MODIFIED)
# =================================================================

def main():
    load_universal_config()
    # --- MODEL DOWNLOAD LOGIC (UNCHANGED) ---
    if os.path.exists(CONFIG_FILE_PATH):
        print(f"✅ Model already found at: {LOCAL_MODEL_PATH}. Skipping download.")
    else:
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        print(f"⬇️ Local model not found. Downloading model and tokenizer for '{MODEL_ID}'...")
        try:
            # Internet Required on FIRST RUN: Download and save
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            tokenizer.save_pretrained(LOCAL_MODEL_PATH)
            model = AutoModel.from_pretrained(MODEL_ID)
            model.save_pretrained(LOCAL_MODEL_PATH)
            print(f"✅ Download complete. Files saved to: {LOCAL_MODEL_PATH}")
            del model, tokenizer; gc.collect()
        except Exception as e:
            print(f"❌ An error occurred during download: {e}")
            return
    # --- END MODEL DOWNLOAD LOGIC ---

    # 1. Ensure spawn start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    # Set necessary environment variables for DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' 

    # 2. Find all categories/subdirectories
    # 🔑 MODIFICATION 1: List only directories AND exclude special files like _SUCCESS.
    all_category_dirs = [
        d for d in os.listdir(INPUT_ROOT_DIR) 
        if os.path.isdir(os.path.join(INPUT_ROOT_DIR, d)) and not d.startswith('_')
    ]

    if not all_category_dirs:
        print(f"❌ No category subdirectories found in {INPUT_ROOT_DIR}. Exiting.")
        return
        
    print(f"Found {len(all_category_dirs)} category directories to process: {all_category_dirs}")

    for category_dir_name in all_category_dirs:
        
        # 🔑 MODIFICATION 2: Strip the 'category=' prefix for the output folder name
        if category_dir_name.startswith('category='):
            target_category = category_dir_name[len('category='):].strip()
        else:
            target_category = category_dir_name # Use as is if prefix is missing (safety)


        # category_dir_name is the full path name (e.g., 'category=Men - Tshirts')
        # target_category is the clean name (e.g., 'Men - Tshirts')
        
        category_input_dir = os.path.join(INPUT_ROOT_DIR, category_dir_name)
        category_output_dir = os.path.join(OUTPUT_ROOT_DIR, target_category) # Use clean name for output
        os.makedirs(category_output_dir, exist_ok=True)
        
        print(f"\n=======================================================")
        print(f"🚀 Processing Target Category: {target_category}")
        print(f"=======================================================")

        # 3. Setup directories and FILTER files for the current category
        all_input_files = glob.glob(os.path.join(category_input_dir, "*.parquet"))
        
        # Filter logic using category-specific output directory
        processed_files = set(os.path.basename(f) for f in glob.glob(os.path.join(category_output_dir, "*.parquet")))
        files_to_process = [
            f for f in all_input_files 
            if os.path.basename(f) not in processed_files
        ]
        
        print(f"Total input files for {target_category}: {len(all_input_files)}")
        print(f"Processed files found in output dir: {len(processed_files)}")
        print(f"Processing {len(files_to_process)} remaining files using DDP on {NUM_GPUS} GPUs.")

        if not files_to_process:
            print(f"✅ All files for category '{target_category}' have been processed. Skipping DDP spawn.")
            continue

        # 4. Launch DDP processes
        world_size = NUM_GPUS
        
        try:
            mp.spawn(
                ddp_worker,
                args=(world_size, files_to_process, target_category), # Pass file list and clean category name
                nprocs=world_size,
                join=True
            )
        except Exception as e:
            print(f"🔥 DDP Spawn failed for {target_category}: {e}")
            
    print("\n✅ All categories processed. Results saved in:", OUTPUT_ROOT_DIR)


if __name__ == "__main__":
    main()