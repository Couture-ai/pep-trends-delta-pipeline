import os
import json
import re
from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from openai_utils import AzureBatchHandler
os.environ["AZURE_OPENAI_API_KEY"]=""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""

BASE_PROMPT = '''
Role & Context
You are a Fashion Trend Analyst AI working for a large-scale fashion ecommerce marketplace.
Your goal is to identify emerging and trending fashion aesthetics from social media signals (Instagram, TikTok, Pinterest, creator content, user searches) and convert them into search-friendly, commerce-ready hashtags that can be used for product discovery, collections, filters, and social-to-PLP flows. generate this output for the category: {category}
 
Task
Generate a weekly list of Trend-Driven Aesthetic Hashtags for women’s fashion, optimized for ecommerce discovery.
 
Guidelines
Focus on aesthetic-led trends, not generic fashion tags
Each hashtag should:
Represent a distinct fashion “vibe” or aesthetic
Be searchable and human-readable
 
Work across social platforms and ecommerce discovery layers
 
Avoid:
Influencer-only tags (e.g., #instagirl)
Generic reach tags (e.g., #fashion, #style)
Non-visual attribute tags (like particular fabric or texture)
Generating similar trends (for ex. do not generate both <relaxed baggy core> and <effortless wide leg>)

Prefer hashtags that can map to category types, Silhouettes, Color palettes,fit / mood
 
Output Requirements
Return 10–15 trends for the week.
 
For each trend, include:
Trend Name (human-readable)
Primary Hashtag
Supporting Hashtags (3–5)
Aesthetic Description (1–2 lines) A brief description focusing on attributes of the products belonging in this trend that can help identify products using catalog attributes.
Confidence: a confidence score (0.0 to 5.0) that denotes how confident you are this trend is  relevant to this category and how popular it is.
Region: relevant region for the trend <India or Global, no other regions>,
Common Product Types (e.g., dresses, co-ords, tops)
Best Use in Ecommerce
(Example: PLP collection, Trend badge, Social discovery widget)


Incldue a mix of both indian and global trends for each category. 
ONLY generate trends for the specified category, do NOT generate trends not relevant directly to the category (for example, do not include "beach sandals" for men - sneakers or "soft festive blouse" for "women - dresses".
Generate UP T0 10-15 trends for the category total. Only Generate distinct, meaningful tags, it is not mandatory to generate a minimum number of tags.
The keys for the JSON returned should be lower-cased and joined by underscore (for example, trend_name, primary_hashtag etc.)
Return the results in a json of the following format:
{{
    "trending_tags_india": [{{list of tag jsons with specified fields}}],
    "trending_tags_global": [{{list of tag jsons with specified fields}}]
}}
'''
# --- 1. State Definition ---
class State(TypedDict):
    categories: List[str]
    output_dir: str

# --- 2. Logic Nodes ---

def clean_llm_json(text: str) -> str:
    """Removes markdown code blocks (```json ... ```) from the LLM response."""
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()

def process_categories_node(state: State):
    """Loop through categories, get single response, and save with exact category string."""
    handler = AzureBatchHandler(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    os.makedirs(state["output_dir"], exist_ok=True)
    
    for category in state["categories"]:
        print(f"Generating trends for: {category}")
        
        prompt = BASE_PROMPT.format(category=category)
        
        # Call the new generate_response method
        raw_response = handler.generate_response(prompt, model_name="gpt-5")
        
        if raw_response:
            cleaned_json = clean_llm_json(raw_response)
            
            try:
                # Validate JSON structure
                json_data = json.loads(cleaned_json)
                trends_json = json.loads(json_data["output"][1]["content"][0]["text"])
                # SAVE USING CATEGORY STRING AS IS
                # We use .json extension as requested
                file_name = f"{category}.json"
                save_path = os.path.join(state["output_dir"], file_name)
                
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(trends_json, f, indent=4)
                    
                print(f"✓ Saved: {save_path}")
            except json.JSONDecodeError:
                print(f"× Error: Invalid JSON returned for {category}")
        else:
            print(f"× Error: No response for {category}")

    return state

# --- 3. Construct Graph ---
workflow = StateGraph(State)
workflow.add_node("process_all", process_categories_node)
workflow.add_edge(START, "process_all")
workflow.add_edge("process_all", END)

app = workflow.compile()

def load_universal_config(config_path=UNIVERSAL_CONFIG_PATH):

        with open("/app/notebooks/pep_dev_ready/config.json") as f:
            config = json.load(f)
        
        category_list = config["category_list"]
        output_dir = config["raw_trends_path"]
        
        print("Universal Config Loaded Successfully!!")

        return path_trend_product_map, path_product_metadata, path_embedding_model, path_processed_tags, dir_output_path

category_list = ['Women - Kurtas',
 'Women - Dresses',
 'Women - Jeans & Jeggings',
 'Women - Tops',
 'Men - Tshirts',
 'Men - Shirts',
 'Men - Sneakers',
 'Women - Kurta Suit Sets',
 'Men - Jackets & Blazers',
 'Men - Sweatshirts & Jackets',
 'Men - Jackets & Coats',
 'Infants - Dresses & Frocks',
 'Girls - Dresses & Frocks',
 'Girls - 3 Piece-Set',
 'Girls - 2 Piece-Sets',
 'Boys - 2P-Kurta Sets']
# category_list = ['Women - Kurtas']
# category_list = [
#  'Men - Sneakers',
#  'Men - Jackets & Blazers',
#  'Men - Jackets & Coats',
#  'Infants - Dresses & Frocks',
#  'Girls - 3 Piece-Set']
inputs = {
        "categories":category_list,
        "output_dir": "./trends_v4_updated_description/"
    }

app.invoke(inputs)
