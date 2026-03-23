import pandas as pd
import json
import os

def process_trend_jsons(input_directory: str) -> pd.DataFrame:
    """
    Reads JSON files from a directory, processes trend data, and returns a DataFrame.
    
    Parameters
    ----------
    input_directory : str
        Path to the directory containing JSON files.
    output_file : str, optional
        If provided, saves the resulting DataFrame to this CSV file.
    
    Returns
    -------
    pd.DataFrame
        Processed trends data.
    """
    
    all_trends = []

    # Iterate through directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            category_name = os.path.splitext(filename)[0]
            file_path = os.path.join(input_directory, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Process each market key
                for market_key in data.keys():
                    market_type = 'India' if 'india' in market_key.lower() else 'Global'

                    for trend in data[market_key]:
                        trend['category'] = category_name
                        trend['market'] = market_type
                        all_trends.append(trend)

    # Convert to DataFrame
    if not all_trends:
        print("No JSON data found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_trends)

    # Reorder columns
    cols = ['category', 'market'] + [c for c in df.columns if c not in ['category', 'market']]
    df = df[cols]

    return df


if __name__=="__main__":
    with open("/app/notebooks/pep_dev_ready/config.json") as f:
        config = json.load(f)

    raw_trends_dir = config["raw_trends_path"]

    if not os.path.exists(raw_trends_dir):
        print("Trends path does not exist!!!")
        exit(1)

    processed_trends = process_trend_jsons(raw_trends_dir)

    processed_trends_path = config["processed_trends_path"]
    output_dir = os.path.dirname(processed_trends_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    processed_trends.to_csv(processed_trends_path, header=True, index=False)
    
    
    


    

    

