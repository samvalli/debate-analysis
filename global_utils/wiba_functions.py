import requests
import pandas as pd
import json
# Base URL of the API
BASE_URL = 'http://wiba.dev/api'


#Each function require as input a df file with a column 'texts which represent the items to be analyzed

def wiba_extract(texts):
    """
    Extract information from given texts using the API's extract endpoint.
    
    Args:
        texts (list): List of text strings to analyze.
    
    Returns:
        pandas.DataFrame: Results of the extraction.
    """
    url = f"{BASE_URL}/extract"
    payload = {"texts": texts}
    response = requests.post(url, json=payload)
    return pd.DataFrame(response.json())

def wiba_detect(df):
    """
    Detect information in texts from a DataFrame using the API's detect endpoint.
    
    Args:
        df (pandas.DataFrame): DataFrame containing texts to analyze.
    
    Returns:
        pandas.DataFrame: Results of the detection.
    """
    url = f"{BASE_URL}/detect"
    csv_data = df.to_csv(index=False)
    payload = {"texts": csv_data}
    response = requests.post(url, json=payload)
    return pd.DataFrame(response.json())

def wiba_stance(texts, topics):
    """
    Analyze the stance of texts on given topics using the API's stance endpoint.
    
    Args:
        texts (list): List of text strings to analyze.
        topics (list): List of topics to analyze stance on.
    
    Returns:
        pandas.DataFrame: Results of the stance analysis.
    """
    url = f"{BASE_URL}/stance"
    payload = {"texts": texts, "topics": topics}

    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return pd.DataFrame(result)  # Convert to DataFrame
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def create_segments(input_file, column_name, window_size=3):
    """
    Create text segments using the API's create_segments endpoint.
    
    Args:
        input_file (str): CSV file content as a string.
        column_name (str): Name of the column containing text to segment.
        window_size (int, optional): Size of the sliding window. Defaults to 3.
    
    Returns:
        pandas.DataFrame: Created segments.
    """
    url = f"{BASE_URL}/create_segments"
    payload = {
        "data": input_file,
        "column_name": column_name,
        "window_size": window_size,
        "step_size": 1
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result_csv = response.json()
        segments_df = pd.DataFrame(result_csv)
        print("Created segments:")
        print(segments_df)
        return segments_df
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def calculate_segments(input_file, column_name, confidence_column):
    """
    Calculate segment scores using the API's calculate_segments endpoint.
    
    Args:
        input_file (str): CSV file content as a string.
        column_name (str): Name of the column containing segments.
        confidence_column (str): Name of the column containing confidence scores.
    
    Returns:
        pandas.DataFrame: Calculated segment scores.
    """
    url = f"{BASE_URL}/calculate_segments"
    payload = {
        "data": input_file,
        "segment_column": column_name,
        "argument_score_column": confidence_column
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result_csv = response.json()
        segments_df = pd.DataFrame(result_csv)
        print("Calculated segments:")
        return segments_df
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None