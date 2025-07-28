import requests
import json

# The URL of your FastAPI endpoint
api_url = "http://127.0.0.1:8000/predict_sum"

def summarize_service(text:str):
    payload = {"text": text}

    try:
        # Make the POST request
        response = requests.post(api_url, json=payload)

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Get the classification result from the JSON response
        result = response.json()
        return result

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
