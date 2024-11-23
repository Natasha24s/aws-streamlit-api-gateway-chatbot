import requests
from config import API_ENDPOINT

def send_message(message):
    """
    Send a message to the AWS API Gateway endpoint.
    """
    try:
        response = requests.post(API_ENDPOINT, json={"message": message})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None
