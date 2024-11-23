import os
from dotenv import load_dotenv

load_dotenv()

API_ENDPOINT = os.getenv("API_ENDPOINT", "https://jqoatdawvh.execute-api.us-east-2.amazonaws.com/prod/chat")

