# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "ecommerce_rag_chatbot"
MODEL_NAME = "models/gemini-embedding-001"

import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "")
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID", "")

