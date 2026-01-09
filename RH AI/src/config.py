import os
import json
from dotenv import load_dotenv

load_dotenv()

SJ_API_KEY = os.getenv("SJ_API_KEY", "YOUR_TOKEN_HERE")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

INDUSTRY_KEYWORDS_PATH = os.path.join(
    BASE_DIR, "src", "data", "industry_keywords.json"
)

def load_industry_keywords():
    with open(INDUSTRY_KEYWORDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
