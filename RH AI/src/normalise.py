# src/normalize.py
import re
from bs4 import BeautifulSoup

def html_to_text(s: str) -> str:
    if not s:
        return ""
    return BeautifulSoup(s, "html.parser").get_text(" ", strip=True)

def normalize_hh(item: dict) -> dict:
    desc = item.get("description") or ""
    desc = BeautifulSoup(desc, "html.parser").get_text(" ", strip=True)

    return {
        "id": f"hh:{item['id']}",
        "source": "hh",
        "title": item.get("name"),
        "employer": (item.get("employer") or {}).get("name"),
        "area": (item.get("area") or {}).get("name"),
        "industry": item.get("_industry"),
        "description": desc,
        "skills_extracted": [],
        "skill_groups": {},
        "meta": {
            "api_loaded_at": item.get("_fetched_at")
        }
    }


def normalize_sj(item: dict) -> dict:
    text = item.get("candidat") or ""
    return {
        "id": f"sj:{item['id']}",
        "source": "sj",
        "title": item.get("profession"),
        "employer": (item.get("client") or {}).get("title"),
        "area": item.get("town", {}).get("title"),
        "industry": item.get("_industry"),
        "description": text,
        "skills_extracted": [],
        "skill_groups": {},
        "meta": {
            "api_loaded_at": item.get("_fetched_at")
        }
    }

