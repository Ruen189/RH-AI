# src/fetch_hh.py
import os
import time
import json
import requests
from datetime import datetime
from typing import List, Dict, Optional

HH_BASE_URL = "https://api.hh.ru"
RAW_DIR = "data/raw"

HEADERS = {"User-Agent": "RH-AI-Memory-Agent/1.0"}

def _safe(fn, *args, retries=3, pause=1.0, **kwargs):
    for i in range(retries):
        try:
            return fn(*args, timeout=20, **kwargs)
        except Exception as e:
            print(f"[WARN] попытка {i+1}/{retries} => {e}")
            time.sleep(pause)
    raise

def fetch_hh_vacancies(query: str, area: Optional[int]=None, pages: int=5) -> List[Dict]:
    all_items = []
    for page in range(pages):
        params = {"text": query, "page": page, "per_page": 100, "search_field": "name"}
        if area:
            params["area"] = area

        resp = _safe(requests.get, f"{HH_BASE_URL}/vacancies", params=params, headers=HEADERS)
        if resp.status_code == 429:
            print("[RATE] 429 Too Many Requests, спим 2 сек")
            time.sleep(2)
            continue
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        print(f"[HH] '{query}' страница {page+1}/{data.get('pages', '?')} — получено {len(items)}")

        all_items.extend(items)
        time.sleep(0.4)
        if page >= data.get("pages", 1) - 1:
            break
    return all_items

def fetch_hh_vacancy_details(vac_id: str) -> Dict:
    resp = _safe(requests.get, f"{HH_BASE_URL}/vacancies/{vac_id}", headers=HEADERS)
    resp.raise_for_status()
    return resp.json()

def collect_hh_batch(queries: List[str], area: Optional[int]=None, pages:int=5):
    os.makedirs(RAW_DIR, exist_ok=True)
    date_tag = datetime.utcnow().strftime("%Y-%m-%d")
    ndjson_path = os.path.join(RAW_DIR, f"hh_{date_tag}.ndjson")

    total = 0
    with open(ndjson_path, "a", encoding="utf-8") as out:
        for q in queries:
            items = fetch_hh_vacancies(q, area=area, pages=pages)
            print(f"[HH] Найдено {len(items)} коротких карточек по запросу '{q}'")
            for idx, it in enumerate(items, start=1):
                vac_id = it["id"]
                full = fetch_hh_vacancy_details(vac_id)
                full["_source"] = "hh"
                full["_fetched_at"] = datetime.utcnow().isoformat()
                # ЛОГ поштучно:
                title = full.get("name")
                print(f"  → HH {q}: {idx}/{len(items)} id={vac_id} | {title}")
                out.write(json.dumps(full, ensure_ascii=False) + "\n")
                total += 1
                time.sleep(0.3)

    print(f"[OK][HH] Сохранено {total} вакансий в {ndjson_path}")
    return ndjson_path

if __name__ == "__main__":
    DEFAULT_PROF_AREAS = ["python", "data scientist", "ml engineer", "backend", "education", "edtech"]
    collect_hh_batch(DEFAULT_PROF_AREAS, area=None, pages=3)
