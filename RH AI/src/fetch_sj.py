# src/fetch_sj.py
import os
import time
import json
import requests
from datetime import datetime
from typing import List, Dict

from config import RAW_DIR, SJ_API_KEY, load_industry_keywords

SJ_BASE_URL = "https://api.superjob.ru/2.0"

HEADERS = {
    "X-Api-App-Id": SJ_API_KEY if SJ_API_KEY else "",
    "User-Agent": "RH-AI-Memory-Agent/1.0",
}

def _ensure_key():
    if not SJ_API_KEY:
        raise RuntimeError("Не установлена переменная окружения SJ_API_KEY")

def fetch_sj_vacancies(keyword: str, pages: int = 5) -> List[Dict]:
    _ensure_key()
    all_items = []
    page = 0
    while page < pages:
        params = {"keyword": keyword, "page": page, "count": 100}
        resp = requests.get(f"{SJ_BASE_URL}/vacancies/", params=params, headers=HEADERS, timeout=20)
        if resp.status_code == 429:
            print("[RATE] SJ 429 Too Many Requests — пауза 2 сек")
            time.sleep(2)
            continue
        resp.raise_for_status()
        data = resp.json()
        objs = data.get("objects", [])
        print(f"[SJ] '{keyword}' стр. {page+1} — {len(objs)} вакансий")
        all_items.extend(objs)
        page += 1
        if not data.get("more"):
            break
        time.sleep(0.5)
    return all_items

def collect_sj_batch(pages: int = 5):
    _ensure_key()
    os.makedirs(RAW_DIR, exist_ok=True)
    date_tag = datetime.utcnow().strftime("%Y-%m-%d")
    ndjson_path = os.path.join(RAW_DIR, f"sj_{date_tag}.ndjson")

    industries = load_industry_keywords()
    total_written = 0

    with open(ndjson_path, "a", encoding="utf-8") as out:
        for ind in industries:
            industry_name = ind["industry"]
            keywords = ind["keywords"]

            # множество просмотренных id в рамках ОДНОЙ индустрии
            seen_ids = set()

            print(f"\n[INDUSTRY] {industry_name} — {len(keywords)} keywords")

            for kw in keywords:
                items = fetch_sj_vacancies(kw, pages=pages)
                print(f"[SJ] {industry_name} / '{kw}' → {len(items)} вакансий (до фильтрации дублей)")

                for idx, it in enumerate(items, start=1):
                    vac_id = str(it.get("id"))

                    # проверка на дубликат в рамках этой индустрии
                    if vac_id in seen_ids:
                        # можно залогировать, чтобы было видно:
                        # print(f"  → SKIP (dup in {industry_name}) id={vac_id}")
                        continue

                    seen_ids.add(vac_id)

                    it["_source"] = "sj"
                    it["_fetched_at"] = datetime.utcnow().isoformat()
                    it["_industry"] = industry_name   # ключевое поле для нормализации
                    title = it.get("profession")

                    print(f"  → SJ {industry_name}: {idx}/{len(items)} id={vac_id} | {title}")
                    out.write(json.dumps(it, ensure_ascii=False) + "\n")
                    total_written += 1
                    time.sleep(0.2)

            print(f"[INDUSTRY] {industry_name} — уникальных вакансий: {len(seen_ids)}")

    print(f"\n[OK][SJ] Сохранено {total_written} уникальных вакансий (по всем индустриям) в {ndjson_path}")
    return ndjson_path

if __name__ == "__main__":
    collect_sj_batch(pages=3)
