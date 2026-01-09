# src/extract_skills.py
import json
import os
import re
from typing import List, Dict

# --- исправленный путь ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "skill_patterns.json")

with open(MODEL_PATH, "r", encoding="utf-8") as f:
    PATTERNS = json.load(f)

# простая лемматизация-замена кир/лат
REPLACERS = {
    "питон": "python",
    "джанго": "django",
}

def normalize_token(t: str) -> str:
    t = t.lower().strip()
    return REPLACERS.get(t, t)

def extract_by_patterns(text: str) -> Dict[str, List[str]]:
    text_low = text.lower()
    found = {k: [] for k in PATTERNS.keys()}
    for group, words in PATTERNS.items():
        for w in words:
            if w in text_low:
                found[group].append(w)
    # удаляем пустые
    return {g: list(set(ws)) for g, ws in found.items() if ws}

def extract_frequent_terms(text: str, top_n: int = 10) -> List[str]:
    # очень грубый сплит
    tokens = re.findall(r"[A-Za-zА-Яа-я0-9\-\+#\.]+", text.lower())
    counts = {}
    for t in tokens:
        t = normalize_token(t)
        if len(t) < 2:
            continue
        counts[t] = counts.get(t, 0) + 1
    # топ по частоте
    return [w for w, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]]

def extract_skills_from_vacancy(vac: dict) -> dict:
    full_text = " ".join([
        vac.get("title") or "",
        vac.get("description_text") or "",
        vac.get("requirements_raw") or "",
    ])
    by_patterns = extract_by_patterns(full_text)
    frequent = extract_frequent_terms(full_text, top_n=15)

    # собрать в единый список
    skills = set(frequent)
    for group, ws in by_patterns.items():
        for w in ws:
            skills.add(w)

    vac["skills_extracted"] = list(skills)
    vac["skill_groups"] = by_patterns
    return vac
