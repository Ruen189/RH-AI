# src/normalise_vacancy_competencies.py
import json
from pathlib import Path
import re

INPUT_PATH = Path("data/derived/industry_competencies_llm.json")
OUTPUT_PATH = Path("data/derived/industry_competencies_llm_clean.json")


# Базовая нормализация (та же, что в проектной версии)
NORMALIZATION_MAP = {
    "python": "Python",
    "java": "Java",
    "c#": "C#",
    "c++": "C++",
    "c": "C",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "node.js": "Node.js",
    "react": "React",
    "vue": "Vue",
    "angular": "Angular",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "sql": "SQL",
    "postgresql": "PostgreSQL",
    "postgress": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Azure",
    "llm": "LLM",
    "ml": "ML",
    "nlp": "NLP",
    "eda": "EDA",
    "datascience": "Data Science",
    "unity": "Unity",
    "unreal engine": "Unreal Engine",
    "webgl": "WebGL",
    "3d": "3D",
    "1c": "1C",
    "1с": "1C",  # русская буква С
}


def normalize_token(token: str) -> str:
    t = token.strip()
    if not t:
        return ""

    # убираем кавычки
    t = t.strip("\"' ")

    low = t.lower()
    if low in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[low]

    # аббревиатура
    if t.isupper() and len(t) <= 6:
        return t

    # по умолчанию — капитализация
    return t[0].upper() + t[1:]


def split_competency_string(s: str):
    """
    Разбиваем строки типа:
    "Python, SQL, Django"
    "Unity/Unreal Engine"
    "C#, .NET, React, Node.js"
    """
    parts = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        # делим unity/unreal или похожие
        if "/" in chunk and len(chunk) < 40:
            sub = [c.strip() for c in chunk.split("/") if c.strip()]
            parts.extend(sub)
        else:
            parts.append(chunk)
    return parts


def clean_competencies_list(raw_list):
    cleaned = []
    seen = set()

    for item in raw_list:
        if not isinstance(item, str):
            continue

        tokens = split_competency_string(item)

        for tok in tokens:
            norm = normalize_token(tok)
            if not norm:
                continue
            if norm not in seen:
                seen.add(norm)
                cleaned.append(norm)

    return cleaned


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {INPUT_PATH}")

    content = INPUT_PATH.read_text(encoding="utf-8").strip()
    if not content:
        data = []
    else:
        data = json.loads(content)

    for obj in data:
        comps = obj.get("competencies", [])
        if not isinstance(comps, list) or not comps:
            obj["competencies"] = []
            continue

        obj["competencies"] = clean_competencies_list(comps)

    OUTPUT_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"[OK] Очищенные компетенции вакансий сохранены в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
