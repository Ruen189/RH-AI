# src/llm_utils.py
import json
import re
from typing import List


def parse_competencies(raw: str) -> List[str]:
    """
    Ожидаем, что модель вернёт JSON-массив строк.
    Если в тексте несколько массивов или лишние слова,
    берём ПЕРВЫЙ массив, который удаётся распарсить.
    """
    raw = raw.strip()

    # 1. Прямая попытка: вдруг это чистый JSON
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x).strip() for x in data if isinstance(x, str) and str(x).strip()]
    except Exception:
        pass

    # 2. Ищем все подстроки вида [ ... ]
    candidates = re.findall(r'\[[^\]]*\]', raw, re.S)
    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, list):
                return [str(x).strip() for x in data if isinstance(x, str) and str(x).strip()]
        except Exception:
            continue

    return []


def safe_parse_llm_json(raw: str):
    """
    Пытается достать JSON из ответа модели.
    Если не удаётся — возвращает {"summary": raw}.
    """

    raw = raw.strip()

    # 1 — если это чистый JSON
    try:
        return json.loads(raw)
    except:
        pass

    # 2 — ищем { ... } в тексте
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except:
            pass

    # 3 — ищем [ ... ] (редко нужно, но бывает)
    match = re.search(r"\[[\s\S]*\]", raw)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except:
            pass

    # 4 — ничего не вышло → возвращаем текст
    return {"summary": raw}
