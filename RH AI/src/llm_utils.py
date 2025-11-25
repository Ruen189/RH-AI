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
