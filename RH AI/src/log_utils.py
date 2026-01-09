# src/log_utils.py
import os

LOG_PATH = "data/log.txt"

def log_raw_response(prefix: str, item_id: str, raw: str):
    """
    Логирует сырые ответы LLM в data/log.txt.
    prefix — тип данных (например "vacancy" или "project").
    item_id — ID элемента.
    raw — ответ от модели.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n===== RAW LLM RESPONSE ({prefix}: {item_id}) =====\n")
        f.write(raw)
        f.write("\n==============================================\n\n")
