# src/main_collect.py
import json
import glob
import os
from normalise import normalize_hh, normalize_sj
from extract_skills import extract_skills_from_vacancy
from config import RAW_DIR, PROCESSED_DIR

def load_raw_files(prefix: str):
    print("\n[STEP] Загрузка файлов:", prefix)
    print("[DEBUG] RAW_DIR =", RAW_DIR)

    patterns = [
        os.path.join(RAW_DIR, f"{prefix}_*.ndjson"),
    ]

    print("[DEBUG] Patterns:", patterns)

    files = []
    for p in patterns:
        found = glob.glob(p)
        print(f"[DEBUG] Проверяю {p} → найдено {len(found)} файлов")
        files.extend(found)

    if not files:
        print(f"[WARN] Не найдено файлов с префиксом {prefix} в {RAW_DIR}")
        return []

    print(f"[INFO] Найдено файлов: {files}")

    data = []
    for f in files:
        print(f"[INFO] Загружаю {f} ...")
        try:
            if f.endswith(".ndjson"):
                with open(f, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        data.append(json.loads(line))
            else:
                with open(f, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                    if isinstance(obj, dict):
                        if "items" in obj: obj = obj["items"]
                        if "objects" in obj: obj = obj["objects"]
                    if isinstance(obj, list):
                        data.extend(obj)
        except Exception as e:
            print(f"[ERROR] Ошибка чтения {f}: {e}")

    print(f"[INFO] Загружено {len(data)} записей ({prefix})\n")
    return data



def process_all():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed = []

    # === HeadHunter ===
    hh_raw = load_raw_files("hh")
    print(f"[INFO] Обработка {len(hh_raw)} вакансий HH ...")
    for idx, item in enumerate(hh_raw, start=1):
        try:
            vac = normalize_hh(item)
            vac = extract_skills_from_vacancy(vac)
            processed.append(vac)
            if idx % 10 == 0 or idx == len(hh_raw):
                print(f"  → HH: обработано {idx}/{len(hh_raw)} вакансий")
        except Exception as e:
            print(f"[ERROR] HH ID={item.get('id')} ошибка: {e}")

    # === SuperJob ===
    sj_raw = load_raw_files("sj")
    print(f"[INFO] Обработка {len(sj_raw)} вакансий SJ ...")
    for idx, item in enumerate(sj_raw, start=1):
        try:
            vac = normalize_sj(item)
            vac = extract_skills_from_vacancy(vac)
            processed.append(vac)
            if idx % 10 == 0 or idx == len(sj_raw):
                print(f"  → SJ: обработано {idx}/{len(sj_raw)} вакансий")
        except Exception as e:
            print(f"[ERROR] SJ ID={item.get('id')} ошибка: {e}")

    # === Итог ===
    print(f"[INFO] Всего обработано {len(processed)} вакансий.")
    if not processed:
        print("[WARN] Нет данных для сохранения! Проверь сырые файлы в data/raw/.")
        return

    out_path = os.path.join(PROCESSED_DIR, "vacancies_processed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"[OK] Файл сохранён: {out_path} ({len(processed)} записей)")

if __name__ == "__main__":
    process_all()
