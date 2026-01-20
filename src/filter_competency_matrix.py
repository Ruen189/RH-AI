# src/filter_competency_matrix.py
import json
from pathlib import Path

MATRIX_INPUT_PATH = Path("data/derived/competency_matrix.json")
MATRIX_OUTPUT_PATH = Path("data/derived/competency_matrix_filtered.json")
WHITELIST_PATH = Path("data/whitelist_competencies.json")


def load_whitelist() -> set:
    if not WHITELIST_PATH.exists():
        raise FileNotFoundError(f"Не найден файл whitelist: {WHITELIST_PATH}")
    content = WHITELIST_PATH.read_text(encoding="utf-8").strip()
    if not content:
        return set()
    data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError("whitelist_competencies.json должен быть JSON-массивом строк")
    return {str(x).strip() for x in data if isinstance(x, str) and x.strip()}


def main():
    if not MATRIX_INPUT_PATH.exists():
        raise FileNotFoundError(f"Не найден входной файл матрицы: {MATRIX_INPUT_PATH}")

    whitelist = load_whitelist()
    print(f"[INFO] Загружено {len(whitelist)} компетенций из whitelist")

    matrix = json.loads(MATRIX_INPUT_PATH.read_text(encoding="utf-8"))

    filtered = []
    for row in matrix:
        comp = row.get("competency")
        if not isinstance(comp, str):
            continue
        if comp not in whitelist:
            continue
        filtered.append(row)

    MATRIX_OUTPUT_PATH.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"[OK] Отфильтрованная матрица сохранена в {MATRIX_OUTPUT_PATH}")
    print(f"Всего записей до: {len(matrix)}, после фильтра: {len(filtered)}")


if __name__ == "__main__":
    main()
