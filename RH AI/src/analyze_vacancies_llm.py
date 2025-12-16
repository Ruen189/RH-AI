# src/analyze_vacancies_llm.py
import json
from typing import List, Dict, Any
from tqdm import tqdm
from llm_client import get_llama
from llm_prompts import VACANCY_COMPETENCIES_PROMPT
from log_utils import log_raw_response
from llm_utils import parse_competencies
BATCH_SIZE = 6      # можно увеличить/уменьшить в зависимости от VRAM
MAX_NEW_TOKENS = 128
import os

def load_vacancies(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_prompt(vac: Dict[str, Any]) -> str:
    industry = vac.get("industry")
    title = vac.get("title")
    description = vac.get("description") or ""
    skills_extracted = vac.get("skills_extracted") or []

    prompt = VACANCY_COMPETENCIES_PROMPT.format(
        industry=industry,
        title=title,
        description=description[:4000],  # safety-ограничение
        skills_extracted=skills_extracted,
    )
    return prompt

def analyze_vacancies(vacancies_path: str, out_path: str):
    ADAPTER_DIR = r"QLoRA/vac_qlora_adapter/checkpoint-200"  # путь до QLoRA
    print("adapter_dir:", ADAPTER_DIR)
    print("exists:", os.path.exists(ADAPTER_DIR))
    print("adapter_config exists:", os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")))
    llama = get_llama(adapter_dir=ADAPTER_DIR)
    vacancies = load_vacancies(vacancies_path)

    results = []

    batch_prompts: List[str] = []
    batch_meta: List[Dict[str, Any]] = []

    def process_batch():
        nonlocal results, batch_prompts, batch_meta
        if not batch_prompts:
            return

        # LLM-ответы пачкой
        raw_answers = llama.ask_batch(
            batch_prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=BATCH_SIZE,
        )

        actual_logs_written = 0

        MAX_LOGS = 2  # максимум сырых ответа за запуск

        for meta, raw in zip(batch_meta, raw_answers):
            # Логируем первые 3 ответа
            if (actual_logs_written < MAX_LOGS):
                log_raw_response("vacancy", meta["vacancy_id"], raw)
                actual_logs_written += 1

            comps = parse_competencies(raw)

            # если модель ничего не выдала → ставим "-"
            if not comps:
                comps = ["-"]
            else:
                # на всякий случай чистим None / пустые строки
                comps = [c if (isinstance(c, str) and c.strip()) else "-" for c in comps]
                # убираем дубли и лишние дефисы
                comps = list(dict.fromkeys(c.strip() for c in comps if c and c != "-")) or ["-"]

            results.append({
                "vacancy_id": meta["vacancy_id"],
                "industry": meta["industry"],
                "title": meta["title"],
                "competencies": comps,
            })

        # очищаем буфер
        batch_prompts = []
        batch_meta = []

    for vac in tqdm(vacancies, desc="LLM: vacancies (batched)"):
        prompt = _build_prompt(vac)

        batch_prompts.append(prompt)
        batch_meta.append({
            "vacancy_id": vac["id"],
            "industry": vac.get("industry"),
            "title": vac.get("title"),
        })

        # как только набрали пачку — отправляем в модель
        if len(batch_prompts) >= BATCH_SIZE:
            process_batch()

    # не забываем хвост
    process_batch()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Индустриальные компетенции по вакансиям сохранены в {out_path}")


if __name__ == "__main__":
    analyze_vacancies(
        "data/processed/vacancies_processed.json",
        "data/derived/industry_competencies_llm.json"
    )