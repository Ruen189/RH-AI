import json
from typing import List, Dict, Any
from tqdm import tqdm
import os

from llm_client import get_llama
from llm_prompts import VACANCY_COMPETENCIES_PROMPT
from llm_utils import parse_competencies

MAX_NEW_TOKENS = 128


def load_vacancies(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_prompt(vac: Dict[str, Any]) -> str:
    industry = vac.get("industry")
    title = vac.get("title")
    description = vac.get("description") or ""
    skills_extracted = vac.get("skills_extracted") or []

    return VACANCY_COMPETENCIES_PROMPT.format(
        industry=industry,
        title=title,
        description=description[:4000],
        skills_extracted=skills_extracted,
    )


def analyze_vacancies(vacancies_path: str, out_path: str):
    ADAPTER_DIR = r"QLoRA/vac_qlora_adapter/checkpoint-200"
    print("adapter_dir:", ADAPTER_DIR)
    print("exists:", os.path.exists(ADAPTER_DIR))
    print("adapter_config exists:", os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")))

    llama = get_llama(ADAPTER_DIR)
    vacancies = load_vacancies(vacancies_path)

    prompts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for vac in tqdm(vacancies, desc="Build prompts: vacancies"):
        prompts.append(_build_prompt(vac))
        metas.append({
            "vacancy_id": vac["id"],
            "industry": vac.get("industry"),
            "title": vac.get("title"),
        })

    raw_answers = llama.generate(
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=1.0,
        use_tqdm=True,
    )

    results: List[Dict[str, Any]] = []
    for meta, raw in zip(metas, raw_answers):
        comps = parse_competencies(raw)

        if not comps:
            comps = ["-"]
        else:
            comps = [c if (isinstance(c, str) and c.strip()) else "-" for c in comps]
            comps = list(dict.fromkeys(c.strip() for c in comps if c and c != "-")) or ["-"]

        results.append({
            "vacancy_id": meta["vacancy_id"],
            "industry": meta["industry"],
            "title": meta["title"],
            "competencies": comps,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Индустриальные компетенции по вакансиям сохранены в {out_path}")


if __name__ == "__main__":
    analyze_vacancies(
        "data/processed/vacancies_processed.json",
        "data/derived/industry_competencies_llm.json"
    )
