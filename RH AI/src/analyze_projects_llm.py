# src/analyze_projects_llm.py
import json
from typing import List, Dict, Any
from tqdm import tqdm

from llm_client import get_llama
from llm_prompts import PROJECT_COMPETENCIES_PROMPT
from llm_utils import parse_competencies
from log_utils import log_raw_response

BATCH_SIZE = 6
MAX_NEW_TOKENS = 128
MAX_LOGS = 2   # логируем только первые ответы


def load_projects(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(project: Dict[str, Any]) -> str:
    industry = project.get("industry", "")
    title = project.get("title", "")
    description = project.get("description", "")
    goal = project.get("goal", "") or "Цель не указана"
    results = project.get("results", "") or "Результаты не указаны"
    tech = project.get("tech", "") or "Технологии не указаны"

    return PROJECT_COMPETENCIES_PROMPT.format(
        industry=industry,
        title=title,
        description=description[:6000],
        goal=goal,
        results=results,
        tech=tech,
    )


def analyze_projects(projects_path: str, out_path: str):
    llama = get_llama()
    projects = load_projects(projects_path)

    results: List[Dict[str, Any]] = []
    batch_prompts: List[str] = []
    batch_meta: List[Dict[str, Any]] = []
    logs_written = 0

    def process_batch():
        nonlocal batch_prompts, batch_meta, results, logs_written
        if not batch_prompts:
            return

        raw_answers = llama.ask_batch(
            batch_prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=BATCH_SIZE,
        )

        for meta, raw in zip(batch_meta, raw_answers):
            # логируем первые несколько сырых ответов
            if logs_written < MAX_LOGS:
                log_raw_response("project", str(meta.get("project_id")), raw)
                logs_written += 1

            comps = parse_competencies(raw)

            if not comps:
                comps = ["-"]
            else:
                comps = [c if (isinstance(c, str) and c.strip()) else "-" for c in comps]
                comps = list(dict.fromkeys(c.strip() for c in comps if c and c != "-")) or ["-"]

            results.append({
                "project_id": meta.get("project_id"),
                "industry": meta.get("industry"),
                "title": meta.get("title"),
                "competencies": comps,
            })

        batch_prompts = []
        batch_meta = []

    for p in tqdm(projects, desc="LLM: projects (batched)"):
        prompt = build_prompt(p)
        batch_prompts.append(prompt)
        batch_meta.append({
            "project_id": p.get("id"),
            "industry": p.get("industry"),
            "title": p.get("title"),
        })

        if len(batch_prompts) >= BATCH_SIZE:
            process_batch()

    process_batch()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Компетенции проектов сохранены в {out_path}")


if __name__ == "__main__":
    analyze_projects(
        "data/raw/projects_with_industries_full.json",   # или processed, если ты туда переложил
        "data/derived/project_competencies_llm.json",
    )
