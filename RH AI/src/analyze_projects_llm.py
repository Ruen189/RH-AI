import json
from typing import List, Dict, Any
from tqdm import tqdm

from llm_client import get_llama
from llm_prompts import PROJECT_COMPETENCIES_PROMPT
from llm_utils import parse_competencies

MAX_NEW_TOKENS = 128


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

    prompts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for p in tqdm(projects, desc="Build prompts: projects"):
        prompts.append(build_prompt(p))
        metas.append({
            "project_id": p.get("id"),
            "industry": p.get("industry"),
            "title": p.get("title"),
        })

    raw_answers = llama.generate(
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,   # deterministic
        top_p=1.0,
        use_tqdm=True,     # прогресс уже на стороне vLLM
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
            "project_id": meta.get("project_id"),
            "industry": meta.get("industry"),
            "title": meta.get("title"),
            "competencies": comps,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Компетенции проектов сохранены в {out_path}")


if __name__ == "__main__":
    analyze_projects(
        "data/projects_with_industries_full.json",
        "data/derived/project_competencies_llm.json",
    )
