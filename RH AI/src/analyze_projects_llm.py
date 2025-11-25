# src/analyze_projects_llm.py
import json
from tqdm import tqdm
from llm_client import get_llama
from llm_prompts import PROJECT_COMPETENCIES_PROMPT
from log_utils import log_raw_response
from llm_utils import parse_competencies

BATCH_SIZE = 8         # можно подобрать по VRAM
MAX_NEW_TOKENS = 128


def load_projects(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(project):
    return PROJECT_COMPETENCIES_PROMPT.format(
        title=project["title"],
        description=project.get("description", "")[:5000],
        industry=project.get("industry"),
        tech=project.get("tech") or "",
    )

def analyze_projects(projects_path, out_path):
    llama = get_llama()
    projects = load_projects(projects_path)

    batch_prompts = []
    batch_meta = []
    results = []

    def process_batch():
        nonlocal batch_prompts, batch_meta, results
        if not batch_prompts:
            return

        raw_answers = llama.ask_batch(
            batch_prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=BATCH_SIZE
        )
        MAX_LOGS = 3
        global_logs_written = 0
        for meta, raw in zip(batch_meta, raw_answers):
            if global_logs_written < MAX_LOGS:
                log_raw_response("project", meta["project_id"], raw)
                global_logs_written += 1

            comps = parse_competencies(raw)

            results.append({
                "project_id": meta["project_id"],
                "industry": meta["industry"],
                "title": meta["title"],
                "competencies": comps,
            })

        batch_prompts = []
        batch_meta = []

    for prj in tqdm(projects, desc="LLM: projects (batched)"):
        prompt = build_prompt(prj)

        batch_prompts.append(prompt)
        batch_meta.append({"project_id": prj["id"]})

        if len(batch_prompts) >= BATCH_SIZE:
            process_batch()

    process_batch()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] Компетенции проектов сохранены в {out_path}")


if __name__ == "__main__":
    analyze_projects(
        "data/projects_with_industries_full.json",
        "data/derived/project_competencies_llm.json"
    )
