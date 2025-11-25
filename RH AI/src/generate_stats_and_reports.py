# src/generate_stats_and_reports.py
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import os
from llm_client import get_llama
from llm_prompts import RECOMMENDATIONS_PROMPT

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_stats(
    industry_comp_path: str,
    project_comp_path: str,
    gaps_path: str,
    out_stats_path: str,
    viz_dir: str = "data/derived/plots"
):
    os.makedirs(viz_dir, exist_ok=True)

    ind_comp = load_json(industry_comp_path)
    proj_comp = load_json(project_comp_path)
    gaps_info = load_json(gaps_path)

    # 5) Статистика востребованности компетенций
    industry_counts = defaultdict(Counter)  # industry -> Counter(skill)
    project_counts = defaultdict(Counter)

    for item in ind_comp:
        industry = item.get("industry") or "Unknown"
        for c in item.get("competencies", []):
            name = c.get("name")
            if name:
                industry_counts[industry][name] += 1

    for item in proj_comp:
        industry = item.get("industry") or "Unknown"
        for c in item.get("competencies", []):
            name = c.get("name")
            if name:
                project_counts[industry][name] += 1

    stats = {}

    for industry in industry_counts.keys() | project_counts.keys():
        ind_counter = industry_counts.get(industry, Counter())
        proj_counter = project_counts.get(industry, Counter())
        gaps = gaps_info.get(industry, {}).get("gaps", [])

        top_ind = ind_counter.most_common(20)
        top_proj = proj_counter.most_common(20)

        stats[industry] = {
            "top_industry_competencies": top_ind,
            "top_project_competencies": top_proj,
            "gaps": gaps
        }

        # 6) Визуализация (простые бар-чарты)
        # востребованность индустрией
        if top_ind:
            skills, counts = zip(*top_ind)
            plt.figure(figsize=(10, 5))
            plt.barh(skills, counts)
            plt.gca().invert_yaxis()
            plt.title(f"Top industry competencies – {industry}")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{industry}_industry_top.png"))
            plt.close()

        # покрытие проектами
        if top_proj:
            skills, counts = zip(*top_proj)
            plt.figure(figsize=(10, 5))
            plt.barh(skills, counts)
            plt.gca().invert_yaxis()
            plt.title(f"Top project competencies – {industry}")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{industry}_projects_top.png"))
            plt.close()

    with open(out_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats

def generate_recommendations(
    stats: dict,
    gaps_path: str,
    out_reco_path: str
):
    llama = get_llama()
    gaps_info = load_json(gaps_path)

    recommendations = {}

    for industry, st in stats.items():
        gaps = gaps_info.get(industry, {}).get("gaps", [])
        # для простоты redundancy пока пустое, можно доработать
        redundancy = gaps_info.get(industry, {}).get("redundant_candidates", [])

        industry_stats = {
            "top_industry_competencies": st["top_industry_competencies"],
        }
        project_stats = {
            "top_project_competencies": st["top_project_competencies"],
        }

        prompt = RECOMMENDATIONS_PROMPT.format(
            industry=industry,
            industry_stats=industry_stats,
            project_stats=project_stats,
            gaps=gaps,
            redundancy=redundancy
        )
        raw = llama.ask_one(prompt)
        try:
            reco = json.loads(raw)
        except Exception:
            import re
            match = re.search(r'(\{.*\})', raw, re.S)
            reco = json.loads(match.group(1)) if match else {
                "summary": raw
            }

        recommendations[industry] = reco

    with open(out_reco_path, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    stats = compute_stats(
        "data/derived/industry_competencies_llm.json",
        "data/derived/project_competencies_llm.json",
        "data/derived/competency_gaps_and_redundancy.json",
        "data/derived/stats_and_recommendations.json"
    )
    generate_recommendations(
        stats,
        "data/derived/competency_gaps_and_redundancy.json",
        "data/derived/recommendations.json"
    )
