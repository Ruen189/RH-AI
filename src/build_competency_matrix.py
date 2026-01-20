# src/build_competency_matrix.py

import json
from collections import defaultdict, Counter
from typing import Dict, List


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_competencies(raw) -> List[str]:
    """
    Приводим поле competencies к списку строк.
    Игнорируем '-', None и пустые строка.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        if raw.strip() in ("", "-"):
            return []
        # на всякий случай — если вдруг компетенции пришли через запятую
        parts = [p.strip() for p in raw.split(",")]
        return [p for p in parts if p]
    if isinstance(raw, list):
        result = []
        for c in raw:
            if not isinstance(c, str):
                continue
            c = c.strip()
            if not c or c == "-":
                continue
            result.append(c)
        return result
    return []


def build_demand_by_industry(industry_file: str) -> Dict[str, Counter]:
    """
    Спрос: по вакансиям.
    industry_competencies_llm_clean_updated.json
    [
      { "vacancy_id": "...", "industry": "...", "title": "...", "competencies": [...] },
      ...
    ]
    """
    data = load_json(industry_file)
    demand: Dict[str, Counter] = defaultdict(Counter)

    for item in data:
        industry = item.get("industry") or "Unknown"
        comps = normalize_competencies(item.get("competencies"))
        for comp in comps:
            demand[industry][comp] += 1

    return demand


def build_supply_by_industry(project_file: str) -> Dict[str, Counter]:
    """
    Предложение: по проектам.
    project_competencies_llm_clean_updated.json
    [
      { "project_id": ..., "industry": "AI/EdTech", "title": "...", "competencies": [...] },
      ...
    ]

    В проекте может быть несколько индустрий через "/":
      "AI/EdTech/GameDev" -> ["AI", "EdTech", "GameDev"]
    Каждой такой индустрии начисляем компетенции этого проекта.
    """
    data = load_json(project_file)
    supply: Dict[str, Counter] = defaultdict(Counter)

    for item in data:
        raw_industry = item.get("industry") or "Unknown"
        # делим по "/", т.к. часто бывают комбинированные индустрии
        industries = [part.strip() for part in str(raw_industry).split("/") if part.strip()]
        if not industries:
            industries = ["Unknown"]

        comps = normalize_competencies(item.get("competencies"))
        for industry in industries:
            for comp in comps:
                supply[industry][comp] += 1

    return supply


def build_matrices(
    industry_input: str,
    project_input: str,
    matrix_output: str,
    gaps_output: str,
):
    # 1) агрегируем спрос и предложение
    demand = build_demand_by_industry(industry_input)
    supply = build_supply_by_industry(project_input)

    # множество всех индустрий
    all_industries = sorted(set(demand.keys()) | set(supply.keys()))

    matrix_rows = []          # для competency_matrix.json
    industry_summaries = []   # для competency_gaps_and_redundancy.json

    for industry in all_industries:
        demand_counter = demand.get(industry, Counter())
        supply_counter = supply.get(industry, Counter())

        all_comps = sorted(set(demand_counter.keys()) | set(supply_counter.keys()))

        gaps = []
        redundancies = []
        matches = []

        for comp in all_comps:
            d = demand_counter.get(comp, 0)
            s = supply_counter.get(comp, 0)

            if d > 0 and s > 0:
                status = "match"
                matches.append({"competency": comp, "demand": d, "supply": s})
            elif d > 0 and s == 0:
                status = "gap"
                gaps.append({"competency": comp, "demand": d})
            elif d == 0 and s > 0:
                status = "redundant"
                redundancies.append({"competency": comp, "supply": s})
            else:
                # d == 0 and s == 0 — сюда вообще не должны попасть, т.к. all_comps — объединение ключей
                continue

            matrix_rows.append(
                {
                    "industry": industry,
                    "competency": comp,
                    "demand": d,
                    "supply": s,
                    "status": status,
                }
            )

        industry_summaries.append(
            {
                "industry": industry,
                "total_demand_competencies": len(
                    [c for c in all_comps if demand_counter.get(c, 0) > 0]
                ),
                "total_supply_competencies": len(
                    [c for c in all_comps if supply_counter.get(c, 0) > 0]
                ),
                "gaps": gaps,
                "redundancies": redundancies,
                "matches": matches,
            }
        )

    # 2) сохраняем результаты
    with open(matrix_output, "w", encoding="utf-8") as f:
        json.dump(matrix_rows, f, ensure_ascii=False, indent=2)

    with open(gaps_output, "w", encoding="utf-8") as f:
        json.dump(industry_summaries, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Competency matrix saved to: {matrix_output}")
    print(f"[INFO] Gaps/redundancy saved to: {gaps_output}")


if __name__ == "__main__":
    build_matrices(
        "data/derived/industry_competencies_llm_clean_updated.json",
        "data/derived/project_competencies_llm_clean_updated.json",
        "data/derived/competency_matrix.json",
        "data/derived/competency_gaps_and_redundancy.json",
    )
