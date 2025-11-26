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


def extract_competency_name(c):
    """ Унифицированное извлечение имени компетенции (строка или старый dict). """
    if isinstance(c, str):
        c = c.strip()
        return c if c and c != "-" else None
    if isinstance(c, dict):
        name = c.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None

def jitter_points(xs, ys, x_amount=0.25, y_amount=0.08):
    """
    Раздвигает точки, которые имеют одинаковые координаты.
    x_amount — сила jitter по оси X (больше),
    y_amount — сила jitter по оси Y (меньше).
    """
    from collections import defaultdict
    import math

    grouped = defaultdict(list)
    for idx, (x, y) in enumerate(zip(xs, ys)):
        grouped[(x, y)].append(idx)

    xs2 = xs[:]
    ys2 = ys[:]

    for (x, y), indices in grouped.items():
        if len(indices) <= 1:
            continue

        n = len(indices)
        for k, idx in enumerate(indices):
            angle = 2 * math.pi * k / n
            xs2[idx] = x + x_amount * math.cos(angle)
            ys2[idx] = y + y_amount * math.sin(angle)

    return xs2, ys2


def compute_stats(
    industry_comp_path: str,
    project_comp_path: str,
    gaps_path: str,
    out_stats_path: str,
    viz_dir: str = "data/derived/plots",
):
    """
    5) Генерация статистики по востребованности компетенций.
    + 3 глобальных графика:
      1) Топ компетенций в проектах (по всем индустриям).
      2) Топ компетенций в вакансиях (по всем индустриям).
      3) Scatter для компетенций со статусом 'match'
         (x = supply в проектах, y = demand в вакансиях, все точки подписаны).
    """
    os.makedirs(viz_dir, exist_ok=True)

    ind_comp = load_json(industry_comp_path)
    proj_comp = load_json(project_comp_path)
    gaps_info = load_json(gaps_path)

    # ============ СЧЁТЧИКИ ============
    industry_counts = defaultdict(Counter)   # спрос по индустриям
    project_counts = defaultdict(Counter)    # предложение по индустриям

    global_industry_counter = Counter()      # спрос по всем индустриям
    global_project_counter = Counter()       # предложение по всем индустриям

    # --- Спрос (вакансии) ---
    for item in ind_comp:
        industry = item.get("industry") or "Unknown"
        for c in item.get("competencies", []):
            name = extract_competency_name(c)
            if name:
                industry_counts[industry][name] += 1
                global_industry_counter[name] += 1

    # --- Предложение (проекты) ---
    for item in proj_comp:
        industry = item.get("industry") or "Unknown"
        for c in item.get("competencies", []):
            name = extract_competency_name(c)
            if name:
                project_counts[industry][name] += 1
                global_project_counter[name] += 1

    # ============ СТАТИСТИКИ ПО ИНДУСТРИЯМ ============
    stats = {}

    all_industries = set(industry_counts.keys()) | set(project_counts.keys())

    for industry in all_industries:
        ind_counter = industry_counts.get(industry, Counter())
        proj_counter = project_counts.get(industry, Counter())

        top_ind = ind_counter.most_common(20)
        top_proj = proj_counter.most_common(20)

        # достаём gaps из gaps_info (он у нас из build_matrices)
        industry_gaps = []
        if isinstance(gaps_info, list):
            for entry in gaps_info:
                if entry.get("industry") == industry:
                    industry_gaps = entry.get("gaps", [])
                    break
        elif isinstance(gaps_info, dict):
            industry_gaps = gaps_info.get(industry, {}).get("gaps", [])

        stats[industry] = {
            "top_industry_competencies": top_ind,
            "top_project_competencies": top_proj,
            "gaps": industry_gaps,
        }

    # сохраняем статистику в JSON
    with open(out_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ============ ГЛОБАЛЬНЫЕ ГРАФИКИ (3 ШТУКИ) ============

    # 1) Топ компетенций в проектах (global_project_counter)
    if global_project_counter:
        top_proj_global = global_project_counter.most_common(30)
        skills, counts = zip(*top_proj_global)
        plt.figure(figsize=(12, 6))
        plt.barh(skills, counts)
        plt.gca().invert_yaxis()
        plt.title("Топ компетенций в учебных проектах (общий по всем индустриям)")
        plt.xlabel("Количество упоминаний в проектах")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "global_projects_top.png"))
        plt.close()

    # 2) Топ компетенций в вакансиях (global_industry_counter)
    if global_industry_counter:
        # --- загрузка белого списка ---
        whitelist_path = "data/whitelist_competencies.json"
        with open(whitelist_path, "r", encoding="utf-8") as f:
            whitelist = set(json.load(f))

        # --- глобальные топы с фильтрацией ---
        filtered_global_industry = {
            skill: count for skill, count in global_industry_counter.items()
            if skill in whitelist
        }
        top_ind_global = Counter(filtered_global_industry).most_common(30)

        skills, counts = zip(*top_ind_global)
        plt.figure(figsize=(12, 6))
        plt.barh(skills, counts)
        plt.gca().invert_yaxis()
        plt.title("Топ компетенций в вакансиях (общий по всем индустриям)")
        plt.xlabel("Количество упоминаний в вакансиях")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "global_industry_top.png"))
        plt.close()

    # 3) Scatter для компетенций со статусом 'match'
    matrix_path_filtered = "data/derived/competency_matrix_filtered.json"
    if os.path.exists(matrix_path_filtered):
        matrix = load_json(matrix_path_filtered)

        xs = []
        ys = []
        labels = []

        for row in matrix:
            if row.get("status") != "match":
                continue
            comp = row.get("competency")
            d = row.get("demand", 0)
            s = row.get("supply", 0)
            if not isinstance(comp, str):
                continue
            # если обе частоты нулевые — смысла нет
            if d == 0 and s == 0:
                continue
            xs.append(s)
            ys.append(d)
            labels.append(comp)

        if xs and ys:
            plt.figure(figsize=(14, 12))
            # слегка раздвинем совпадающие точки
            xs_jit, ys_jit = jitter_points(xs, ys, x_amount=0.5, y_amount=0.4)

            plt.scatter(xs_jit, ys_jit, alpha=0.7)

            # подписи ВСЕХ компетенций
            for label, sx, sy in zip(labels, xs_jit, ys_jit):
                plt.annotate(
                    label,
                    (sx, sy),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=14,
                    alpha=0.85,
                )

            plt.xlabel(
                "Сколько раз встречается в проектах (supply)",
                fontsize=30,
            )
            plt.ylabel(
                "Сколько раз встречается в вакансиях (demand)",
                fontsize=30,
            )
            plt.title(
                "Компетенции со статусом MATCH (есть в обоих случаях)",
                fontsize=30,
                pad=20
            )
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "match_scatter.png"))
            plt.close()

    return stats


def generate_recommendations(
    stats: dict,
    gaps_path: str,
    out_reco_path: str,
    log_path: str = "data/derived/reco_log.txt",
):
    llama = get_llama()
    gaps_info = load_json(gaps_path)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")

    def log(msg):
        log_f.write(msg + "\n")

    # приводим gaps_info к списку индустрий
    if isinstance(gaps_info, list):
        entries = gaps_info
    else:
        entries = [{"industry": k, **v} for k, v in gaps_info.items()]

    recommendations = {}

    for entry in entries:
        industry = entry["industry"]
        gaps = entry.get("gaps", [])

        st = stats.get(industry, {})
        top_ind = st.get("top_industry_competencies", [])
        top_proj = st.get("top_project_competencies", [])

        # если нет данных (ни проектов, ни вакансий)
        if not top_ind and not top_proj:
            recommendations[industry] = "Недостаточно данных для анализа индустрии."
            continue

        # если нет gaps — нет дефицита → рекомендация не нужна
        if not gaps:
            recommendations[industry] = "Дефицитные компетенции отсутствуют."
            continue

        # очистим gaps: уберём компетенции по названию индустрии
        cleaned_gaps = [
            g for g in gaps
            if g.get("competency") != industry
        ]

        prompt = RECOMMENDATIONS_PROMPT.format(
            industry=industry,
            industry_stats=top_ind,
            project_stats=top_proj,
            gaps=cleaned_gaps,
            redundancy=entry.get("redundancies", []),
        )

        log(f"=== INDUSTRY: {industry} ===")
        log("PROMPT:\n" + prompt)

        raw = llama.ask_one(prompt)
        log("\nLLM RESPONSE:\n" + raw + "\n")

        text = raw.strip() or "Модель не вернула ответа."

        recommendations[industry] = text

    log_f.close()

    with open(out_reco_path, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    stats = compute_stats(
        "data/derived/industry_competencies_llm_clean_updated.json",
        "data/derived/project_competencies_llm_clean_updated.json",
        "data/derived/competency_gaps_and_redundancy.json",
        "data/derived/stats.json",
    )

    generate_recommendations(
        stats,
        "data/derived/competency_gaps_and_redundancy.json",
        "data/derived/recommendations.json",
    )

