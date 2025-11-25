# src/build_competency_matrix.py
import json
from collections import defaultdict
from tqdm import tqdm
from llm_client import get_llama
from llm_prompts import MATRIX_SIMILARITY_PROMPT

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def group_by_industry(competencies_list, id_field: str):
    """
    competencies_list: список объектов вида
      { id_field, industry, competencies: [строки или старые dict'ы] }
    """
    res = defaultdict(lambda: defaultdict(list))
    for item in competencies_list:
        ind = item.get("industry") or "Unknown"
        item_id = item[id_field]

        names = []
        for c in item.get("competencies", []):
            if isinstance(c, dict):
                name = c.get("name")
            else:
                name = str(c)
            if name:
                names.append(name.strip())

        res[ind][item_id] = names
    return res

def build_matrices(
    industry_comp_path: str,
    project_comp_path: str,
    matrix_out_path: str,
    gaps_out_path: str
):
    llama = get_llama()

    ind_comp = load_json(industry_comp_path)
    proj_comp = load_json(project_comp_path)

    ind_by_industry = group_by_industry(ind_comp, id_field="vacancy_id")
    proj_by_industry = group_by_industry(proj_comp, id_field="project_id")

    matrices = {}
    gaps_and_redundancy = {}

    for industry, vac_dict in ind_by_industry.items():
        project_dict = proj_by_industry.get(industry, {})

        industry_competencies = list({c for comps in vac_dict.values() for c in comps})
        project_competencies = list({c for comps in project_dict.values() for c in comps})

        if not industry_competencies or not project_competencies:
            continue

        # LLM: матрица соответствия
        prompt = MATRIX_SIMILARITY_PROMPT.format(
            industry_competencies=industry_competencies[:50],  # ограничим
            project_competencies=project_competencies[:50]
        )
        raw = llama.ask_one(prompt)
        try:
            matrix = json.loads(raw)
        except Exception:
            import re
            match = re.search(r'(\{.*\})', raw, re.S)
            matrix = json.loads(match.group(1)) if match else {"matches": []}

        matrices[industry] = matrix

        # Аналитика пробелов/избыточности на основе матрицы
        covered = set()
        for m in matrix.get("matches", []):
            for pc in m.get("project_competencies", []):
                if pc.get("similarity", 0) >= 0.6:
                    covered.add(m["industry_competency"])

        industry_set = set(industry_competencies)
        gaps = sorted(list(industry_set - covered))

        # примитивная логика избыточности:
        # competence проекта, которая почти не встречается в индустрии
        proj_set = set(project_competencies)
        redundant = []  # можно наполнить позже через статистику

        gaps_and_redundancy[industry] = {
            "industry_competencies": industry_competencies,
            "project_competencies": project_competencies,
            "gaps": gaps,
            "redundant_candidates": list(redundant),
        }

    with open(matrix_out_path, "w", encoding="utf-8") as f:
        json.dump(matrices, f, ensure_ascii=False, indent=2)

    with open(gaps_out_path, "w", encoding="utf-8") as f:
        json.dump(gaps_and_redundancy, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    build_matrices(
        "data/derived/industry_competencies_llm.json",
        "data/derived/project_competencies_llm.json",
        "data/derived/competency_matrix.json",
        "data/derived/competency_gaps_and_redundancy.json"
    )
