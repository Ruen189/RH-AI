# src/run_phase1_analysis.py
from analyze_vacancies_llm import analyze_vacancies
from analyze_projects_llm import analyze_projects
from build_competency_matrix import build_matrices
from generate_stats_and_reports import compute_stats, generate_recommendations

def main():
    # 1) индустриальные требования
    analyze_vacancies(
        "data/processed/vacancies_processed.json",
        "data/derived/industry_competencies_llm.json"
    )
    # 2) компетенции проектов
    analyze_projects(
        "data/processed/projects_with_industries_full.json",
        "data/derived/project_competencies_llm.json"
    )
    # 3–4) матрицы и пробелы
    build_matrices(
        "data/derived/industry_competencies_llm.json",
        "data/derived/project_competencies_llm.json",
        "data/derived/competency_matrix.json",
        "data/derived/competency_gaps_and_redundancy.json"
    )
    # 5–7) статистика, визуализации и рекомендации
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

if __name__ == "__main__":
    main()
