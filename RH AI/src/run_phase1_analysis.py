# src/run_phase1_analysis.py
from analyze_vacancies_llm import analyze_vacancies
from analyze_projects_llm import analyze_projects
from build_competency_matrix import build_matrices
from generate_stats_and_reports import compute_stats, generate_recommendations
from filter_competency_matrix import main as filter_matrix  # <-- добавили импорт


def main():

    # 1) вакансии
    analyze_vacancies(
        "data/processed/vacancies_processed.json",
        "data/derived/industry_competencies_llm_clean_updated.json",
    )

    # 2) проекты
    analyze_projects(
        "data/raw/projects_with_industries_full.json",
        "data/derived/project_competencies_llm_clean_updated.json",
    )

    # 3) матрица спрос/предложение
    build_matrices(
        "data/derived/industry_competencies_llm_clean_updated.json",
        "data/derived/project_competencies_llm_clean_updated.json",
        "data/derived/competency_matrix.json",
        "data/derived/competency_gaps_and_redundancy.json",
    )

    # 3.1) фильтрация матрицы по белому списку
    filter_matrix()  # создаст competency_matrix_filtered.json

    # 4) статистика + рекомендации
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

if __name__ == "__main__":
    main()
