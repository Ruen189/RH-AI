from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_POST
import config as cfg
import fetch_sj as sj


def index(request):
    if request.method == "POST":
        # реальные ключи
        for name in ("sj_api_key", "hf_token"):
            val = request.POST.get(name, "").strip()
            if val:
                request.session[name] = val

        request.session.modified = True
        return redirect(request.path)

    return render(request, "qloraapp/index.html")

def collect_jobs(request):
    return render(request, 'qloraapp/jobs.html')

@require_POST
def start_collect_jobs(request):
    sj_key = request.session.get("sj_api_key")
    if not sj_key:
        return JsonResponse({"ok": False, "error": "Не задан SuperJob API key"}, status=400)

    try:
        keywords = cfg.load_industry_keywords()
        out = sj.collect_sj_batch(
            sj_api_key=sj_key,
            industry_keywords=keywords,
            out_dir=cfg.RAW_DIR,
            pages=3,
        )
        return JsonResponse({"ok": True, "file": out})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)

def analyze_vacancies(request):
    return render(request, 'qloraapp/vacancies.html')

def analyze_project_data(request):
    return render(request, 'qloraapp/project_data.html')

def matrices_and_statistics(request):
    return render(request, 'qloraapp/statistics.html')