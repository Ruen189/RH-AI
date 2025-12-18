from django.urls import path, include
from . import views


urlpatterns = [
    path('', views.index, name='home'),
    path('jobs/', views.collect_jobs, name='jobs'),
    path('vacancies/', views.analyze_vacancies, name='vacancies'),
    path('project_data/', views.analyze_project_data, name='project_data'),
    path('statistics/', views.matrices_and_statistics, name='statistics'),
]