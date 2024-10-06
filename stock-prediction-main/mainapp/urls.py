from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Homepage
    path('news/', views.news, name='news'),  # News page
    path('visualization/', views.visualize_csv_form, name='visualization'),  # CSV visualization page
    path('admin_dashboard/', views.admin_dashboard, name='admin_dashboard'),  # Admin dashboard for model training
    path('auto_download/', views.auto_download, name='auto_download'),  # Auto download functionality
    path('predict', views.predict, name='predict'),  # Prediction endpoint
]
