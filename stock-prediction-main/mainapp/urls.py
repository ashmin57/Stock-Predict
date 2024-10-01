from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('news/', views.news, name='news'),  # Added trailing slash
    path('visualization/', views.visualize_csv_form, name='visualization'),  # Trailing slash is already here
    path('data_download/', views.data_download, name='data_download'),  # Trailing slash is already here
    path('auto_download/', views.auto_download, name='auto_download'),  # Added trailing slash
    path('predict', views.predict, name='predict'),  # Added trailing slash
]
