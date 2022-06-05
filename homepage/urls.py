from django.urls import path
from . import views
from .views import HomeView, show_result

urlpatterns = [
    path('', HomeView.as_view(), name='index'),
    path('show_result/', show_result, name='show_result')
]