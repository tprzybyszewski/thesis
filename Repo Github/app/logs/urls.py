from django.urls import path
from . import views

urlpatterns = [
    path('', views.log_view, name='log_view'),
]
