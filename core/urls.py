from django.urls import path
from .views import index, get_box

urlpatterns = [
    path("", index, name="index"),
    path("get-box/", get_box, name="get_box"),
]
