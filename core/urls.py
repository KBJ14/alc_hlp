from django.urls import path
from .views import index, get_box, get_segmentation

urlpatterns = [
    path("", index, name="index"),
    path("get-box/", get_box, name="get_box"),
    path("get-segmentation/", get_segmentation, name="get_segmentation"),
]
