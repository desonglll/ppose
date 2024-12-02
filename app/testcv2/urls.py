from django.urls import path
from . import views

urlpatterns = [
    path('video-feed/', views.video_feed, name='video_feed'),  # 视频流视图
    path('video-page-single/', views.single_video_page, name='video_page_single'),
    path('video-page-double/', views.double_video_page, name='video_page_double'),
]
