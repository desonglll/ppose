from django.urls import path
from . import views

urlpatterns = [
    path('video-feed/', views.video_feed, name='video_feed'),  # 视频流视图
    path('video-page/', views.video_page, name='video_page'),

]
