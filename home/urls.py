from django.urls import path, include
from home import views


urlpatterns = [
    path('', views.index, name='index'),
    path('home', views.home, name='home'),
    path('video_feed', views.video_feed, name='video_feed'),
    ]
