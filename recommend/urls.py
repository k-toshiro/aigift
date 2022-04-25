from django.urls import path, include
import django.contrib.auth.views
from . import views

app_name = 'recommend'

urlpatterns=[
    path("", views.toppage, name="toppage"),
    path("by_id", views.home, name="home"),
    path("by_keyword", views.by_keyword, name="by_keyword"),
    path('logout',
        django.contrib.auth.views.LogoutView.as_view(template_name = 'recommend/logout.html'),
        name='logout'),
    path('login', # ログイン
     django.contrib.auth.views.LoginView.as_view(template_name = 'recommend/login.html'),
     name='login'),

    path('create', views.create.as_view(), name='create'),
    path('see', views.see.as_view(), name='see'),
    path('postdata', views.postdata, name='postdata')
]