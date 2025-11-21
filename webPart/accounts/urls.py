"""
URL configuration for webPart project - API Integration Version
"""
from django.contrib import admin
from django.urls import path
from django.shortcuts import redirect
from django.conf import settings
from django.conf.urls.static import static

# Импортируем views из файла views.py
from . import views

urlpatterns = [
    # Admin (может не использоваться)
    path('admin/', admin.site.urls),

    # Главная страница - редирект на login
    path('', lambda request: redirect('login')),

    # ==================== AUTHENTICATION ====================
    path('accounts/register/', views.register_view, name='register'),
    path('accounts/login/', views.login_view, name='login'),
    path('accounts/logout/', views.logout_view, name='logout'),

    # ==================== ADMIN URLS ====================
    path('accounts/admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('accounts/admin-profile/edit/', views.admin_profile_edit, name='admin_profile_edit'),

    # ==================== USER URLS ====================
    path('accounts/dashboard/', views.user_dashboard, name='user_dashboard'),
    path('accounts/profile/edit/', views.user_profile_edit, name='user_profile_edit'),

    # ==================== TASK MANAGEMENT ====================
    path('accounts/submit/', views.submit_task_view, name='submit_task'),
    path('accounts/my-tasks/', views.my_tasks_view, name='my_tasks'),
    path('accounts/task/<int:task_id>/', views.task_detail_view, name='task_detail'),

    # ==================== UTILITY ====================
    path('accounts/access-denied/', views.access_denied, name='access_denied'),
    path('health/', views.health_check, name='health_check'),
]

# Для разработки - обслуживание медиа файлов (не используется, но оставляем)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
