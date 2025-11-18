from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Admin URLs
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('admin-profile/edit/', views.admin_profile_edit, name='admin_profile_edit'),

    # User URLs
    path('dashboard/', views.user_dashboard, name='user_dashboard'),
    path('profile/edit/', views.user_profile_edit, name='user_profile_edit'),

    # Task management URLs
    path('submit/', views.submit_task_view, name='submit_task'),
    path('my-tasks/', views.my_tasks_view, name='my_tasks'),
    path('task/<uuid:task_id>/', views.task_detail_view, name='task_detail'),

    # Access denied
    path('access-denied/', views.access_denied, name='access_denied'),
]