"""
Django views для работы через FastAPI бэкенд
Все операции с БД идут только через HTTP API
"""
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.contrib.sessions.backends.db import SessionStore
from django.http import HttpResponse
import httpx
from typing import Optional

# Предполагается, что api_client.py находится в той же директории
from .api_client import api_client


# ==================== HELPER FUNCTIONS ====================

def get_user_from_session(request):
    """Получение данных пользователя из сессии"""
    user_id = request.session.get('user_id')
    if not user_id:
        return None

    try:
        user_data = api_client.get_user_full(user_id)
        return user_data
    except httpx.HTTPError:
        return None


def login_required_api(view_func):
    """Декоратор для проверки авторизации через API"""

    def wrapper(request, *args, **kwargs):
        user = get_user_from_session(request)
        if not user:
            messages.error(request, 'Необходимо войти в систему')
            return redirect('login')
        request.user_api = user
        return view_func(request, *args, **kwargs)

    return wrapper


def admin_required(view_func):
    """Декоратор для проверки прав администратора"""

    @login_required_api
    def wrapper(request, *args, **kwargs):
        if request.user_api['user_type'] != 'admin':
            return redirect('access_denied')
        return view_func(request, *args, **kwargs)

    return wrapper


def ordinary_user_required(view_func):
    """Декоратор для проверки прав обычного пользователя"""

    @login_required_api
    def wrapper(request, *args, **kwargs):
        if request.user_api['user_type'] != 'user':
            return redirect('access_denied')
        return view_func(request, *args, **kwargs)

    return wrapper


# ==================== AUTHENTICATION VIEWS ====================

@require_http_methods(["GET", "POST"])
def register_view(request):
    """Регистрация нового пользователя"""
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password1')
        password2 = request.POST.get('password2')
        user_type = request.POST.get('user_type', 'ordinary')

        # Валидация
        if password != password2:
            messages.error(request, 'Пароли не совпадают')
            return render(request, 'accounts/register.html')

        try:
            if user_type == 'admin':
                result = api_client.register_admin(
                    username=username,
                    email=email,
                    password=password,
                    department=request.POST.get('department'),
                    phone=request.POST.get('phone'),
                )
            else:
                result = api_client.register_user(
                    username=username,
                    email=email,
                    password=password,
                    bio=request.POST.get('bio'),
                    phone=request.POST.get('phone'),
                    city=request.POST.get('city'),
                )

            # Автоматический вход после регистрации
            request.session['user_id'] = result['id']
            request.session['username'] = result['username']
            request.session['user_type'] = result['user_type']

            messages.success(request, f'Добро пожаловать, {username}!')

            if user_type == 'admin':
                return redirect('admin_dashboard')
            else:
                return redirect('user_dashboard')

        except httpx.HTTPError as e:
            messages.error(request, f'Ошибка регистрации: {str(e)}')

    return render(request, 'accounts/register.html')


@require_http_methods(["GET", "POST"])
def login_view(request):
    """Вход в систему"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        try:
            # Получаем данные пользователя через API
            auth_info = api_client.get_auth_info(username)

            # Проверяем пароль (в реальном проекте используйте хеширование!)
            if auth_info['password'] == password:
                # Сохраняем в сессию
                request.session['user_id'] = auth_info['id']
                request.session['username'] = auth_info['username']
                request.session['user_type'] = auth_info['user_type']

                messages.success(request, f'Добро пожаловать, {username}!')

                if auth_info['user_type'] == 'admin':
                    return redirect('admin_dashboard')
                else:
                    return redirect('user_dashboard')
            else:
                messages.error(request, 'Неверное имя пользователя или пароль')

        except httpx.HTTPError:
            messages.error(request, 'Пользователь не найден')

    return render(request, 'accounts/login.html')


def logout_view(request):
    """Выход из системы"""
    request.session.flush()
    messages.info(request, 'Вы вышли из системы')
    return redirect('login')


# ==================== ADMIN VIEWS ====================

@admin_required
def admin_dashboard(request):
    """Панель администратора"""
    user_data = request.user_api

    # Здесь можно добавить статистику через API
    context = {
        'user': user_data,
        'profile': user_data.get('admin', {}),
        'total_users': 0,  # TODO: добавить endpoint в API
        'total_admins': 0,  # TODO: добавить endpoint в API
    }
    return render(request, 'accounts/admin_dashboard.html', context)


@admin_required
@require_http_methods(["GET", "POST"])
def admin_profile_edit(request):
    """Редактирование профиля администратора"""
    user_data = request.user_api

    if request.method == 'POST':
        try:
            update_data = {
                'department': request.POST.get('department'),
                'phone': request.POST.get('phone'),
                'permissions_level': int(request.POST.get('permissions_level', 1)),
                'access_code': request.POST.get('access_code'),
            }

            api_client.update_user(user_data['id'], **update_data)
            messages.success(request, 'Профиль обновлен!')
            return redirect('admin_dashboard')

        except httpx.HTTPError as e:
            messages.error(request, f'Ошибка обновления: {str(e)}')

    context = {
        'user': user_data,
        'profile': user_data.get('admin', {}),
    }
    return render(request, 'accounts/admin_profile_edit.html', context)


# ==================== USER VIEWS ====================

@ordinary_user_required
def user_dashboard(request):
    """Панель пользователя"""
    user_data = request.user_api

    try:
        # Получаем задачи пользователя
        tasks = api_client.list_user_tasks(user_data['id'])
        recent_tasks = tasks[:5] if tasks else []

        # Вычисляем статистику
        statistics = {
            'total_tasks': len(tasks),
            'completed_tasks': sum(1 for t in tasks if t.get('status') == 'completed'),
            'failed_tasks': sum(1 for t in tasks if t.get('status') == 'failed'),
            'running_tasks': sum(1 for t in tasks if t.get('status') == 'running'),
            'pending_tasks': sum(1 for t in tasks if t.get('status') == 'pending'),
        }
    except httpx.HTTPError:
        recent_tasks = []
        statistics = {}

    context = {
        'user': user_data,
        'profile': user_data.get('details', {}),
        'recent_tasks': recent_tasks,
        'statistics': statistics,
    }
    return render(request, 'accounts/user_dashboard.html', context)


@ordinary_user_required
@require_http_methods(["GET", "POST"])
def user_profile_edit(request):
    """Редактирование профиля пользователя"""
    user_data = request.user_api

    if request.method == 'POST':
        try:
            update_data = {
                'bio': request.POST.get('bio'),
                'phone': request.POST.get('phone'),
                'city': request.POST.get('city'),
                'birth_date': request.POST.get('birth_date'),
                'subscription_active': request.POST.get('subscription_active') == 'on',
            }

            # Обработка аватара отдельно
            if 'avatar' in request.FILES:
                api_client.upload_avatar(user_data['id'], request.FILES['avatar'])

            api_client.update_user(user_data['id'], **update_data)
            messages.success(request, 'Профиль обновлен!')
            return redirect('user_dashboard')

        except httpx.HTTPError as e:
            messages.error(request, f'Ошибка обновления: {str(e)}')

    context = {
        'user': user_data,
        'profile': user_data.get('details', {}),
    }
    return render(request, 'accounts/user_profile_edit.html', context)


# ==================== TASK VIEWS ====================

@ordinary_user_required
@require_http_methods(["GET", "POST"])
def submit_task_view(request):
    """Создание новой задачи"""
    if request.method == 'POST':
        filename = request.POST.get('filename', 'script.py')
        python_code = request.POST.get('python_code')
        python_file = request.FILES.get('python_file')

        try:
            # Создаем временный файл с кодом
            files_to_upload = []

            if python_file:
                files_to_upload.append(python_file)
            elif python_code:
                # Создаем временный файл из кода
                from django.core.files.uploadedfile import SimpleUploadedFile
                temp_file = SimpleUploadedFile(
                    filename,
                    python_code.encode('utf-8'),
                    content_type='text/x-python'
                )
                files_to_upload.append(temp_file)

            # Создаем задачу через API
            task = api_client.create_task(
                user_id=request.user_api['id'],
                name=filename,
                description=f"Python script: {filename}",
                files=files_to_upload if files_to_upload else None
            )

            messages.success(request, f'✅ Задача {task["name"]} успешно создана!')
            return redirect('my_tasks')

        except httpx.HTTPError as e:
            messages.error(request, f'Ошибка создания задачи: {str(e)}')

    return render(request, 'accounts/submit_task.html')


@ordinary_user_required
def my_tasks_view(request):
    """Список задач пользователя"""
    try:
        tasks = api_client.list_user_tasks(request.user_api['id'])

        # Статистика
        statistics = {
            'total_tasks': len(tasks),
            'completed_tasks': sum(1 for t in tasks if t.get('status') == 'completed'),
            'failed_tasks': sum(1 for t in tasks if t.get('status') == 'failed'),
            'running_tasks': sum(1 for t in tasks if t.get('status') == 'running'),
            'pending_tasks': sum(1 for t in tasks if t.get('status') == 'pending'),
        }
    except httpx.HTTPError:
        tasks = []
        statistics = {}

    context = {
        'tasks': tasks,
        'statistics': statistics,
    }
    return render(request, 'accounts/my_tasks.html', context)


@ordinary_user_required
def task_detail_view(request, task_id: int):
    """Детали задачи"""
    try:
        # Получаем список всех задач и находим нужную
        tasks = api_client.list_user_tasks(request.user_api['id'])
        task = next((t for t in tasks if t['id'] == task_id), None)

        if not task:
            messages.error(request, 'Задача не найдена')
            return redirect('my_tasks')

        # Получаем файлы
        input_files = api_client.get_input_files(task_id)
        result_files = api_client.get_result_files(task_id)

        context = {
            'task': task,
            'input_files': input_files,
            'result_files': result_files,
        }
        return render(request, 'accounts/task_detail.html', context)

    except httpx.HTTPError as e:
        messages.error(request, f'Ошибка загрузки задачи: {str(e)}')
        return redirect('my_tasks')


# ==================== UTILITY VIEWS ====================

def access_denied(request):
    """Страница отказа в доступе"""
    return render(request, 'accounts/access_denied.html', status=403)


def health_check(request):
    """Проверка состояния системы"""
    try:
        api_status = api_client.health_check()
        return HttpResponse(f"Django: OK, API: {api_status}", status=200)
    except:
        return HttpResponse("Django: OK, API: ERROR", status=500)
