from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from .forms import TaskSubmissionForm
from .api_client import api_client
from asgiref.sync import async_to_sync
import hashlib


def is_admin(user):
    return user.is_authenticated and user.user_type == 'admin'


def is_ordinary(user):
    return user.is_authenticated and user.user_type == 'ordinary'


def hash_password(password: str) -> str:
    """Простое хеширование пароля"""
    return hashlib.sha256(password.encode()).hexdigest()


def register_view(request):
    """Регистрация через FastAPI"""
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password1')
        password2 = request.POST.get('password2')
        user_type = request.POST.get('user_type', 'ordinary')
        
        if password != password2:
            messages.error(request, 'Пароли не совпадают')
            return render(request, 'accounts/register.html')
        
        try:
            hashed_pw = hash_password(password)
            
            if user_type == 'admin':
                result = async_to_sync(api_client.register_admin)(
                    username=username,
                    email=email,
                    password=hashed_pw
                )
            else:
                result = async_to_sync(api_client.register_user)(
                    username=username,
                    email=email,
                    password=hashed_pw
                )
            
            # Создаем локальную сессию Django
            request.session['user_id'] = result['id']
            request.session['username'] = username
            request.session['user_type'] = user_type
            
            messages.success(request, f'Добро пожаловать, {username}!')
            
            if user_type == 'admin':
                return redirect('admin_dashboard')
            else:
                return redirect('user_dashboard')
                
        except Exception as e:
            messages.error(request, f'Ошибка регистрации: {str(e)}')
    
    return render(request, 'accounts/register.html')


def login_view(request):
    """Аутентификация через FastAPI"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        try:
            auth_info = async_to_sync(api_client.get_auth_info)(username)
            hashed_pw = hash_password(password)
            
            if auth_info['password'] == hashed_pw:
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
        except Exception as e:
            messages.error(request, 'Неверное имя пользователя или пароль')
    
    return render(request, 'accounts/login.html')


@login_required
def logout_view(request):
    logout(request)
    messages.info(request, 'Вы вышли из системы')
    return redirect('login')


@login_required
def user_dashboard(request):
    """Дашборд пользователя с данными из FastAPI"""
    user_id = request.session.get('user_id')
    
    try:
        user_data = async_to_sync(api_client.get_user_full)(user_id)
        tasks = async_to_sync(api_client.list_user_tasks)(user_id)
        
        # Статистика
        total_tasks = len(tasks)
        completed = sum(1 for t in tasks if t.get('status') == 'completed')
        running = sum(1 for t in tasks if t.get('status') == 'running')
        pending = sum(1 for t in tasks if t.get('status') == 'pending')
        
        statistics = {
            'total_tasks': total_tasks,
            'completed_tasks': completed,
            'running_tasks': running,
            'pending_tasks': pending
        }
        
        context = {
            'profile': user_data.get('details', {}),
            'user': user_data,
            'statistics': statistics,
            'recent_tasks': tasks[:5]
        }
        
        return render(request, 'accounts/user_dashboard.html', context)
        
    except Exception as e:
        messages.error(request, f'Ошибка загрузки данных: {str(e)}')
        return redirect('login')


@login_required
def submit_task_view(request):
    """Создание задачи через FastAPI"""
    if request.method == 'POST':
        form = TaskSubmissionForm(request.POST, request.FILES)
        if form.is_valid():
            user_id = request.session.get('user_id')
            filename = form.cleaned_data['filename']
            python_code = form.cleaned_data.get('python_code')
            python_file = form.cleaned_data.get('python_file')
            
            # Определяем содержимое
            if python_file:
                content = python_file.read()
            else:
                content = python_code.encode('utf-8')
            
            try:
                # Создаем временный файл для отправки
                from django.core.files.uploadedfile import SimpleUploadedFile
                temp_file = SimpleUploadedFile(
                    filename,
                    content,
                    content_type='text/x-python'
                )
                
                task = async_to_sync(api_client.create_task)(
                    user_id=user_id,
                    name=filename,
                    description=f"Python task: {filename}",
                    files=[temp_file]
                )
                
                messages.success(request, f'✅ Задача {filename} успешно создана!')
                return redirect('my_tasks')
                
            except Exception as e:
                messages.error(request, f'Ошибка создания задачи: {str(e)}')
    else:
        form = TaskSubmissionForm()
    
    return render(request, 'accounts/submit_task.html', {'form': form})


@login_required
def my_tasks_view(request):
    """Список задач пользователя"""
    user_id = request.session.get('user_id')
    
    try:
        tasks = async_to_sync(api_client.list_user_tasks)(user_id)
        
        # Статистика
        total = len(tasks)
        completed = sum(1 for t in tasks if t.get('status') == 'completed')
        running = sum(1 for t in tasks if t.get('status') == 'running')
        pending = sum(1 for t in tasks if t.get('status') == 'pending')
        failed = sum(1 for t in tasks if t.get('status') == 'failed')
        
        statistics = {
            'total_tasks': total,
            'completed_tasks': completed,
            'running_tasks': running,
            'pending_tasks': pending,
            'failed_tasks': failed
        }
        
        context = {
            'tasks': tasks,
            'statistics': statistics
        }
        
        return render(request, 'accounts/my_tasks.html', context)
        
    except Exception as e:
        messages.error(request, f'Ошибка загрузки задач: {str(e)}')
        return render(request, 'accounts/my_tasks.html', {'tasks': [], 'statistics': {}})


@login_required
def task_detail_view(request, task_id):
    """Детали задачи"""
    user_id = request.session.get('user_id')
    
    try:
        tasks = async_to_sync(api_client.list_user_tasks)(user_id)
        task = next((t for t in tasks if t['id'] == task_id), None)
        
        if not task:
            messages.error(request, 'Задача не найдена')
            return redirect('my_tasks')
        
        # Получаем файлы
        input_files = async_to_sync(api_client.get_task_input_files)(task_id)
        result_files = async_to_sync(api_client.get_task_result_files)(task_id)
        
        context = {
            'task': task,
            'input_files': input_files,
            'result_files': result_files
        }
        
        return render(request, 'accounts/task_detail.html', context)
        
    except Exception as e:
        messages.error(request, f'Ошибка загрузки данных: {str(e)}')
        return redirect('my_tasks')


@login_required
@user_passes_test(is_admin, login_url='/access-denied/')
def admin_dashboard(request):
    """Панель администратора"""
    user_id = request.session.get('user_id')
    
    try:
        user_data = async_to_sync(api_client.get_user_full)(user_id)
        
        context = {
            'profile': user_data.get('admin', {}),
            'user': user_data,
            'total_users': 0,  # TODO: добавить endpoint в API
            'total_admins': 0
        }
        
        return render(request, 'accounts/admin_dashboard.html', context)
        
    except Exception as e:
        messages.error(request, f'Ошибка загрузки данных: {str(e)}')
        return redirect('login')


@login_required
def user_profile_edit(request):
    """Редактирование профиля через API"""
    user_id = request.session.get('user_id')
    
    if request.method == 'POST':
        update_data = {
            'bio': request.POST.get('bio'),
            'phone': request.POST.get('phone'),
            'city': request.POST.get('city'),
            'birth_date': request.POST.get('birth_date')
        }
        
        try:
            async_to_sync(api_client.update_user)(user_id, update_data)
            messages.success(request, 'Профиль обновлен!')
            return redirect('user_dashboard')
        except Exception as e:
            messages.error(request, f'Ошибка обновления: {str(e)}')
    
    try:
        user_data = async_to_sync(api_client.get_user_full)(user_id)
        return render(request, 'accounts/user_profile_edit.html', {'profile': user_data.get('details', {})})
    except Exception as e:
        messages.error(request, f'Ошибка загрузки данных: {str(e)}')
        return redirect('user_dashboard')


@login_required
@user_passes_test(is_admin, login_url='/access-denied/')
def admin_profile_edit(request):
    """Редактирование профиля администратора"""
    user_id = request.session.get('user_id')
    
    if request.method == 'POST':
        update_data = {
            'department': request.POST.get('department'),
            'phone': request.POST.get('phone'),
            'permissions_level': int(request.POST.get('permissions_level', 1)),
            'access_code': request.POST.get('access_code')
        }
        
        try:
            async_to_sync(api_client.update_user)(user_id, update_data)
            messages.success(request, 'Профиль обновлен!')
            return redirect('admin_dashboard')
        except Exception as e:
            messages.error(request, f'Ошибка обновления: {str(e)}')
    
    try:
        user_data = async_to_sync(api_client.get_user_full)(user_id)
        return render(request, 'accounts/admin_profile_edit.html', {'profile': user_data.get('admin', {})})
    except Exception as e:
        messages.error(request, f'Ошибка загрузки данных: {str(e)}')
        return redirect('admin_dashboard')


def access_denied(request):
    return render(request, 'accounts/access_denied.html', status=403)
