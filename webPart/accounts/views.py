from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from .forms import UserRegistrationForm, AdminProfileForm, OrdinaryUserProfileForm
from .models import User
from django.contrib.auth.decorators import login_required
from .forms import TaskSubmissionForm
import uuid
import json
import os
from django.contrib import messages
from django.utils import timezone
from django.conf import settings
import psycopg2
from psycopg2.extras import RealDictCursor
from django.core.paginator import Paginator


def get_db_connection():
    """
    Создает подключение к PostgreSQL базе данных
    """
    db_config = settings.DATABASES['default']
    return psycopg2.connect(
        dbname=db_config['NAME'],
        user=db_config['USER'],
        password=db_config['PASSWORD'],
        host=db_config['HOST'],
        port=db_config['PORT']
    )


def save_task_to_database(task_data, file_content):
    """
    Сохраняет задачу в PostgreSQL базу данных
    
    Args:
        task_data: Словарь с данными задачи
        file_content: Содержимое Python файла (строка)
    
    Returns:
        bool: True если успешно, False если ошибка
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # SQL запрос для вставки данных
        insert_query = """
        INSERT INTO tasks 
            (task_id, username, email, user_id_in_app, user_type, 
             task_payload, file_path, original_filename, python_file, timestamp_utc)
        VALUES 
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Конвертируем содержимое файла в байты
        file_bytes = file_content.encode('utf-8')
        
        # Данные для вставки
        values = (
            task_data['task_id'],
            task_data['submitted_by']['username'],
            task_data['submitted_by']['email'],
            task_data['submitted_by']['user_id_in_app'],
            task_data['user_type'],
            task_data['task_payload'],
            task_data['file_path'],
            task_data['original_filename'],
            psycopg2.Binary(file_bytes),  # Сохраняем как BYTEA
            task_data['timestamp_utc']
        )
        
        cursor.execute(insert_query, values)
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Ошибка при сохранении в БД: {e}")
        return False


def get_user_tasks(user_id, limit=None, offset=0):
    """
    Получает список задач пользователя из PostgreSQL
    
    Args:
        user_id: ID пользователя
        limit: Количество записей (None = все)
        offset: Смещение для пагинации
    
    Returns:
        list: Список задач с результатами
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Запрос с LEFT JOIN для получения последнего результата
        query = """
        SELECT 
            t.id,
            t.task_id,
            t.original_filename,
            t.task_payload,
            t.timestamp_utc,
            tr.status,
            tr.solution,
            tr.execution_time_seconds,
            tr.error_message,
            tr.output,
            tr.created_at as result_created_at
        FROM tasks t
        LEFT JOIN LATERAL (
            SELECT * FROM task_results
            WHERE task_id = t.task_id
            ORDER BY created_at DESC
            LIMIT 1
        ) tr ON true
        WHERE t.user_id_in_app = %s
        ORDER BY t.timestamp_utc DESC
        """
        
        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        cursor.execute(query, (user_id,))
        tasks = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return tasks
        
    except Exception as e:
        print(f"Ошибка при получении задач: {e}")
        return []


def get_task_details(task_id, user_id):
    """
    Получает детальную информацию о задаче
    
    Args:
        task_id: UUID задачи
        user_id: ID пользователя (для проверки прав)
    
    Returns:
        dict: Детали задачи или None
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Получаем задачу с проверкой прав доступа
        query = """
        SELECT 
            t.*,
            convert_from(t.python_file, 'UTF8') as python_code
        FROM tasks t
        WHERE t.task_id = %s AND t.user_id_in_app = %s
        """
        
        cursor.execute(query, (task_id, user_id))
        task = cursor.fetchone()
        
        if not task:
            cursor.close()
            conn.close()
            return None
        
        # Получаем все результаты выполнения
        results_query = """
        SELECT * FROM task_results
        WHERE task_id = %s
        ORDER BY created_at DESC
        """
        
        cursor.execute(results_query, (task_id,))
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            'task': dict(task),
            'results': [dict(r) for r in results]
        }
        
    except Exception as e:
        print(f"Ошибка при получении деталей задачи: {e}")
        return None


def get_user_statistics(user_id):
    """
    Получает статистику по задачам пользователя
    
    Args:
        user_id: ID пользователя
    
    Returns:
        dict: Статистика
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
        SELECT 
            COUNT(DISTINCT t.task_id) as total_tasks,
            COUNT(CASE WHEN tr.status = 'completed' THEN 1 END) as completed_tasks,
            COUNT(CASE WHEN tr.status = 'failed' THEN 1 END) as failed_tasks,
            COUNT(CASE WHEN tr.status = 'running' THEN 1 END) as running_tasks,
            COUNT(CASE WHEN tr.status = 'pending' THEN 1 END) as pending_tasks,
            ROUND(AVG(CASE WHEN tr.status = 'completed' THEN tr.execution_time_seconds END), 3) 
                as avg_execution_time
        FROM tasks t
        LEFT JOIN task_results tr ON t.task_id = tr.task_id
        WHERE t.user_id_in_app = %s
        """
        
        cursor.execute(query, (user_id,))
        stats = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return dict(stats) if stats else {}
        
    except Exception as e:
        print(f"Ошибка при получении статистики: {e}")
        return {}


def is_admin(user):
    return user.is_authenticated and user.user_type == 'admin'


def is_ordinary(user):
    return user.is_authenticated and user.user_type == 'ordinary'


def register_view(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Добро пожаловать, {user.username}!')

            if user.user_type == 'admin':
                return redirect('admin_dashboard')
            else:
                return redirect('user_dashboard')
    else:
        form = UserRegistrationForm()

    return render(request, 'accounts/register.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, f'Добро пожаловать, {user.username}!')

            if user.user_type == 'admin':
                return redirect('admin_dashboard')
            else:
                return redirect('user_dashboard')
        else:
            messages.error(request, 'Неверное имя пользователя или пароль')

    return render(request, 'accounts/login.html')


@login_required
def logout_view(request):
    logout(request)
    messages.info(request, 'Вы вышли из системы')
    return redirect('login')


@login_required
@user_passes_test(is_admin, login_url='/access-denied/')
def admin_dashboard(request):
    profile = request.user.admin_profile
    all_users = User.objects.filter(user_type='ordinary').count()
    all_admins = User.objects.filter(user_type='admin').count()

    context = {
        'profile': profile,
        'total_users': all_users,
        'total_admins': all_admins,
    }
    return render(request, 'accounts/admin_dashboard.html', context)


@login_required
@user_passes_test(is_admin, login_url='/access-denied/')
def admin_profile_edit(request):
    profile = request.user.admin_profile

    if request.method == 'POST':
        form = AdminProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Профиль обновлен!')
            return redirect('admin_dashboard')
    else:
        form = AdminProfileForm(instance=profile)

    return render(request, 'accounts/admin_profile_edit.html', {'form': form})


@login_required
@user_passes_test(is_ordinary, login_url='/access-denied/')
def user_dashboard(request):
    profile = request.user.ordinary_profile
    
    # Получаем статистику пользователя
    statistics = get_user_statistics(request.user.id)
    
    # Получаем последние 5 задач
    recent_tasks = get_user_tasks(request.user.id, limit=5)

    context = {
        'profile': profile,
        'statistics': statistics,
        'recent_tasks': recent_tasks,
    }
    return render(request, 'accounts/user_dashboard.html', context)


@login_required
@user_passes_test(is_ordinary, login_url='/access-denied/')
def user_profile_edit(request):
    profile = request.user.ordinary_profile

    if request.method == 'POST':
        form = OrdinaryUserProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Профиль обновлен!')
            return redirect('user_dashboard')
    else:
        form = OrdinaryUserProfileForm(instance=profile)

    return render(request, 'accounts/user_profile_edit.html', {'form': form})


def access_denied(request):
    return render(request, 'accounts/access_denied.html', status=403)


@login_required
def submit_task_view(request):
    """
    Отображает страницу для отправки задачи с Python-кодом,
    сохраняет файл на диск и в PostgreSQL базу данных.
    """
    if request.method == 'POST':
        form = TaskSubmissionForm(request.POST, request.FILES)
        if form.is_valid():
            user = request.user
            task_id = str(uuid.uuid4())
            
            # Получаем данные из формы
            filename = form.cleaned_data['filename']
            python_code = form.cleaned_data.get('python_code')
            python_file = form.cleaned_data.get('python_file')
            
            # Определяем код для сохранения
            if python_file:
                code_content = python_file.read().decode('utf-8')
            else:
                code_content = python_code
            
            # Создаем директорию для задач, если её нет
            tasks_dir = os.path.join(settings.BASE_DIR, 'user_tasks')
            if not os.path.exists(tasks_dir):
                os.makedirs(tasks_dir)
            
            # Создаем поддиректорию для текущего пользователя
            user_tasks_dir = os.path.join(tasks_dir, f'user_{user.id}')
            if not os.path.exists(user_tasks_dir):
                os.makedirs(user_tasks_dir)
            
            # Формируем полный путь к файлу
            safe_filename = f"{task_id}_{filename}"
            file_path = os.path.join(user_tasks_dir, safe_filename)
            
            # Сохраняем Python файл на диск
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(code_content)
            
            # Собираем все данные в словарь
            task_dict = {
                'task_id': task_id,
                'submitted_by': {
                    'username': user.username,
                    'email': user.email,
                    'user_id_in_app': user.id
                },
                'user_type': user.user_type,
                'task_payload': safe_filename,
                'file_path': file_path,
                'original_filename': filename,
                'timestamp_utc': timezone.now().isoformat()
            }

            # Преобразуем словарь в JSON-строку
            json_output = json.dumps(task_dict, indent=4, ensure_ascii=False)

            # Выводим результат в консоль
            print("----------- НОВАЯ ЗАДАЧА -----------")
            print(json_output)
            print("---------------------------------")
            
            # Сохраняем JSON метаданные в файл
            json_file_path = os.path.join(user_tasks_dir, f"{task_id}_metadata.json")
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json_file.write(json_output)
            
            # Сохраняем задачу в PostgreSQL
            db_saved = save_task_to_database(task_dict, code_content)
            
            if db_saved:
                success_message = (
                    f"✅ Задача с ID {task_id} успешно отправлена на обработку!"
                )
                messages.success(request, success_message)
                print("✅ Задача успешно сохранена в PostgreSQL")
            else:
                warning_message = (
                    f"⚠️ Задача {task_id} сохранена локально, но возникла ошибка при сохранении в БД."
                )
                messages.warning(request, warning_message)
                print("⚠️ Ошибка при сохранении в PostgreSQL")
            
            return redirect('my_tasks')
    else:
        form = TaskSubmissionForm()

    return render(request, 'accounts/submit_task.html', {'form': form})


@login_required
@user_passes_test(is_ordinary, login_url='/access-denied/')
def my_tasks_view(request):
    """
    Отображает список всех задач пользователя с пагинацией
    """
    # Получаем все задачи пользователя
    all_tasks = get_user_tasks(request.user.id)
    
    # Пагинация
    paginator = Paginator(all_tasks, 10)  # 10 задач на страницу
    page_number = request.GET.get('page', 1)
    tasks_page = paginator.get_page(page_number)
    
    # Статистика
    statistics = get_user_statistics(request.user.id)
    
    context = {
        'tasks': tasks_page,
        'statistics': statistics,
    }
    
    return render(request, 'accounts/my_tasks.html', context)


@login_required
@user_passes_test(is_ordinary, login_url='/access-denied/')
def task_detail_view(request, task_id):
    """
    Отображает детальную информацию о задаче
    """
    task_details = get_task_details(task_id, request.user.id)
    
    if not task_details:
        messages.error(request, 'Задача не найдена или у вас нет прав для её просмотра')
        return redirect('my_tasks')
    
    context = {
        'task': task_details['task'],
        'results': task_details['results'],
    }
    
    return render(request, 'accounts/task_detail.html', context)