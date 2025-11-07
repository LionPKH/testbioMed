from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from .forms import UserRegistrationForm, AdminProfileForm, OrdinaryUserProfileForm
from .models import User
from django.contrib.auth.decorators import login_required
from .forms import TaskSubmissionForm
import uuid
import json
from django.contrib import messages
from django.utils import timezone



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

    context = {
        'profile': profile,
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
    Отображает страницу для отправки задачи и обрабатывает отправку,
    формируя JSON-объект.
    """
    if request.method == 'POST':
        form = TaskSubmissionForm(request.POST)
        if form.is_valid():
            user = request.user
            task_data = form.cleaned_data['task_input']
            task_id = str(uuid.uuid4())

            # Собираем все данные в словарь Python
            task_dict = {
                'task_id': task_id,
                'submitted_by': {
                    'username': user.username,
                    'email': user.email,
                    'user_id_in_app': user.id
                },
                'user_type': user.user_type,
                'task_payload': task_data,
                'timestamp_utc': timezone.now().isoformat()
            }

            # Преобразуем словарь в JSON-строку
            json_output = json.dumps(task_dict, indent=4, ensure_ascii=False)

            # Выводим результат в консоль
            print("----------- НОВАЯ ЗАДАЧА -----------")
            print(json_output)
            print("---------------------------------")

            messages.success(request, f"Задача с ID {task_id} успешно отправлена на обработку!")
            return redirect('user_dashboard')
    else:
        form = TaskSubmissionForm()

    return render(request, 'accounts/submit_task.html', {'form': form})