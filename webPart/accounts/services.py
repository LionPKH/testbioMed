import requests
import json
from django.conf import settings

# Укажите здесь адрес вашего API (папка src)
API_BASE_URL = getattr(settings, 'API_URL', 'http://localhost:8000/api/')


def api_create_task(task_data):
    """
    Отправляет задачу в API для сохранения в БД и обработки.
    task_data: словарь с данными задачи и кодом файла
    """
    try:
        # Отправляем POST запрос в API
        response = requests.post(f"{API_BASE_URL}/tasks/", json=task_data)

        # Если API вернул успешный код (200 или 201)
        if response.status_code in [200, 201]:
            return True
        else:
            print(f"API Error ({response.status_code}): {response.text}")
            return False
    except requests.RequestException as e:
        print(f"Connection Error: {e}")
        return False


def api_get_user_tasks(user_id, limit=10, page=1):
    """Получает список задач пользователя из API"""
    try:
        params = {'user_id': user_id, 'limit': limit, 'page': page}
        response = requests.get(f"{API_BASE_URL}/tasks/user/{user_id}", params=params)

        if response.status_code == 200:
            return response.json()  # Ожидаем список словарей
        return []
    except requests.RequestException:
        return []


def api_get_task_detail(task_id, user_id):
    """Получает детали задачи из API"""
    try:
        response = requests.get(f"{API_BASE_URL}/tasks/{task_id}", params={'user_id': user_id})
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


def api_get_statistics(user_id):
    """Получает статистику пользователя из API"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats/{user_id}")
        if response.status_code == 200:
            return response.json()
        return {}
    except requests.RequestException:
        return {}