# webPart/app_name/services.py
import requests

API_BASE_URL = "http://localhost:8000"  # Адрес твоего API из папки src

def get_all_records():
    """Получение списка записей из API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/records/")
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []

def create_record(data):
    """Отправка данных в API для записи в БД"""
    response = requests.post(f"{API_BASE_URL}/api/records/", json=data)
    return response