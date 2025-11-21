"""
HTTP клиент для взаимодействия Django с FastAPI бэкендом
"""
import httpx
from typing import Optional, Dict, Any, List
from django.conf import settings


class APIClient:
    """Клиент для взаимодействия с FastAPI бэкендом"""

    def __init__(self):
        self.base_url = getattr(settings, 'FASTAPI_BASE_URL', 'http://localhost:8000')
        self.timeout = 30.0

    def _get_client(self):
        """Создает httpx клиент с таймаутом"""
        return httpx.Client(timeout=self.timeout)

    # ==================== USER ENDPOINTS ====================

    def register_user(self, username: str, email: str, password: str, **kwargs) -> Dict[str, Any]:
        """Регистрация обычного пользователя"""
        with self._get_client() as client:
            payload = {
                "username": username,
                "email": email,
                "password": password,
                "bio": kwargs.get('bio'),
                "birth_date": kwargs.get('birth_date').isoformat() if kwargs.get('birth_date') else None,
                "phone": kwargs.get('phone'),
                "city": kwargs.get('city'),
                "subscription_active": kwargs.get('subscription_active', False)
            }
            response = client.post(f"{self.base_url}/users/register", json=payload)
            response.raise_for_status()
            return response.json()

    def register_admin(self, username: str, email: str, password: str, **kwargs) -> Dict[str, Any]:
        """Регистрация администратора"""
        with self._get_client() as client:
            payload = {
                "username": username,
                "email": email,
                "password": password,
                "department": kwargs.get('department'),
                "phone": kwargs.get('phone'),
                "permissions_level": kwargs.get('permissions_level', 1),
                "access_code": kwargs.get('access_code', 'default')
            }
            response = client.post(f"{self.base_url}/admins/register", json=payload)
            response.raise_for_status()
            return response.json()

    def get_auth_info(self, username: str) -> Dict[str, Any]:
        """Получение информации для аутентификации"""
        with self._get_client() as client:
            response = client.get(f"{self.base_url}/auth/{username}")
            response.raise_for_status()
            return response.json()

    def get_user_full(self, user_id: int) -> Dict[str, Any]:
        """Получение полной информации о пользователе"""
        with self._get_client() as client:
            response = client.get(f"{self.base_url}/users/{user_id}")
            response.raise_for_status()
            return response.json()

    def update_user(self, user_id: int, **kwargs) -> Dict[str, Any]:
        """Обновление информации о пользователе"""
        with self._get_client() as client:
            # Фильтруем только непустые значения
            payload = {k: v for k, v in kwargs.items() if v is not None}
            # Конвертируем даты
            if 'birth_date' in payload and payload['birth_date']:
                payload['birth_date'] = payload['birth_date'].isoformat()

            response = client.put(f"{self.base_url}/users/{user_id}", json=payload)
            response.raise_for_status()
            return response.json()

    def upload_avatar(self, user_id: int, file) -> Dict[str, Any]:
        """Загрузка аватара пользователя"""
        with self._get_client() as client:
            files = {'file': (file.name, file.read(), file.content_type)}
            response = client.put(f"{self.base_url}/users/{user_id}/avatar", files=files)
            response.raise_for_status()
            return response.json()

    def get_avatar_url(self, user_id: int) -> str:
        """Получение URL аватара"""
        return f"{self.base_url}/users/{user_id}/avatar"

    def delete_avatar(self, user_id: int) -> Dict[str, Any]:
        """Удаление аватара"""
        with self._get_client() as client:
            response = client.delete(f"{self.base_url}/users/{user_id}/avatar")
            response.raise_for_status()
            return response.json()

    def delete_user(self, user_id: int) -> Dict[str, Any]:
        """Удаление пользователя"""
        with self._get_client() as client:
            response = client.delete(f"{self.base_url}/users/{user_id}")
            response.raise_for_status()
            return response.json()

    # ==================== TASK ENDPOINTS ====================

    def list_user_tasks(self, user_id: int) -> List[Dict[str, Any]]:
        """Получение списка задач пользователя"""
        with self._get_client() as client:
            response = client.get(f"{self.base_url}/tasks/user/{user_id}")
            response.raise_for_status()
            return response.json()

    def create_task(self, user_id: int, name: str, description: Optional[str] = None,
                    files: Optional[List] = None) -> Dict[str, Any]:
        """Создание новой задачи"""
        with self._get_client() as client:
            data = {
                "user_id": user_id,
                "name": name,
                "description": description
            }

            files_to_upload = None
            if files:
                files_to_upload = [
                    ('files', (f.name, f.read(), f.content_type or 'application/octet-stream'))
                    for f in files
                ]

            response = client.post(
                f"{self.base_url}/tasks/",
                data=data,
                files=files_to_upload
            )
            response.raise_for_status()
            return response.json()

    def start_task(self, task_id: int) -> Dict[str, Any]:
        """Запуск задачи"""
        with self._get_client() as client:
            response = client.post(f"{self.base_url}/tasks/{task_id}/start")
            response.raise_for_status()
            return response.json()

    def complete_task(self, task_id: int, result_files: Optional[List] = None) -> Dict[str, Any]:
        """Завершение задачи с результатами"""
        with self._get_client() as client:
            files_to_upload = None
            if result_files:
                files_to_upload = [
                    ('result_files', (f.name, f.read(), f.content_type or 'application/octet-stream'))
                    for f in result_files
                ]

            response = client.post(
                f"{self.base_url}/tasks/{task_id}/complete",
                files=files_to_upload
            )
            response.raise_for_status()
            return response.json()

    def delete_task(self, task_id: int) -> None:
        """Удаление задачи"""
        with self._get_client() as client:
            response = client.delete(f"{self.base_url}/tasks/{task_id}")
            response.raise_for_status()

    def get_input_files(self, task_id: int) -> List[Dict[str, Any]]:
        """Получение списка входных файлов задачи"""
        with self._get_client() as client:
            response = client.get(f"{self.base_url}/tasks/{task_id}/input-files")
            response.raise_for_status()
            return response.json()

    def get_result_files(self, task_id: int) -> List[Dict[str, Any]]:
        """Получение списка результирующих файлов задачи"""
        with self._get_client() as client:
            response = client.get(f"{self.base_url}/tasks/{task_id}/result-files")
            response.raise_for_status()
            return response.json()

    def download_input_file(self, task_id: int, file_id: int) -> bytes:
        """Скачивание входного файла"""
        with self._get_client() as client:
            response = client.get(f"{self.base_url}/tasks/{task_id}/input-files/{file_id}")
            response.raise_for_status()
            return response.content

    def download_result_file(self, task_id: int, file_id: int) -> bytes:
        """Скачивание результирующего файла"""
        with self._get_client() as client:
            response = client.get(f"{self.base_url}/tasks/{task_id}/result-files/{file_id}")
            response.raise_for_status()
            return response.content

    # ==================== HEALTH CHECK ====================

    def health_check(self) -> Dict[str, Any]:
        """Проверка доступности API"""
        with self._get_client() as client:
            response = client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()


# Singleton instance
api_client = APIClient()
