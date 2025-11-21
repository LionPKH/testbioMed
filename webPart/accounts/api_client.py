import httpx
from django.conf import settings

# Если настройки URL нет в settings, используем дефолт
API_BASE_URL = getattr(settings, 'API_URL', 'http://localhost:8000')

class APIClient:
    def __init__(self):
        self.base_url = API_BASE_URL

    async def _post(self, endpoint, data=None, files=None):
        async with httpx.AsyncClient() as client:
            # Если есть файлы, хедеры не ставим (httpx сам поставит multipart)
            if files:
                response = await client.post(f"{self.base_url}{endpoint}", data=data, files=files)
            else:
                response = await client.post(f"{self.base_url}{endpoint}", json=data)
            
            response.raise_for_status()
            
            # --- ВОТ ЗДЕСЬ БЫЛА ОШИБКА ---
            # НЕПРАВИЛЬНО: return await response.json()
            # ПРАВИЛЬНО:
            return response.json() 

    async def _get(self, endpoint, params=None):
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            # Тоже без await
            return response.json()

    async def _put(self, endpoint, data):
        async with httpx.AsyncClient() as client:
            response = await client.put(f"{self.base_url}{endpoint}", json=data)
            response.raise_for_status()
            return response.json()

    # --- Методы регистрации ---

    async def register_user(self, username, email, password):
        payload = {
            "username": username,
            "email": email,
            "password": password,
            "user_type": "ordinary"
        }
        # Адрес в API должен совпадать с тем, что в логах (users/register)
        return await self._post("/users/register", data=payload)

    async def register_admin(self, username, email, password):
        payload = {
            "username": username,
            "email": email,
            "password": password,
            "user_type": "admin"
        }
        return await self._post("/users/register", data=payload)

    async def get_auth_info(self, username):
        return await self._get(f"/users/auth/{username}")

    async def get_user_full(self, user_id):
        return await self._get(f"/users/{user_id}")
        
    async def update_user(self, user_id, data):
        return await self._put(f"/users/{user_id}", data)

    # --- Методы задач ---

    async def create_task(self, user_id, name, description, files):
        # files передаем как словарь или список кортежей, ожидаемых httpx
        # data - это текстовые поля
        data = {
            "user_id": user_id,
            "name": name,
            "description": description
        }
        # В httpx файлы передаются аргументом files
        return await self._post("/tasks/", data=data, files=files)

    async def list_user_tasks(self, user_id):
        return await self._get(f"/tasks/user/{user_id}")

    async def get_task_input_files(self, task_id):
        # Пример, если API отдает список
        return await self._get(f"/tasks/{task_id}/input_files")

    async def get_task_result_files(self, task_id):
        return await self._get(f"/tasks/{task_id}/result_files")


api_client = APIClient()