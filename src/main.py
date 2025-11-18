from fastapi import FastAPI
from api import user as user_router
from api import tasks as tasks_api

app = FastAPI(title="Storage User API (async)")
app.include_router(user_router.router)
app.include_router(tasks_api.router)
