import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
STORAGE_ROOT = os.getenv("STORAGE_ROOT", "./data")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

os.makedirs(STORAGE_ROOT, exist_ok=True)
