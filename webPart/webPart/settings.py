"""
Django settings for webPart project - API Integration Version

Этот файл настроен для работы только через FastAPI бэкенд.
Django НЕ работает напрямую с PostgreSQL - все через HTTP API.
"""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================
# FASTAPI BACKEND CONFIGURATION
# ============================================
FASTAPI_BASE_URL = os.environ.get('FASTAPI_BASE_URL', 'http://localhost:8000')

# ============================================
# SECURITY
# ============================================
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-r_^y4--2_i9=%ygt93*c3ss(pq!^6nft8g3hrq5!v=1q*s-2y1')
DEBUG = os.environ.get('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', '*').split(',')

# ============================================
# APPLICATIONS
# ============================================
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',  # Для хранения сессий
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # 'accounts' больше не нужен - используем только views
    'accounts',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',  # Обязательно
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',

    # --- ВОТ ЭТОЙ СТРОКИ НЕ ХВАТАЕТ ---
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    # ----------------------------------

    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'webPart.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],  # Или ваш путь к шаблонам
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',

                # --- ВОТ ЭТОЙ СТРОКИ НЕ ХВАТАЕТ ---
                'django.contrib.auth.context_processors.auth',
                # ----------------------------------

                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'webPart.wsgi.application'

# ============================================
# DATABASE - ТОЛЬКО ДЛЯ DJANGO СЕССИЙ
# ============================================
# Django использует SQLite ТОЛЬКО для своих внутренних нужд (сессии, admin)
# Все бизнес-данные хранятся в PostgreSQL через FastAPI
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'django_sessions.db',  # Только для сессий!
    }
}

# ============================================
# SESSION CONFIGURATION
# ============================================
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
SESSION_COOKIE_AGE = 86400  # 24 часа
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'

# ============================================
# PASSWORD VALIDATION (не используется, но оставляем)
# ============================================
AUTH_PASSWORD_VALIDATORS = []  # Валидация будет на стороне FastAPI

# ============================================
# INTERNATIONALIZATION
# ============================================
LANGUAGE_CODE = 'ru-RU'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# ============================================
# STATIC FILES
# ============================================
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_DIRS = [BASE_DIR / 'static'] if (BASE_DIR / 'static').exists() else []

# ============================================
# MEDIA FILES (не используются - все через API)
# ============================================
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# ============================================
# AUTHENTICATION (отключено - используем сессии)
# ============================================
# AUTH_USER_MODEL не используется
LOGIN_URL = 'login'
LOGIN_REDIRECT_URL = 'user_dashboard'
LOGOUT_REDIRECT_URL = 'login'

# ============================================
# DEFAULT SETTINGS
# ============================================
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ============================================
# FILE UPLOAD
# ============================================
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10 MB

# ============================================
# LOGGING
# ============================================
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# ============================================
# HTTPX CLIENT CONFIGURATION
# ============================================
HTTPX_TIMEOUT = 30.0
HTTPX_MAX_CONNECTIONS = 100
