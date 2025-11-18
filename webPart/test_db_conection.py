#!/usr/bin/env python3
"""
Скрипт для тестирования подключения к PostgreSQL
Использование: python test_db_connection.py
"""

import sys

def test_psycopg2_import():
    """Проверка установки psycopg2"""
    print("1. Проверка импорта psycopg2...")
    try:
        import psycopg2
        print("   ✅ psycopg2 установлен")
        print(f"   Версия: {psycopg2.__version__}")
        return True
    except ImportError as e:
        print(f"   ❌ Ошибка: {e}")
        print("   Установите: pip install psycopg2-binary")
        return False


def test_connection(host='localhost', port='5432', dbname='python_tasks_db', 
                   user='postgres', password=None):
    """Тестирование подключения к базе данных"""
    import psycopg2
    
    if password is None:
        password = input("Введите пароль для пользователя postgres: ")
    
    print("\n2. Тестирование подключения к PostgreSQL...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Database: {dbname}")
    print(f"   User: {user}")
    
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        print("   ✅ Подключение установлено успешно!")
        
        # Проверяем версию PostgreSQL
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"   PostgreSQL версия: {version.split(',')[0]}")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.OperationalError as e:
        print(f"   ❌ Ошибка подключения: {e}")
        print("\n   Возможные причины:")
        print("   - PostgreSQL не запущен")
        print("   - Неверный пароль")
        print("   - База данных не существует")
        print("   - Неверный host или port")
        return False
    except Exception as e:
        print(f"   ❌ Неожиданная ошибка: {e}")
        return False


def test_tables(host='localhost', port='5432', dbname='python_tasks_db',
               user='postgres', password=None):
    """Проверка наличия таблиц"""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    print("\n3. Проверка наличия таблиц...")
    
    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname,
            user=user, password=password
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Проверяем таблицы
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        
        if tables:
            print(f"   ✅ Найдено таблиц: {len(tables)}")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table['table_name']}")
                count = cursor.fetchone()['count']
                print(f"      - {table['table_name']}: {count} записей")
        else:
            print("   ⚠️  Таблицы не найдены. Запустите database_setup.sql")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return False


def test_insert(host='localhost', port='5432', dbname='python_tasks_db',
               user='postgres', password=None):
    """Тестирование вставки данных"""
    import psycopg2
    import uuid
    from datetime import datetime
    
    print("\n4. Тест вставки данных...")
    
    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname,
            user=user, password=password
        )
        cursor = conn.cursor()
        
        # Пробуем вставить тестовую задачу
        test_task_id = str(uuid.uuid4())
        test_code = "print('Test task')"
        
        cursor.execute("""
            INSERT INTO tasks 
                (task_id, username, email, user_id_in_app, user_type,
                 task_payload, file_path, original_filename, python_file, timestamp_utc)
            VALUES 
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            test_task_id,
            'test_user',
            'test@example.com',
            999,
            'ordinary',
            f'{test_task_id}_test.py',
            f'/test/{test_task_id}_test.py',
            'test.py',
            psycopg2.Binary(test_code.encode('utf-8')),
            datetime.now()
        ))
        
        task_id = cursor.fetchone()[0]
        conn.commit()
        
        print(f"   ✅ Тестовая задача вставлена (ID: {task_id})")
        
        # Удаляем тестовую задачу
        cursor.execute("DELETE FROM tasks WHERE id = %s", (task_id,))
        conn.commit()
        print("   ✅ Тестовая задача удалена")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return False


def main():
    """Главная функция"""
    print("=" * 60)
    print("ТЕСТ ПОДКЛЮЧЕНИЯ К POSTGRESQL")
    print("=" * 60)
    
    # Тест 1: Импорт
    if not test_psycopg2_import():
        sys.exit(1)
    
    # Запрашиваем параметры подключения
    print("\n" + "=" * 60)
    print("Параметры подключения:")
    print("(нажмите Enter для значения по умолчанию)")
    print("=" * 60)
    
    host = input("Host [localhost]: ").strip() or 'localhost'
    port = input("Port [5432]: ").strip() or '5432'
    dbname = input("Database [python_tasks_db]: ").strip() or 'python_tasks_db'
    user = input("User [postgres]: ").strip() or 'postgres'
    password = input("Password: ").strip()
    
    if not password:
        print("❌ Пароль обязателен!")
        sys.exit(1)
    
    # Тест 2: Подключение
    if not test_connection(host, port, dbname, user, password):
        sys.exit(1)
    
    # Тест 3: Таблицы
    test_tables(host, port, dbname, user, password)
    
    # Тест 4: Вставка
    answer = input("\nВыполнить тест вставки данных? (y/n) [n]: ").strip().lower()
    if answer == 'y':
        test_insert(host, port, dbname, user, password)
    
    print("\n" + "=" * 60)
    print("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 60)
    print("\nБаза данных готова к использованию!")
    print("\nСледующие шаги:")
    print("1. Обновите settings.py с правильными параметрами подключения")
    print("2. Установите переменную окружения DB_PASSWORD")
    print("3. Запустите Django сервер: python manage.py runserver")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Прервано пользователем")
        sys.exit(1)