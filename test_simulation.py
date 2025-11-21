import time
import json
import uuid


# --- 1. MOCK (ИМИТАЦИЯ) ВНЕШНИХ СИСТЕМ ---

class MockDatabaseHandler:
    """Имитация базы данных. Просто печатает результат."""

    def save_result(self, task_id, combined_data, status="completed"):
        print("\n" + "=" * 50)
        print(f"[MOCK DB] УСПЕХ! Запись в БД для задачи {task_id}")
        print(f"[MOCK DB] Статус: {status}")
        print(f"[MOCK DB] Данные: {json.dumps(combined_data, indent=2, ensure_ascii=False)}")
        print("=" * 50 + "\n")


class MockMasterTransport:
    """
    Имитация транспорта.
    Вместо MQTT просто складывает сообщения во внутренний список.
    """

    def __init__(self):
        self.fake_buffer = {}  # Буфер ответов от "воркеров"

    def start(self):
        return True

    def set_dead_nodes_callback(self, cb):
        pass  # В тесте нам это не нужно

    def get_nodes_by_status(self, status):
        # Притворяемся, что у нас есть 3 активных ESP32
        return [
            {'node_id': 'sim_worker_01', 'status': 'ready'},
            {'node_id': 'sim_worker_02', 'status': 'ready'},
            {'node_id': 'sim_worker_03', 'status': 'ready'}
        ]

    def send_task(self, node_id, payload, priority=5):
        """
        Самая важная часть симуляции.
        Когда мастер отправляет задачу, этот метод СРАЗУ ЖЕ генерирует ответ,
        как будто ESP32 мгновенно все посчитала.
        """
        print(f"[Transport] -> Отправка на {node_id}: Subtask {payload['subtask_index']}")

        # 1. Генерируем "фейковый" результат работы ESP32
        fake_result_payload = {
            "task_id": payload['task_id'],
            "subtask_index": payload['subtask_index'],
            # Добавим какую-то "полезную нагрузку", чтобы проверить склейку
            "data": f"Processed part {payload['subtask_index']} by {node_id}"
        }

        # 2. Кладем этот ответ в буфер, как будто он пришел по MQTT
        if node_id not in self.fake_buffer:
            self.fake_buffer[node_id] = []

        # Добавляем обертку {'result': payload}, как это делает ваш реальный транспорт
        self.fake_buffer[node_id].append({'result': fake_result_payload})

    def get_results(self, disposable=True):
        """Мастер вызывает это, чтобы забрать ответы"""
        if not self.fake_buffer:
            return {}

        # Возвращаем накопленные ответы
        current_data = self.fake_buffer.copy()
        if disposable:
            self.fake_buffer = {}
        return current_data


# --- 2. ЛОГИКА МАСТЕРА (Копия того, что мы писали ранее, но адаптированная) ---
# Для теста я вставил сюда урезанные версии классов Splitter и Executor.

class TaskSplitter:
    def analyze_task_complexity(self, code):
        # Для теста всегда говорим, что задача сложная и нужно делить
        return {'needs_splitting': True}

    def split_complex_task(self, code, data):
        # Делим задачу на 3 части
        return [
            {'program_code': code, 'task_data': data},
            {'program_code': code, 'task_data': data},
            {'program_code': code, 'task_data': data}
        ]


def process_worker_responses(transport_api, active_tasks_registry, db_handler):
    """
    Та самая функция объединения, которую мы писали.
    """
    nodes_results = transport_api.get_results(disposable=True)

    if not nodes_results:
        return

    for node_id, messages in nodes_results.items():
        for msg in messages:
            payload = msg.get('result')
            task_id = payload.get("task_id")

            if not task_id or task_id not in active_tasks_registry:
                continue

            task_info = active_tasks_registry[task_id]
            subtask_idx = payload.get("subtask_index", 0)
            worker_data = payload.get("data")

            print(f"[Logic] Принят ответ части {subtask_idx} от {node_id}")
            task_info['received_parts'][subtask_idx] = worker_data

            # Проверка завершения
            if len(task_info['received_parts']) == task_info['total_parts']:
                print(f"[Logic] ВСЕ ЧАСТИ СОБРАНЫ! Склеиваем...")

                sorted_parts = []
                for i in range(task_info['total_parts']):
                    sorted_parts.append(task_info['received_parts'].get(i))

                final_result = {
                    "task_id": task_id,
                    "aggregated_output": sorted_parts
                }

                # Сохраняем в MOCK DB
                db_handler.save_result(task_id, final_result)
                del active_tasks_registry[task_id]


class BioMedExecutorTest:
    def __init__(self):
        # ПОДМЕНА: Используем Mock классы
        self.transport = MockMasterTransport()
        self.db = MockDatabaseHandler()
        self.splitter = TaskSplitter()
        self.active_tasks = {}

    def submit_task(self, program_code):
        task_id = str(uuid.uuid4())
        # Разбиваем (в нашем Mock Splitter всегда на 3 части)
        subtasks = self.splitter.split_complex_task(program_code, {})

        self.active_tasks[task_id] = {
            'total_parts': len(subtasks),
            'received_parts': {},
            'created_at': time.time()
        }

        # Распределение
        nodes = self.transport.get_nodes_by_status("ready")
        node_ids = [n['node_id'] for n in nodes]

        print(f"\n[Test] Распределение задачи {task_id} на {len(subtasks)} части...")

        for i, subtask in enumerate(subtasks):
            target = node_ids[i % len(node_ids)]
            payload = {
                "task_id": task_id,
                "subtask_index": i,
                "data": {}
            }
            # Это вызовет метод send_task в MockMasterTransport,
            # который тут же сгенерирует ответ
            self.transport.send_task(target, payload)

    def process_cycle(self):
        # Один цикл проверки ответов
        process_worker_responses(self.transport, self.active_tasks, self.db)


# --- 3. ЗАПУСК ТЕСТА ---

if __name__ == "__main__":
    print("--- ЗАПУСК ТЕСТОВОЙ СИМУЛЯЦИИ ---")

    executor = BioMedExecutorTest()

    # 1. Отправляем задачу
    # Внутри MockTransport.send_task сразу сгенерируются ответы
    executor.submit_task("print('Hello World')")

    print("\n[Test] Имитация ожидания (sleep)...")
    time.sleep(1)

    # 2. Запускаем обработку ответов
    # Функция process_worker_responses заберет сгенерированные ответы и соберет их
    print("\n[Test] Сбор результатов...")
    executor.process_cycle()

    print("--- ТЕСТ ЗАВЕРШЕН ---")