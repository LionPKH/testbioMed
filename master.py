import redis
import json
import time
import uuid
import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
import hashlib
import sys
import json
import time
import uuid
# Импорт вашего транспорта
from transport.master_transport import MasterNodeTransport


# --- НАСТРОЙКИ СУЩЕСТВУЮЩЕЙ БД ---
class DatabaseHandler:
    def __init__(self):
        # TODO: Укажите параметры вашей существующей базы данных
        self.config = {
            "dbname": "testDB",  # Имя вашей БД
            "user": "petrk",  # Ваш пользователь
            "password": "123",  # Ваш пароль
            "host": "localhost",
            "port": 5432
        }

    def get_connection(self):
        """Возвращает соединение с БД. Замените библиотеку на нужную (psycopg2, mysql и т.д.)"""
        import sqlite3
        import psycopg2  # Если используете Postgres

        # Пример для SQLite (файловая БД):
        # conn = sqlite3.connect("my_existing_database.db")

        # Пример для PostgreSQL:
        conn = psycopg2.connect(**self.config)

        return conn

    def save_result(self, task_id, combined_data, status="completed"):
        """
        Метод сохранения результата в СУЩЕСТВУЮЩУЮ таблицу.
        Вам нужно проверить название таблицы и колонок.
        """
        conn = self.get_connection()
        try:
            cur = conn.cursor()

            # Сериализуем данные в JSON строку перед записью
            data_json = json.dumps(combined_data, ensure_ascii=False)

            # TODO: Проверьте название таблицы (results?) и колонок
            query = """
                INSERT INTO results (task_id, data, status, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """
            # Для Postgres используйте %s вместо ?

            cur.execute(query, (task_id, data_json, status))
            conn.commit()
            print(f"[DB] Данные задачи {task_id} сохранены в существующую БД.")

        except Exception as e:
            print(f"[DB Error] Ошибка записи в БД: {e}")
            conn.rollback()
        finally:
            conn.close()


def process_worker_responses(transport_api, active_tasks_registry):
    """
    Считывает ответы, объединяет подзадачи и сохраняет в БД.

    :param transport_api: экземпляр MasterNodeTransport
    :param active_tasks_registry: словарь active_tasks из класса Master
    """

    # 1. Забираем результаты из транспорта и очищаем буфер (disposable=True)
    # Возвращает dict: { 'node_id': [ { 'result': ..., 'timestamp': ... } ] }
    nodes_results = transport_api.get_results(disposable=True)

    if not nodes_results:
        return

    for node_id, messages in nodes_results.items():
        for msg in messages:
            try:
                # В 'result' лежит payload, который отправила ESP32
                payload = msg.get('result')

                # Если payload пришел строкой, парсим JSON
                if isinstance(payload, str):
                    payload = json.loads(payload)

                # Ожидаем структуру: {"task_id": "...", "subtask_index": 0, "data": "..."}
                task_id = payload.get("task_id")

                if not task_id or task_id not in active_tasks_registry:
                    # Либо мусор, либо задача уже выполнена/отменена
                    continue

                task_info = active_tasks_registry[task_id]
                subtask_idx = payload.get("subtask_index", 0)
                worker_data = payload.get("data") or payload.get("output")

                # Сохраняем часть результата
                print(f"[Logic] Пришла часть {subtask_idx} для {task_id} от {node_id}")
                task_info['received_parts'][subtask_idx] = worker_data

                # 2. Проверка: все ли части собраны?
                if len(task_info['received_parts']) == task_info['total_parts']:
                    print(f"[Logic] Все части задачи {task_id} собраны. Объединение...")

                    # Сортируем части по индексу, чтобы данные не перемешались
                    sorted_parts = []
                    for i in range(task_info['total_parts']):
                        part_data = task_info['received_parts'].get(i)
                        sorted_parts.append(part_data)

                    # Формируем итоговый объект
                    final_result = {
                        "task_id": task_id,
                        "original_params": task_info['params'],
                        "aggregated_output": sorted_parts
                    }

                    # 3. Сохранение в СУЩЕСТВУЮЩУЮ БД
                    db_handler.save_result(task_id, final_result)

                    # Удаляем задачу из активных, так как она завершена
                    del active_tasks_registry[task_id]

            except Exception as e:
                print(f"[Error] Ошибка обработки ответа от {node_id}: {e}")

# Инициализация хендлера БД
db_handler = DatabaseHandler()
class TaskComplexityAnalyzer:
    """Анализатор сложности задач для предсказания времени выполнения"""
    
    def __init__(self):
        self.program_features = {}
        self.execution_history = defaultdict(list)
        self.feature_weights = {
            'code_length': 0.25,
            'import_complexity': 0.15,
            'function_count': 0.15,
            'loop_depth': 0.2,
            'memory_operations': 0.15,
            'data_structure_complexity': 0.1
        }
    
    def extract_program_features(self, program_code: str) -> Dict:
        """Извлечение характеристик программы для анализа сложности"""
        features = {
            'code_length': len(program_code),
            'import_complexity': self._calculate_import_complexity(program_code),
            'function_count': program_code.count('def '),
            'class_count': program_code.count('class '),
            'loop_depth': self._estimate_loop_depth(program_code),
            'memory_operations': self._count_memory_operations(program_code),
            'library_usage': self._detect_library_usage(program_code),
            'data_structures': self._detect_data_structures(program_code),
            'recursive_calls': program_code.count('def ') and program_code.count('(') > program_code.count('def ') * 2,
            'nested_loops': self._count_nested_loops(program_code)
        }
        
        # Создаем хеш программы для идентификации похожих программ
        program_hash = hashlib.md5(program_code.encode()).hexdigest()
        features['program_hash'] = program_hash
        
        return features
    
    def _calculate_import_complexity(self, code: str) -> int:
        """Оценка сложности импортов"""
        import_lines = [line for line in code.split('\n') if line.strip().startswith('import') or line.strip().startswith('from')]
        complexity = 0
        
        for line in import_lines:
            if 'from' in line and 'import' in line:
                complexity += 2  # Более сложные импорты
            else:
                complexity += 1
        
        return complexity
    
    def _estimate_loop_depth(self, code: str) -> int:
        """Оценка глубины циклов"""
        lines = code.split('\n')
        max_depth = 0
        current_depth = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('for ', 'while ', 'async for ')):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif stripped and not stripped.startswith(('#', '"', "'")) and current_depth > 0:
                # Уменьшаем глубину когда встречаем отступ меньше
                if len(line) - len(line.lstrip()) < current_depth * 4:
                    current_depth -= 1
        
        return max_depth
    
    def _count_nested_loops(self, code: str) -> int:
        """Подсчет вложенных циклов"""
        lines = code.split('\n')
        nested_count = 0
        current_depth = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('for ', 'while ')):
                if current_depth > 0:
                    nested_count += current_depth
                current_depth += 1
            elif stripped and not stripped.startswith(('#', '"', "'")) and current_depth > 0:
                if len(line) - len(line.lstrip()) < current_depth * 4:
                    current_depth -= 1
        
        return nested_count
    
    def _count_memory_operations(self, code: str) -> int:
        """Подсчет операций с памятью"""
        memory_ops = 0
        memory_keywords = ['append', 'extend', 'insert', 'pop', 'remove', 
                          'allocate', 'malloc', 'free', 'gc', 'memory',
                          'array', 'list', 'dict', 'set', 'bytearray', 'copy', 'deepcopy']
        
        for keyword in memory_keywords:
            memory_ops += code.count(keyword)
        
        return memory_ops
    
    def _detect_library_usage(self, code: str) -> List[str]:
        """Обнаружение использования библиотек"""
        libraries = []
        common_libs = ['numpy', 'pandas', 'tensorflow', 'torch', 'sklearn', 
                      'matplotlib', 'requests', 'sqlalchemy', 'django', 'multiprocessing']
        
        for lib in common_libs:
            if lib in code.lower():
                libraries.append(lib)
        
        return libraries
    
    def _detect_data_structures(self, code: str) -> Dict[str, int]:
        """Обнаружение структур данных"""
        structures = {
            'lists': code.count('list(') + code.count('[]'),
            'dicts': code.count('dict(') + code.count('{}'),
            'sets': code.count('set(') + code.count('{') - code.count('{}'),
            'arrays': code.count('array(') + code.count('np.array'),
            'dataframes': code.count('DataFrame') + code.count('pd.DataFrame'),
            'queues': code.count('Queue') + code.count('queue'),
            'stacks': code.count('deque') + code.count('LifoQueue')
        }
        return structures
    
    def predict_execution_time(self, program_code: str, historical_data: List[float] = None) -> float:
        """Предсказание времени выполнения программы"""
        features = self.extract_program_features(program_code)
        
        # Базовое предсказание на основе характеристик
        base_prediction = self._calculate_base_prediction(features)
        
        # Корректировка на основе исторических данных
        if historical_data:
            historical_adjustment = self._calculate_historical_adjustment(historical_data)
            base_prediction *= historical_adjustment
        
        # Корректировка на основе библиотек
        library_adjustment = self._calculate_library_adjustment(features.get('library_usage', []))
        base_prediction *= library_adjustment
        
        # Корректировка на рекурсию и вложенные циклы
        complexity_adjustment = self._calculate_complexity_adjustment(features)
        base_prediction *= complexity_adjustment
        
        return max(1.0, base_prediction)  # Минимум 1 секунда
    
    def _calculate_base_prediction(self, features: Dict) -> float:
        """Базовое предсказание на основе характеристик"""
        prediction = 0
        
        # Взвешенная сумма характеристик
        prediction += features['code_length'] * 0.001 * self.feature_weights['code_length']
        prediction += features['import_complexity'] * 0.5 * self.feature_weights['import_complexity']
        prediction += features['function_count'] * 2.0 * self.feature_weights['function_count']
        prediction += features['loop_depth'] * 3.0 * self.feature_weights['loop_depth']
        prediction += features['memory_operations'] * 0.8 * self.feature_weights['memory_operations']
        
        # Корректировка на структуры данных
        structures = features.get('data_structures', {})
        data_structure_complexity = (
            structures.get('dataframes', 0) * 5.0 +
            structures.get('arrays', 0) * 2.0 +
            structures.get('queues', 0) * 1.5 +
            structures.get('stacks', 0) * 1.2
        )
        prediction += data_structure_complexity * self.feature_weights['data_structure_complexity']
        
        return prediction
    
    def _calculate_complexity_adjustment(self, features: Dict) -> float:
        """Корректировка на сложность алгоритмов"""
        adjustment = 1.0
        
        # Рекурсия увеличивает время
        if features.get('recursive_calls', False):
            adjustment *= 1.8
        
        # Вложенные циклы значительно увеличивают время
        nested_loops = features.get('nested_loops', 0)
        if nested_loops > 0:
            adjustment *= (1.0 + nested_loops * 0.3)
        
        return adjustment
    
    def _calculate_historical_adjustment(self, historical_data: List[float]) -> float:
        """Корректировка на основе исторических данных"""
        if not historical_data:
            return 1.0
        
        # Используем медиану для устойчивости к выбросам
        median_time = np.median(historical_data)
        expected_time = np.mean(historical_data)
        
        if median_time > 0:
            return expected_time / median_time
        return 1.0
    
    def _calculate_library_adjustment(self, libraries: List[str]) -> float:
        """Корректировка на основе используемых библиотек"""
        adjustment = 1.0
        library_factors = {
            'numpy': 1.1,
            'pandas': 1.3,
            'tensorflow': 2.0,
            'torch': 1.8,
            'sklearn': 1.2,
            'matplotlib': 1.1,
            'requests': 1.0,
            'sqlalchemy': 1.4,
            'django': 1.5,
            'multiprocessing': 1.3
        }
        
        for lib in libraries:
            adjustment *= library_factors.get(lib, 1.0)
        
        return adjustment
    
    def record_execution_time(self, program_hash: str, execution_time: float):
        """Запись времени выполнения для будущих предсказаний"""
        if len(self.execution_history[program_hash]) > 100:  # Ограничиваем историю
            self.execution_history[program_hash].pop(0)
        self.execution_history[program_hash].append(execution_time)

class TaskSplitter:
    """Класс для разбивки сложных задач на подзадачи"""
    
    def __init__(self):
        self.split_strategies = {
            'data_processing': self._split_data_processing,
            'machine_learning': self._split_machine_learning,
            'matrix_operations': self._split_matrix_operations,
            'file_processing': self._split_file_processing
        }
    
    def analyze_task_complexity(self, program_code: str) -> Dict:
        """Анализ сложности задачи для определения необходимости разбивки"""
        features = {
            'nested_loops': self._count_nested_loops(program_code),
            'data_volume': self._estimate_data_volume(program_code),
            'independent_sections': self._find_independent_sections(program_code),
            'function_calls': program_code.count('def '),
            'loop_iterations': self._estimate_loop_iterations(program_code),
            'memory_intensive': self._detect_memory_intensive_operations(program_code)
        }
        
        # Оценка необходимости разбивки (0-1)
        split_score = (
            min(features['nested_loops'] * 0.2, 0.3) +
            min(features['data_volume'] * 0.001, 0.3) +
            min(features['independent_sections'] * 0.15, 0.2) +
            min(features['memory_intensive'] * 0.1, 0.2)
        )
        
        features['needs_splitting'] = split_score > 0.4
        features['split_score'] = split_score
        
        return features
    
    def split_complex_task(self, program_code: str, task_data: Dict) -> List[Dict]:
        """Разбить сложную задачу на подзадачи"""
        complexity = self.analyze_task_complexity(program_code)
        
        if not complexity['needs_splitting']:
            return [{'program_code': program_code, 'task_data': task_data, 'is_subtask': False}]
        
        # Определяем стратегию разбивки
        strategy = self._determine_split_strategy(program_code)
        subtasks = self.split_strategies[strategy](program_code, task_data)
        
        print(f" Splitting complex task into {len(subtasks)} subtasks using {strategy} strategy")
        
        return subtasks
    
    def _determine_split_strategy(self, program_code: str) -> str:
        """Определить стратегию разбивки на основе анализа кода"""
        if 'DataFrame' in program_code or 'pd.' in program_code:
            return 'data_processing'
        elif 'tensorflow' in program_code or 'torch' in program_code or 'fit(' in program_code:
            return 'machine_learning'
        elif 'for i in range' in program_code and 'for j in range' in program_code:
            return 'matrix_operations'
        elif 'open(' in program_code or 'read()' in program_code:
            return 'file_processing'
        else:
            return 'data_processing'
    
    def _split_data_processing(self, program_code: str, task_data: Dict) -> List[Dict]:
        """Разбивка задач обработки данных"""
        subtasks = []
        
        # Пример: разбивка по обработке разных колонок или чанков данных
        lines = program_code.split('\n')
        data_sections = self._extract_data_sections(lines)
        
        for i, section in enumerate(data_sections):
            subtask_code = self._create_subtask_template(section, i, len(data_sections))
            subtask_data = {
                **task_data,
                'subtask_id': i,
                'total_subtasks': len(data_sections),
                'section': section
            }
            subtasks.append({
                'program_code': subtask_code,
                'task_data': subtask_data,
                'is_subtask': True
            })
        
        return subtasks
    
    def _split_machine_learning(self, program_code: str, task_data: Dict) -> List[Dict]:
        """Разбивка ML задач на этапы"""
        subtasks = []
        
        # Этап 1: Предобработка данных
        preprocess_code = """
# Data preprocessing subtask
def preprocess_data():
    # Extract preprocessing logic from main task
    print("Preprocessing data...")
    # Add actual preprocessing code here
    return "preprocessed_data"

result = preprocess_data()
"""
        subtasks.append({
            'program_code': preprocess_code,
            'task_data': {**task_data, 'stage': 'preprocessing', 'subtask_id': 0},
            'is_subtask': True
        })
        
        # Этап 2: Обучение модели
        training_code = """
# Model training subtask
def train_model():
    print("Training model...")
    # Add actual training code here
    return "trained_model"

result = train_model()
"""
        subtasks.append({
            'program_code': training_code,
            'task_data': {**task_data, 'stage': 'training', 'subtask_id': 1},
            'is_subtask': True
        })
        
        # Этап 3: Валидация
        validation_code = """
# Model validation subtask
def validate_model():
    print("Validating model...")
    # Add actual validation code here
    return "validation_results"

result = validate_model()
"""
        subtasks.append({
            'program_code': validation_code,
            'task_data': {**task_data, 'stage': 'validation', 'subtask_id': 2},
            'is_subtask': True
        })
        
        return subtasks
    
    def _split_matrix_operations(self, program_code: str, task_data: Dict) -> List[Dict]:
        """Разбивка матричных операций на блоки"""
        subtasks = []
        
        # Разбиваем матричные операции на блоки 2x2
        for i in range(2):
            for j in range(2):
                block_code = f"""
# Matrix block operation ({i},{j})
def process_matrix_block():
    print("Processing matrix block ({i},{j})...")
    # Add actual matrix block processing code here
    return "block_result_{i}_{j}"

result = process_matrix_block()
"""
                subtasks.append({
                    'program_code': block_code,
                    'task_data': {**task_data, 'block_x': i, 'block_y': j, 'subtask_id': i*2 + j},
                    'is_subtask': True
                })
        
        return subtasks
    
    def _split_file_processing(self, program_code: str, task_data: Dict) -> List[Dict]:
        """Разбивка обработки файлов на части"""
        subtasks = []
        
        # Разбиваем на 3 части: чтение, обработка, запись
        stages = ['reading', 'processing', 'writing']
        for i, stage in enumerate(stages):
            stage_code = f"""
# File {stage} subtask
def {stage}_files():
    print("File {stage} stage...")
    # Add actual file {stage} code here
    return "{stage}_complete"

result = {stage}_files()
"""
            subtasks.append({
                'program_code': stage_code,
                'task_data': {**task_data, 'stage': stage, 'subtask_id': i},
                'is_subtask': True
            })
        
        return subtasks
    
    def _count_nested_loops(self, code: str) -> int:
        """Подсчет вложенных циклов"""
        lines = code.split('\n')
        max_depth = 0
        current_depth = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('for ', 'while ')):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif stripped and not stripped.startswith(('#', '"', "'")) and current_depth > 0:
                if len(line) - len(line.lstrip()) < current_depth * 4:
                    current_depth -= 1
        
        return max_depth
    
    def _estimate_data_volume(self, code: str) -> int:
        """Оценка объема обрабатываемых данных"""
        data_indicators = [
            code.count('range(100'),
            code.count('range(500'),
            code.count('range(1000'),
            code.count('range(5000'),
            code.count('range(10000')
        ]
        
        volume_score = sum(data_indicators[i] * (10 ** i) for i in range(len(data_indicators)))
        return volume_score
    
    def _find_independent_sections(self, code: str) -> int:
        """Поиск независимых секций кода"""
        # Простая эвристика: считаем независимые функции
        return code.count('def ')
    
    def _estimate_loop_iterations(self, code: str) -> int:
        """Оценка количества итераций циклов"""
        iteration_patterns = [
            ('range(10', 10), ('range(50', 50), ('range(100', 100),
            ('range(500', 500), ('range(1000', 1000), ('range(5000', 5000)
        ]
        
        total_iterations = 0
        for pattern, multiplier in iteration_patterns:
            total_iterations += code.count(pattern) * multiplier
        
        return max(total_iterations, 1)
    
    def _detect_memory_intensive_operations(self, code: str) -> int:
        """Обнаружение операций, интенсивно использующих память"""
        memory_ops = [
            'append', 'extend', 'insert', 'list(', 'dict(', 'set(',
            'DataFrame', 'np.array', 'zeros', 'ones', 'empty'
        ]
        
        return sum(code.count(op) for op in memory_ops)
    
    def _extract_data_sections(self, lines: List[str]) -> List[str]:
        """Извлечение секций обработки данных из кода"""
        # Упрощенная реализация - в реальной системе здесь был бы парсинг AST
        sections = []
        current_section = []
        
        for line in lines:
            if line.strip().startswith('def ') and current_section:
                sections.append('\n'.join(current_section))
                current_section = []
            current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections if sections else ['\n'.join(lines)]
    
    def _create_subtask_template(self, base_code: str, subtask_id: int, total_subtasks: int) -> str:
        """Создание шаблона подзадачи"""
        return f"""
# Subtask {subtask_id + 1} of {total_subtasks}
# This is an automatically generated subtask

def process_subtask():
    print("Executing subtask {subtask_id + 1}/{total_subtasks}")
    # Base code section:
{self._indent_code(base_code)}
    return "subtask_result_{subtask_id}"

result = process_subtask()
"""
    
    def _indent_code(self, code: str, indent: int = 4) -> str:
        """Добавление отступа к коду"""
        spaces = ' ' * indent
        return '\n'.join(f"{spaces}{line}" for line in code.split('\n'))


# ... (Здесь должны быть ваши классы TaskSplitter и TaskComplexityAnalyzer без изменений) ...

class BioMedExecutor:
    def __init__(self):
        # Подключаем API транспорта
        self.transport = MasterNodeTransport(broker_host='localhost', broker_port=1883)

        # Логика разбиения
        self.splitter = TaskSplitter()

        # Реестр активных задач (вместо Redis храним состояние текущих вычислений в памяти)
        # Структура: {
        #   'task_uuid': {
        #       'total_parts': 3,
        #       'received_parts': {0: data, 1: data},
        #       'params': {...}
        #   }
        # }
        self.active_tasks = {}

        # Флаг работы
        self.running = False

    def start(self):
        print("[Master] Запуск системы...")
        if self.transport.start():
            self.running = True
            print("[Master] Транспорт подключен.")
            # Подписка на уведомления об отключении нод
            self.transport.set_dead_nodes_callback(self._handle_dead_nodes)
        else:
            print("[Master Error] Не удалось подключиться к MQTT брокеру.")

    def _handle_dead_nodes(self, dead_nodes_ids):
        print(f"[Alert] Потеряна связь с нодами: {dead_nodes_ids}")
        # Здесь можно добавить логику переназначения задач (re-queue), если нода умерла

    def submit_task(self, program_code: str, params: dict = None):
        """
        Главный метод: Принимает задачу -> Разбивает -> Отправляет на ESP32
        """
        if not self.running:
            print("[Error] Мастер не запущен.")
            return

        if params is None:
            params = {}

        task_id = str(uuid.uuid4())
        print(f"\n[Task] Новая задача {task_id}. Анализ...")

        # 1. Анализ и разбивка (используем ваш Splitter)
        complexity = self.splitter.analyze_task_complexity(program_code)

        if complexity.get('needs_splitting', False):
            # Разбиваем на подзадачи
            subtasks_list = self.splitter.split_complex_task(program_code, params)
        else:
            # Одна задача
            subtasks_list = [{
                'program_code': program_code,
                'task_data': params,
                'is_subtask': False
            }]

        total_parts = len(subtasks_list)
        print(f"[Task] Задача разбита на {total_parts} частей.")

        # 2. Регистрируем задачу в памяти, чтобы ждать ответы
        self.active_tasks[task_id] = {
            'total_parts': total_parts,
            'received_parts': {},
            'params': params,
            'created_at': time.time()
        }

        # 3. Отправка воркерам
        self._distribute_subtasks(task_id, subtasks_list)

        return task_id

    def _distribute_subtasks(self, task_id, subtasks):
        """Распределяет части задачи по доступным нодам"""
        # Получаем список живых нод через API
        nodes = self.transport.get_nodes_by_status("ready")
        if not nodes:
            nodes = self.transport.get_nodes_by_status("connected")

        if not nodes:
            print("[Error] Нет доступных воркеров! Задача висит.")
            # В реальной системе тут нужна очередь ожидания
            return

        available_node_ids = [n['node_id'] for n in nodes]
        worker_count = len(available_node_ids)

        for i, subtask in enumerate(subtasks):
            # Простой Round-Robin распределитель
            target_node = available_node_ids[i % worker_count]

            # Формируем пакет для ESP32
            # ВАЖНО: ESP32 должна вернуть task_id и subtask_index в ответе!
            payload = {
                "task_id": task_id,
                "subtask_index": i,
                "program_code": subtask['program_code'],
                "params": subtask['task_data'],
                # Добавляем priority, так как API send_task его поддерживает
                "priority": 5
            }

            self.transport.send_task(target_node, payload, priority=5)
            print(f" -> Часть {i} отправлена на {target_node}")

    def loop(self):
        """Основной цикл обработки результатов"""
        try:
            while self.running:
                # ВЫЗОВ ФУНКЦИИ СБОРА ОТВЕТОВ
                process_worker_responses(self.transport, self.active_tasks)

                time.sleep(0.5)  # Не грузим CPU
        except KeyboardInterrupt:
            print("[Master] Остановка...")


# --- ЗАПУСК ---
if __name__ == "__main__":
    executor = BioMedExecutor()
    executor.start()

    # Даем время на подключение
    time.sleep(2)

    # Пример отправки задачи
    code = "def heavy_computation(): return 42"
    executor.submit_task(code, {"type": "bio_signal_analysis"})

    # Запускаем бесконечный цикл ожидания ответов
    executor.loop()
# Примеры программ разной сложности (остаются без изменений)
SIMPLE_TASKS = [
   # Простая задача - вычисление суммы
    """
def calculate_sum(n):
    total = 0
    for i in range(n):
        total += i
    return total

result = calculate_sum(1000)
print(f"Sum: {result}")
""",

    # Простая задача - работа со списком
    """
def process_list(items):
    result = []
    for item in items:
        if item % 2 == 0:
            result.append(item * 2)
        else:
            result.append(item * 3)
    return result

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
processed = process_list(data)
print(f"Processed: {processed}")
""",

    # Простая задача - фильтрация данных
    """
def filter_data(data, threshold):
    filtered = []
    for item in data:
        if item > threshold:
            filtered.append(item)
    return filtered

numbers = [5, 12, 8, 3, 15, 7, 20, 1, 9]
result = filter_data(numbers, 8)
print(f"Filtered: {result}")
"""
]

MEDIUM_TASKS = [
    # Средняя сложность - вложенные циклы
    """
def matrix_operations(matrix1, matrix2):
    result = []
    for i in range(len(matrix1)):
        row = []
        for j in range(len(matrix2[0])):
            sum_val = 0
            for k in range(len(matrix2)):
                sum_val += matrix1[i][k] * matrix2[k][j]
            row.append(sum_val)
        result.append(row)
    return result

mat1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
mat2 = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
result = matrix_operations(mat1, mat2)
print(f"Matrix result: {result}")
""",

    # Средняя сложность - рекурсивные вычисления
    """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def calculate_fibonacci_sequence(count):
    sequence = []
    for i in range(count):
        sequence.append(fibonacci(i))
    return sequence

fib_sequence = calculate_fibonacci_sequence(15)
print(f"Fibonacci: {fib_sequence}")
""",

    # Средняя сложность - работа со словарями
    """
def analyze_text(text):
    words = text.split()
    word_count = {}
    
    for word in words:
        word = word.lower().strip('.,!?;:')
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    
    # Сортировка по частоте
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:10]

text = "hello world hello python world programming python data analysis machine learning python data science"
result = analyze_text(text)
print(f"Top words: {result}")
"""
]

COMPLEX_TASKS = [
    # Сложная задача - алгоритмы сортировки и поиска
    """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Генерация и обработка данных
large_dataset = [random.randint(1, 1000) for _ in range(1000)]
sorted_data = quick_sort(large_dataset)

# Поиск нескольких элементов
targets = [42, 123, 789, 999]
for target in targets:
    index = binary_search(sorted_data, target)
    print(f"Target {target} found at index: {index}")
""",
     # Сложная задача - алгоритмы сортировки и поиска
    """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Генерация и обработка данных
large_dataset = [random.randint(1, 1000) for _ in range(1000)]
sorted_data = quick_sort(large_dataset)

# Поиск нескольких элементов
targets = [42, 123, 789, 999]
for target in targets:
    index = binary_search(sorted_data, target)
    print(f"Target {target} found at index: {index}")
""",

    # Сложная задача - имитация обработки данных
    """
def process_large_dataset():
    # Создание большого набора данных
    data = []
    for i in range(1000):
        row = {
            'id': i,
            'value1': random.random() * 100,
            'value2': random.random() * 200,
            'value3': random.random() * 300
        }
        data.append(row)
    
    # Сложная обработка с несколькими циклами
    results = []
    for item in data:
        processed = {}
        processed['id'] = item['id']
        processed['sum'] = item['value1'] + item['value2'] + item['value3']
        processed['avg'] = processed['sum'] / 3
        processed['product'] = item['value1'] * item['value2'] * item['value3']
        
        # Дополнительные вычисления
        if processed['avg'] > 100:
            processed['category'] = 'high'
        elif processed['avg'] > 50:
            processed['category'] = 'medium'
        else:
            processed['category'] = 'low'
        
        results.append(processed)
    
    # Агрегация результатов
    category_count = {'high': 0, 'medium': 0, 'low': 0}
    total_sum = 0
    for result in results:
        category_count[result['category']] += 1
        total_sum += result['sum']
    
    return {
        'results': results[:10],  # Возвращаем только первые 10 для вывода
        'category_count': category_count,
        'total_sum': total_sum
    }

result = process_large_dataset()
print(f"Processing complete: {result['category_count']}")
""",

    # Сложная задача - многопоточная имитация
    """
import threading
import queue

class DataProcessor:
    def __init__(self, data_chunks):
        self.data_chunks = data_chunks
        self.results = queue.Queue()
        self.threads = []
    
    def process_chunk(self, chunk, chunk_id):
        # Имитация сложной обработки
        chunk_result = []
        for item in chunk:
            processed = {}
            processed['original'] = item
            processed['squared'] = item ** 2
            processed['sqrt'] = item ** 0.5
            processed['log'] = item + 1  # Упрощенный логарифм
            
            # Вложенные вычисления
            for i in range(5):
                processed[f'iter_{i}'] = processed['squared'] * i
            
            chunk_result.append(processed)
        
        self.results.put((chunk_id, chunk_result))
    
    def process_all(self):
        # Запуск потоков для обработки чанков
        for i, chunk in enumerate(self.data_chunks):
            thread = threading.Thread(target=self.process_chunk, args=(chunk, i))
            self.threads.append(thread)
            thread.start()
        
        # Ожидание завершения всех потоков
        for thread in self.threads:
            thread.join()
        
        # Сбор результатов
        all_results = []
        while not self.results.empty():
            chunk_id, chunk_result = self.results.get()
            all_results.extend(chunk_result)
        
        return all_results

# Подготовка данных
data = [random.randint(1, 100) for _ in range(500)]
chunks = [data[i:i+100] for i in range(0, len(data), 100)]

processor = DataProcessor(chunks)
results = processor.process_all()
print(f"Processed {len(results)} items with threading")
"""
]


def example_usage_with_task_splitting():
    
    balancer = EnhancedQueueTaskLoadBalancer()
    balancer.clear_all_queues()
    
    # Регистрируем узлы
    balancer.update_node_metrics("worker_1", {
        'cpu_load': 20.0, 'memory_usage': 45.0, 'memory_capacity': 2048,
        'response_time': 50, 'is_available': True
    })
    
    balancer.update_node_metrics("worker_2", {
        'cpu_load': 15.0, 'memory_usage': 30.0, 'memory_capacity': 4096,
        'response_time': 30, 'is_available': True
    })
    
    balancer.update_node_metrics("worker_3", {
        'cpu_load': 60.0, 'memory_usage': 80.0, 'memory_capacity': 1024,
        'response_time': 100, 'is_available': True
    })
    
    # Словарь с задачами разной сложности
    TASK_POOL = {
        'SIMPLE': SIMPLE_TASKS,
        'MEDIUM': MEDIUM_TASKS, 
        'COMPLEX': COMPLEX_TASKS
    }
    
    # Вероятности выбора задач разной сложности
    COMPLEXITY_WEIGHTS = {
        'SIMPLE': 0.4,    # 40% простых задач
        'MEDIUM': 0.35,   # 35% средних задач  
        'COMPLEX': 0.25   # 25% сложных задач
    }
    
    print(" Advanced Task Load Balancer with Intelligent Splitting")
    print("=" * 70)
    print(" Task complexity distribution:")
    for complexity, weight in COMPLEXITY_WEIGHTS.items():
        print(f"   {complexity}: {weight*100}%")
    print("=" * 70)
    
    added_tasks = []
    total_tasks_to_add = 15
    
    # Создаем уникальный пул задач, чтобы избежать дубликатов
    used_programs = set()
    available_tasks = {
        'SIMPLE': SIMPLE_TASKS.copy(),
        'MEDIUM': MEDIUM_TASKS.copy(),
        'COMPLEX': COMPLEX_TASKS.copy()
    }
    
    composite_tasks = []
    
    for i in range(total_tasks_to_add):
        # Случайный выбор сложности задачи на основе весов
        complexity = random.choices(
            list(COMPLEXITY_WEIGHTS.keys()), 
            weights=list(COMPLEXITY_WEIGHTS.values())
        )[0]
        
        # Выбираем задачу из доступных для этой сложности
        if available_tasks[complexity]:
            program = random.choice(available_tasks[complexity])
            available_tasks[complexity].remove(program)
        else:
            # Если задачи закончились, берем любую из этой категории
            program = random.choice(TASK_POOL[complexity])
        
        best_node = balancer.get_best_node('weighted_composite', check_failures=False)
        
        if best_node:
            task_data = {
                'type': 'program_execution', 
                'name': f'{complexity}_Task_{i+1:02d}',
                'category': complexity,
                'iteration': i + 1
            }
            
            print(f"\n Adding {complexity} task {i+1:02d}...")
            
            if complexity == 'COMPLEX':
                # Для сложных задач используем автоматическую разбивку
                composite_task_id = balancer.add_complex_task(best_node, program, task_data)
                composite_tasks.append(composite_task_id)
                print(f"    Complex task split into subtasks: {composite_task_id}")
            else:
                # Для простых и средних задач используем обычный метод
                task_id = balancer.add_program_task(best_node, program, task_data, use_cache=False)
                added_tasks.append((task_id, complexity, best_node))
        else:
            print(f" No available node for {complexity} task {i+1:02d}")

    # Показываем состояние очередей
    print("\n" + "=" * 70)
    print(" Queues state with predicted times")
    print("=" * 70)
    
    queues_info = balancer.get_all_queues_info()
    total_tasks = 0
    total_memory = 0
    total_time = 0
    
    for node_id, info in queues_info.items():
        print(f"  Node {node_id}: {info['queue_length']:2d} tasks, "
              f"{info['total_memory_usage']:7.2f}MB memory, "
              f"{info['total_execution_time']:7.2f}s total time")
        total_tasks += info['queue_length']
        total_memory += info['total_memory_usage']
        total_time += info['total_execution_time']
    
    print(f"\n TOTAL: {total_tasks} tasks, {total_memory:.2f}MB, {total_time:.2f}s")

    # Мониторинг составных задач
    if composite_tasks:
        print(f"\n Monitoring {len(composite_tasks)} composite tasks...")
        for task_id in composite_tasks:
            status = balancer.check_composite_task_status(task_id)
            print(f"   {task_id}: {status['status']} - {status['progress']:.1f}% complete")

    # Обрабатываем ВСЕ задачи
    print("\n" + "=" * 70)
    print(" Processing all tasks...")
    print("=" * 70)
    
    balancer.process_all_tasks()
    
    # Финальная статистика
    print("\n" + "=" * 70)
    print("Final execution report")
    print("=" * 70)
    
    total_tasks = balancer.get_total_tasks_count()
    print(f"Remaining tasks: {total_tasks}")
    
    # Статистика выполнения для составных задач
    if composite_tasks:
        print(f"\n Composite tasks results:")
        for task_id in composite_tasks:
            status = balancer.check_composite_task_status(task_id)
            results = balancer.get_composite_task_results(task_id)
            
            print(f"   {task_id}:")
            print(f"      Status: {status['status']}")
            print(f"      Progress: {status['progress']:.1f}%")
            print(f"      Subtasks: {status['subtasks_completed']}/{status['subtasks_total']} completed")
            
            if results['summary']['failed'] > 0:
                print(f"Failed: {results['summary']['failed']} subtasks")
            else:
                print(f"All subtasks completed successfully")

if __name__ == "__main__":
    print("Starting Enhanced Queue Task Load Balancer")
    print("With intelligent task splitting and complexity analysis")
    print("=" * 70)
    
    # Запускаем пример с разбивкой задач
    example_usage_with_task_splitting()
    
