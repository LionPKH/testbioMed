import redis
import json
import time
import uuid
import random
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
import hashlib

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

class EnhancedQueueTaskLoadBalancer:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_password=None, socket_timeout=5, socket_connect_timeout=5):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        
        # Создаем соединение
        self.redis = self._create_redis_connection()
        
        self.nodes_key = "load_balancer:queue_nodes"
        self.queues_key_prefix = "queue:node:"
        self.failed_queues_key = "load_balancer:failed_queues"
        self.node_stats_key_prefix = "node:stats:"
        self.health_check_key_prefix = "node:health:"
        self.task_analyzer = TaskComplexityAnalyzer()
        self.task_splitter = TaskSplitter()
        
        # Кэш предсказаний для похожих программ
        self.prediction_cache = {}
        self.cache_ttl = 3600  # 1 час
        
        # Для управления составными задачами
        self.task_dependencies = {}
        self.composite_tasks = {}
    
    def _create_redis_connection(self):
        """Создать соединение с Redis с обработкой ошибок"""
        try:
            connection = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                decode_responses=True,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                health_check_interval=30
            )
            # Проверяем подключение
            connection.ping()
            return connection
        except redis.ConnectionError as e:
            raise
        except redis.AuthenticationError as e:
            raise
        except Exception as e:
            raise
    
    def _ensure_connection(self, max_retries=3):
        """Проверить и восстановить соединение при необходимости"""
        for attempt in range(max_retries):
            try:
                self.redis.ping()
                return True
            except (redis.ConnectionError, redis.TimeoutError) as e:
                print(f"Redis connection lost (attempt {attempt + 1}/{max_retries}), reconnecting...")
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    try:
                        self.redis = self._create_redis_connection()
                        return True
                    except:
                        continue
                else:
                    print(" Failed to reconnect to Redis after all attempts")
                    raise e
        return False
    
    def _execute_with_retry(self, operation, *args, **kwargs):
        """Выполнить операцию с повторными попытками"""
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                self._ensure_connection()
                return operation(*args, **kwargs)
            except (redis.ConnectionError, redis.TimeoutError, redis.ResponseError) as e:
                last_exception = e
                print(f"Redis operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    try:
                        self.redis = self._create_redis_connection()
                    except:
                        continue
                else:
                    print(f" Operation failed after {max_retries} attempts: {last_exception}")
                    raise last_exception
        
        raise last_exception

    def predict_task_duration(self, program_code: str, task_data: Dict = None, use_cache: bool = True) -> float:
        """Предсказание времени выполнения задачи"""
        # Создаем ключ для кэша
        cache_key = hashlib.md5(program_code.encode()).hexdigest()
        
        # Проверяем кэш только если разрешено
        if use_cache and cache_key in self.prediction_cache:
            cached_time, timestamp = self.prediction_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                features = self.task_analyzer.extract_program_features(program_code)
                print(f" Using cached prediction: {cached_time:.2f}s | Code: {features['code_length']} chars | "
                      f"Functions: {features['function_count']} | Loops depth: {features['loop_depth']} | "
                      f"Memory ops: {features['memory_operations']}")
                return cached_time
        
        # Получаем исторические данные для похожих программ
        historical_data = self._get_historical_execution_data(program_code)
        
        # Предсказываем время выполнения
        predicted_time = self.task_analyzer.predict_execution_time(program_code, historical_data)
        
        # Сохраняем в кэш
        self.prediction_cache[cache_key] = (predicted_time, time.time())
        
        features = self.task_analyzer.extract_program_features(program_code)
        print(f" Predicted execution time: {predicted_time:.2f}s | Code: {features['code_length']} chars | "
              f"Functions: {features['function_count']} | Loops depth: {features['loop_depth']} | "
              f"Memory ops: {features['memory_operations']}")
        
        return predicted_time
    
    def _get_historical_execution_data(self, program_code: str) -> List[float]:
        """Получение исторических данных выполнения для похожих программ"""
        return []
    
    def estimate_memory_usage(self, program_code: str, task_data: Dict = None, use_cache: bool = True) -> float:
        """Оценка использования памяти программой"""
        # Создаем ключ для кэша памяти
        cache_key = f"memory_{hashlib.md5(program_code.encode()).hexdigest()}"
        
        # Проверяем кэш только если разрешено
        if use_cache and cache_key in self.prediction_cache:
            cached_memory, timestamp = self.prediction_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                print(f" Using cached memory estimate: {cached_memory:.2f}MB")
                return cached_memory
        
        base_memory = 50
        
        # Корректировка на основе размера кода
        code_memory = len(program_code) / 1024
        
        # Корректировка на основе структур данных
        features = self.task_analyzer.extract_program_features(program_code)
        structures = features.get('data_structures', {})
        
        memory_adjustment = 0
        memory_adjustment += structures.get('dataframes', 0) * 100
        memory_adjustment += structures.get('arrays', 0) * 50
        memory_adjustment += structures.get('lists', 0) * 10
        memory_adjustment += structures.get('dicts', 0) * 15
        
        # Корректировка на библиотеки
        libraries = features.get('library_usage', [])
        for lib in libraries:
            if lib in ['pandas', 'numpy']:
                memory_adjustment += 200
            elif lib in ['tensorflow', 'torch']:
                memory_adjustment += 500
        
        total_memory = base_memory + code_memory + memory_adjustment
        
        # Сохраняем в кэш
        self.prediction_cache[cache_key] = (total_memory, time.time())
        
        print(f"Estimated memory usage: {total_memory:.2f}MB")
        return total_memory
    
    def add_program_task(self, node_id: str, program_code: str, task_data: Dict = None, use_cache: bool = True) -> str:
        """Добавить задачу-программу в очередь с автоматическим предсказанием характеристик"""
        if task_data is None:
            task_data = {}
        
        # Предсказываем время выполнения и использование памяти
        expected_duration = self.predict_task_duration(program_code, task_data, use_cache=use_cache)
        memory_weight = self.estimate_memory_usage(program_code, task_data, use_cache=use_cache)
        
        # Добавляем программу в данные задачи
        task_data['program_code'] = program_code
        task_data['program_features'] = self.task_analyzer.extract_program_features(program_code)
        
        # Используем существующий метод для добавления задачи
        return self.add_task_to_queue(
            node_id, 
            task_data, 
            memory_weight=memory_weight,
            expected_duration=expected_duration
        )
    
    def add_complex_task(self, node_id: str, program_code: str, task_data: Dict = None) -> str:
        """Добавить сложную задачу с автоматической разбивкой на подзадачи"""
        if task_data is None:
            task_data = {}
        
        # Анализируем необходимость разбивки
        complexity = self.task_splitter.analyze_task_complexity(program_code)
        
        if not complexity['needs_splitting']:
            print(f" Task doesn't need splitting (score: {complexity['split_score']:.2f})")
            return self.add_program_task(node_id, program_code, task_data)
        
        print(f"Complex task detected (score: {complexity['split_score']:.2f}), splitting...")
        
        # Создаем составную задачу
        composite_task_id = str(uuid.uuid4())
        self.composite_tasks[composite_task_id] = {
            'main_task_id': composite_task_id,
            'original_code': program_code,
            'subtasks': [],
            'status': 'splitting',
            'created_at': time.time()
        }
        
        # Разбиваем на подзадачи
        subtasks = self.task_splitter.split_complex_task(program_code, task_data)
        
        print(f" Split into {len(subtasks)} subtasks")
        
        # Добавляем подзадачи в очередь
        subtask_ids = []
        for i, subtask in enumerate(subtasks):
            subtask_data = subtask['task_data']
            subtask_data['composite_task_id'] = composite_task_id
            subtask_data['subtask_index'] = i
            
            subtask_id = self.add_program_task(
                node_id, 
                subtask['program_code'], 
                subtask_data,
                use_cache=False  # Отключаем кэш для подзадач
            )
            
            subtask_ids.append(subtask_id)
            
            # Сохраняем информацию о подзадаче
            self.composite_tasks[composite_task_id]['subtasks'].append({
                'subtask_id': subtask_id,
                'status': 'queued',
                'index': i
            })
        
        self.composite_tasks[composite_task_id]['status'] = 'distributed'
        self.composite_tasks[composite_task_id]['subtask_ids'] = subtask_ids
        
        print(f" Composite task {composite_task_id} distributed with {len(subtask_ids)} subtasks")
        return composite_task_id
    
    def check_composite_task_status(self, composite_task_id: str) -> Dict:
        """Проверить статус составной задачи"""
        if composite_task_id not in self.composite_tasks:
            return {'status': 'not_found'}
        
        composite_task = self.composite_tasks[composite_task_id]
        completed = 0
        failed = 0
        processing = 0
        queued = 0
        
        for subtask in composite_task['subtasks']:
            task_info_json = self.redis.get(f"task:{subtask['subtask_id']}:info")
            if task_info_json:
                task_info = json.loads(task_info_json)
                status = task_info.get('status', 'unknown')
                
                if status == 'completed':
                    completed += 1
                elif status == 'failed':
                    failed += 1
                elif status == 'processing':
                    processing += 1
                else:
                    queued += 1
        
        total = len(composite_task['subtasks'])
        progress = (completed / total) * 100 if total > 0 else 0
        
        # Обновляем статус составной задачи
        if completed == total:
            composite_task['status'] = 'completed'
        elif failed > 0:
            composite_task['status'] = 'partial_failure'
        elif processing > 0 or queued > 0:
            composite_task['status'] = 'in_progress'
        
        return {
            'composite_task_id': composite_task_id,
            'status': composite_task['status'],
            'progress': progress,
            'subtasks_total': total,
            'subtasks_completed': completed,
            'subtasks_failed': failed,
            'subtasks_processing': processing,
            'subtasks_queued': queued
        }
    
    def get_composite_task_results(self, composite_task_id: str) -> Dict:
        """Получить результаты всех подзадач составной задачи"""
        if composite_task_id not in self.composite_tasks:
            return {'error': 'Composite task not found'}
        
        composite_task = self.composite_tasks[composite_task_id]
        results = {}
        
        for subtask in composite_task['subtasks']:
            task_info_json = self.redis.get(f"task:{subtask['subtask_id']}:info")
            if task_info_json:
                task_info = json.loads(task_info_json)
                results[subtask['subtask_id']] = {
                    'status': task_info.get('status', 'unknown'),
                    'result': task_info.get('result'),
                    'duration': task_info.get('actual_duration'),
                    'subtask_data': task_info.get('data', {})
                }
        
        return {
            'composite_task_id': composite_task_id,
            'original_task': composite_task['original_code'][:100] + "...",
            'results': results,
            'summary': {
                'total_subtasks': len(composite_task['subtasks']),
                'completed': len([r for r in results.values() if r['status'] == 'completed']),
                'failed': len([r for r in results.values() if r['status'] == 'failed'])
            }
        }

    # Остальные методы QueueTaskLoadBalancer остаются без изменений
    def update_node_metrics(self, node_id: str, metrics: Dict):
        def _update():
            current_stats = self._get_node_execution_stats(node_id)
            
            metrics_data = {
                'cpu_load': metrics.get('cpu_load', 0),
                'memory_usage': metrics.get('memory_usage', 0),
                'memory_capacity': metrics.get('memory_capacity', 1024),
                'last_updated': time.time(),
                'is_available': metrics.get('is_available', True),
                'response_time': metrics.get('response_time', 0),
                'queue_length': self.get_queue_length(node_id),
                'queue_memory_usage': self.get_queue_memory_usage(node_id),
                'queue_execution_time': self.get_queue_execution_time(node_id),
                'avg_execution_time': current_stats.get('avg_execution_time', 0),
                'tasks_processed': current_stats.get('tasks_processed', 0)
            }
            
            self.redis.hset(self.nodes_key, node_id, json.dumps(metrics_data))
            self._update_health_check(node_id)
        
        self._execute_with_retry(_update)
    
    def _update_health_check(self, node_id: str):
        def _update():
            health_key = f"{self.health_check_key_prefix}{node_id}"
            self.redis.setex(health_key, 60, str(time.time()))
        
        self._execute_with_retry(_update)
    
    def check_node_health(self, node_id: str) -> bool:
        def _check():
            health_key = f"{self.health_check_key_prefix}{node_id}"
            last_health = self.redis.get(health_key)
            
            if not last_health:
                return False
            
            return time.time() - float(last_health) < 45
        
        try:
            return self._execute_with_retry(_check)
        except:
            return False
    
    def auto_detect_failed_nodes(self) -> List[str]:
        def _detect():
            nodes_data = self.redis.hgetall(self.nodes_key)
            failed_nodes = []
            
            for node_id, data_str in nodes_data.items():
                data = json.loads(data_str)
                
                if not data.get('is_available', True):
                    continue
                
                if not self.check_node_health(node_id):
                    failed_nodes.append(node_id)
            
            for node_id in failed_nodes:
                print(f"Auto-detected failed node: {node_id}")
                self.mark_node_failed(node_id)
            
            return failed_nodes
        
        try:
            return self._execute_with_retry(_detect)
        except:
            return []
    
    def get_queue_key(self, node_id: str) -> str:
        return f"{self.queues_key_prefix}{node_id}"
    
    def get_queue_length(self, node_id: str) -> int:
        def _get_length():
            queue_key = self.get_queue_key(node_id)
            return self.redis.llen(queue_key)
        
        try:
            return self._execute_with_retry(_get_length)
        except:
            return 0
    
    def get_queue_memory_usage(self, node_id: str) -> float:
        def _get_memory():
            queue_key = self.get_queue_key(node_id)
            tasks_json = self.redis.lrange(queue_key, 0, -1)
            
            total_memory = 0
            for task_json in tasks_json:
                try:
                    task = json.loads(task_json)
                    total_memory += task.get('memory_weight', 0)
                except (json.JSONDecodeError, KeyError):
                    continue
            
            return total_memory
        
        try:
            return self._execute_with_retry(_get_memory)
        except:
            return 0.0
    
    def get_queue_execution_time(self, node_id: str) -> float:
        def _get_time():
            queue_key = self.get_queue_key(node_id)
            tasks_json = self.redis.lrange(queue_key, 0, -1)
            
            total_time = 0
            for task_json in tasks_json:
                try:
                    task = json.loads(task_json)
                    total_time += task.get('expected_duration', 0)
                except (json.JSONDecodeError, KeyError):
                    continue
            
            return total_time
        
        try:
            return self._execute_with_retry(_get_time)
        except:
            return 0.0
    
    def _get_node_execution_stats(self, node_id: str) -> Dict:
        def _get_stats():
            stats_key = f"{self.node_stats_key_prefix}{node_id}"
            stats_json = self.redis.get(stats_key)
            
            if stats_json:
                return json.loads(stats_json)
            
            return {
                'avg_execution_time': 0,
                'tasks_processed': 0,
                'total_execution_time': 0,
                'last_calculated': time.time()
            }
        
        try:
            return self._execute_with_retry(_get_stats)
        except:
            return {
                'avg_execution_time': 0,
                'tasks_processed': 0,
                'total_execution_time': 0,
                'last_calculated': time.time()
            }
    
    def _update_node_execution_stats(self, node_id: str, execution_time: float):
        def _update_stats():
            stats = self._get_node_execution_stats(node_id)
            
            total_tasks = stats['tasks_processed'] + 1
            total_time = stats['total_execution_time'] + execution_time
            
            stats.update({
                'avg_execution_time': total_time / total_tasks if total_tasks > 0 else 0,
                'tasks_processed': total_tasks,
                'total_execution_time': total_time,
                'last_calculated': time.time()
            })
            
            stats_key = f"{self.node_stats_key_prefix}{node_id}"
            self.redis.setex(stats_key, 86400, json.dumps(stats))
        
        try:
            self._execute_with_retry(_update_stats)
        except Exception as e:
            print(f"Warning: Failed to update node stats: {e}")
    
    def calculate_node_score(self, node_data: Dict, current_queue_length: int, 
                           strategy: str = 'balanced', task_memory: float = 0, 
                           expected_duration: float = 0) -> float:
        try:
            max_queue_size = node_data.get('max_queue_size', 100)
            max_queue_memory = node_data.get('max_queue_memory', 512)
            current_memory_usage = node_data.get('queue_memory_usage', 0)
            current_execution_time = node_data.get('queue_execution_time', 0)
            
            
            avg_node_time = node_data.get('avg_execution_time', 0)
            if avg_node_time > 0 and expected_duration > avg_node_time * 3:
                return float('inf')
            
            if strategy == 'cpu':
                score = node_data['cpu_load']
            elif strategy == 'response_time':
                score = node_data['response_time']
            elif strategy == 'queue_length':
                score = current_queue_length
            elif strategy == 'memory_aware':
                memory_usage_ratio = (current_memory_usage + task_memory) / max_queue_memory
                cpu_ratio = node_data['cpu_load'] / 100
                score = (memory_usage_ratio * 0.6) + (cpu_ratio * 0.4)
            elif strategy == 'execution_time_aware':
                queue_ratio = current_queue_length / max_queue_size
                memory_ratio = (current_memory_usage + task_memory) / max_queue_memory
                execution_ratio = min(node_data.get('avg_execution_time', 0) / 1000, 1.0)
                current_time_ratio = (current_execution_time + expected_duration) / (max_queue_size * avg_node_time) if avg_node_time > 0 else 0
                
                score = (queue_ratio * 0.3) + (memory_ratio * 0.3) + (execution_ratio * 0.2) + (current_time_ratio * 0.2)
            elif strategy == 'weighted_composite':
                queue_ratio = current_queue_length / max_queue_size
                memory_ratio = (current_memory_usage + task_memory) / max_queue_memory
                cpu_ratio = node_data['cpu_load'] / 100
                
                time_penalty = 0
                if avg_node_time > 0:
                    historical_time_ratio = min(expected_duration / avg_node_time, 3.0) / 3.0
                    queue_time_ratio = (current_execution_time + expected_duration) / (max_queue_size * avg_node_time * 2)
                    time_penalty = (historical_time_ratio * 0.7) + (queue_time_ratio * 0.3)
                
                score = (queue_ratio * 0.3) + (memory_ratio * 0.3) + (cpu_ratio * 0.2) + (time_penalty * 0.2)
            else:
                queue_ratio = current_queue_length / max_queue_size
                memory_ratio = (current_memory_usage + task_memory) / max_queue_memory
                cpu_ratio = node_data['cpu_load'] / 100
                
                score = (queue_ratio * 0.4) + (memory_ratio * 0.3) + (cpu_ratio * 0.3)
            
            return score
        except Exception as e:
            print(f"Error calculating node score: {e}")
            return float('inf')
    
    def get_best_node(self, strategy='balanced', task_memory: float = 0, 
                     expected_duration: float = 0, check_failures: bool = False) -> Optional[str]:
        def _find_best_node():
            if check_failures:
                self.auto_detect_failed_nodes()
            
            nodes_data = self.redis.hgetall(self.nodes_key)
            if not nodes_data:
                return None
            
            available_nodes = []
            
            for node_id, data_str in nodes_data.items():
                data = json.loads(data_str)
                
                if not data.get('is_available', True):
                    continue
                
                if time.time() - data['last_updated'] > 30:
                    pass
                
                current_queue_length = self.get_queue_length(node_id)
                
                score = self.calculate_node_score(data, current_queue_length, strategy, task_memory, expected_duration)
                
                available_nodes.append((node_id, score, data, current_queue_length))
            
            if not available_nodes:
                return None
            
            best_node = min(available_nodes, key=lambda x: x[1])
            return best_node[0]
        
        try:
            return self._execute_with_retry(_find_best_node)
        except:
            return None
    
    def add_task_to_queue(self, node_id: str, task_data: Dict, 
                         memory_weight: float = 0, expected_duration: float = 0) -> str:
        def _add_task():
            if not self._is_node_available(node_id):
                new_node_id = self.get_best_node('balanced', memory_weight, expected_duration, check_failures=False)
                if new_node_id:
                    print(f"Node {node_id} is not available, redirecting task to {new_node_id}")
                    node_id_actual = new_node_id
                else:
                    raise Exception(f"No available nodes for task. Original node {node_id} is down.")
            else:
                node_id_actual = node_id
            
            task_id = str(uuid.uuid4())
            task = {
                'task_id': task_id,
                'data': task_data,
                'created_at': time.time(),
                'node_id': node_id_actual,
                'status': 'queued',
                'memory_weight': memory_weight,
                'expected_duration': expected_duration
            }
            
            queue_key = self.get_queue_key(node_id_actual)
            self.redis.rpush(queue_key, json.dumps(task))
            
            self._update_node_queue_metrics(node_id_actual)
            
            self.redis.setex(f"task:{task_id}:info", 86400, json.dumps({
                'node_id': node_id_actual,
                'status': 'queued',
                'created_at': task['created_at'],
                'memory_weight': memory_weight,
                'expected_duration': expected_duration
            }))
            
            print(f"Task {task_id} assigned to {node_id_actual} (memory: {memory_weight:.2f}MB, duration: {expected_duration:.2f}s)")
            
            return task_id
        
        return self._execute_with_retry(_add_task)
    
    def _is_node_available(self, node_id: str) -> bool:
        def _check_available():
            node_data = self.redis.hget(self.nodes_key, node_id)
            if not node_data:
                return False
            
            data = json.loads(node_data)
            return data.get('is_available', True) and self.check_node_health(node_id)
        
        try:
            return self._execute_with_retry(_check_available)
        except:
            return False
    
    def get_next_task(self, node_id: str) -> Optional[Dict]:
        def _get_task():
            if not self._is_node_available(node_id):
                print(f"Node {node_id} is not available, cannot get next task")
                return None
                
            queue_key = self.get_queue_key(node_id)
            task_json = self.redis.lpop(queue_key)
            
            if task_json:
                task = json.loads(task_json)
                task['status'] = 'processing'
                task['started_at'] = time.time()
                
                self.redis.setex(f"task:{task['task_id']}:info", 86400, json.dumps({
                    'node_id': node_id,
                    'status': 'processing',
                    'created_at': task['created_at'],
                    'started_at': task['started_at'],
                    'memory_weight': task.get('memory_weight', 0),
                    'expected_duration': task.get('expected_duration', 0)
                }))
                
                self._update_node_queue_metrics(node_id)
                
                return task
            
            return None
        
        try:
            return self._execute_with_retry(_get_task)
        except:
            return None
    
    def complete_task(self, task_id: str, result: Dict = None, actual_duration: float = None):
        def _complete():
            task_info_json = self.redis.get(f"task:{task_id}:info")
            if task_info_json:
                task_info = json.loads(task_info_json)
                node_id = task_info.get('node_id')
                
                # Исправление: создаем локальную переменную для использования в замыкании
                calculated_duration = actual_duration
                if calculated_duration is None and task_info.get('started_at'):
                    calculated_duration = time.time() - task_info['started_at']
                
                if calculated_duration is not None and calculated_duration > 0:
                    self._update_node_execution_stats(node_id, calculated_duration)
                
                task_info['status'] = 'completed'
                task_info['completed_at'] = time.time()
                task_info['result'] = result
                if calculated_duration is not None:
                    task_info['actual_duration'] = calculated_duration
                
                self.redis.setex(f"task:{task_id}:info", 3600, json.dumps(task_info))
                
                if node_id:
                    self._update_node_queue_metrics(node_id)
        
        try:
            self._execute_with_retry(_complete)
        except Exception as e:
            print(f"Warning: Failed to complete task {task_id}: {e}")

    
    def fail_task(self, task_id: str, error: str = None):
        def _fail():
            task_info_json = self.redis.get(f"task:{task_id}:info")
            if task_info_json:
                task_info = json.loads(task_info_json)
                node_id = task_info.get('node_id')
                
                task_info['status'] = 'failed'
                task_info['failed_at'] = time.time()
                task_info['error'] = error
                
                self.redis.setex(f"task:{task_id}:info", 3600, json.dumps(task_info))
                
                if node_id:
                    self._update_node_queue_metrics(node_id)
        
        try:
            self._execute_with_retry(_fail)
        except Exception as e:
            print(f"Warning: Failed to mark task {task_id} as failed: {e}")
    
    def mark_node_failed(self, node_id: str, auto_redistribute: bool = True):
        def _mark_failed():
            print(f"Marking node {node_id} as failed...")
            
            current_data = self.redis.hget(self.nodes_key, node_id)
            if current_data:
                metrics = json.loads(current_data)
                metrics['is_available'] = False
                metrics['last_updated'] = time.time()
                self.redis.hset(self.nodes_key, node_id, json.dumps(metrics))
            
            failed_queue_info = {
                'node_id': node_id,
                'failed_at': time.time(),
                'queue_length': self.get_queue_length(node_id),
                'total_memory_usage': self.get_queue_memory_usage(node_id),
                'total_execution_time': self.get_queue_execution_time(node_id),
                'redistributed': False
            }
            self.redis.hset(self.failed_queues_key, node_id, json.dumps(failed_queue_info))
            
            if auto_redistribute:
                self._redistribute_failed_queue(node_id)
        
        self._execute_with_retry(_mark_failed)
    
    def _redistribute_failed_queue(self, failed_node_id: str):
        def _redistribute():
            print(f"Redistributing tasks from failed node {failed_node_id}...")
            
            queue_key = self.get_queue_key(failed_node_id)
            tasks_json = self.redis.lrange(queue_key, 0, -1)
            
            if not tasks_json:
                self._mark_queue_redistributed(failed_node_id)
                print(f"No tasks to redistribute from node {failed_node_id}")
                return
            
            redistributed_count = 0
            failed_count = 0
            
            for task_json in tasks_json:
                task = json.loads(task_json)
                task_id = task['task_id']
                memory_weight = task.get('memory_weight', 0)
                expected_duration = task.get('expected_duration', 0)
                
                new_node_id = self.get_best_node('weighted_composite', memory_weight, 
                                               expected_duration, check_failures=False)
                
                if new_node_id:
                    new_queue_key = self.get_queue_key(new_node_id)
                    self.redis.rpush(new_queue_key, task_json)
                    
                    task_info = {
                        'node_id': new_node_id,
                        'status': 'queued',
                        'created_at': task['created_at'],
                        'original_node': failed_node_id,
                        'redistributed_at': time.time(),
                        'memory_weight': memory_weight,
                        'expected_duration': expected_duration
                    }
                    self.redis.setex(f"task:{task_id}:info", 86400, json.dumps(task_info))
                    
                    redistributed_count += 1
                    self._update_node_queue_metrics(new_node_id)
                    
                    print(f"Task {task_id} redistributed from {failed_node_id} to {new_node_id} (memory: {memory_weight}MB, duration: {expected_duration}s)")
                else:
                    print(f"Warning: No available nodes for task {task_id} (memory: {memory_weight}MB, duration: {expected_duration}s)")
                    failed_count += 1
            
            self.redis.delete(queue_key)
            self._mark_queue_redistributed(failed_node_id, redistributed_count)
            
            print(f"Redistribution completed: {redistributed_count} tasks redistributed, {failed_count} failed from node {failed_node_id}")
        
        self._execute_with_retry(_redistribute)
    
    def _mark_queue_redistributed(self, node_id: str, redistributed_count: int = 0):
        def _mark():
            failed_queue_info = {
                'node_id': node_id,
                'failed_at': time.time(),
                'queue_length': 0,
                'redistributed': True,
                'redistributed_at': time.time(),
                'redistributed_count': redistributed_count
            }
            self.redis.hset(self.failed_queues_key, node_id, json.dumps(failed_queue_info))
        
        self._execute_with_retry(_mark)
    
    def _update_node_queue_metrics(self, node_id: str):
        def _update_metrics():
            current_data = self.redis.hget(self.nodes_key, node_id)
            if current_data:
                metrics = json.loads(current_data)
                metrics['queue_length'] = self.get_queue_length(node_id)
                metrics['queue_memory_usage'] = self.get_queue_memory_usage(node_id)
                metrics['queue_execution_time'] = self.get_queue_execution_time(node_id)
                metrics['last_updated'] = time.time()
                self.redis.hset(self.nodes_key, node_id, json.dumps(metrics))
        
        try:
            self._execute_with_retry(_update_metrics)
        except Exception as e:
            print(f"Warning: Failed to update node metrics for {node_id}: {e}")
    
    def get_failed_queues_info(self) -> Dict:
        def _get_failed():
            failed_queues = self.redis.hgetall(self.failed_queues_key)
            result = {}
            
            for node_id, data_str in failed_queues.items():
                result[node_id] = json.loads(data_str)
            
            return result
        
        try:
            return self._execute_with_retry(_get_failed)
        except:
            return {}
    
    def recover_node(self, node_id: str) -> bool:
        def _recover():
            node_data = self.redis.hget(self.nodes_key, node_id)
            if not node_data:
                return False
            
            data = json.loads(node_data)
            data['is_available'] = True
            data['last_updated'] = time.time()
            
            self.redis.hset(self.nodes_key, node_id, json.dumps(data))
            self._update_health_check(node_id)
            self.redis.hdel(self.failed_queues_key, node_id)
            
            print(f"Node {node_id} recovered successfully")
            return True
        
        try:
            return self._execute_with_retry(_recover)
        except:
            return False
    
    def get_node_queue_info(self, node_id: str) -> Dict:
        def _get_info():
            queue_key = self.get_queue_key(node_id)
            tasks_json = self.redis.lrange(queue_key, 0, -1)
            
            tasks = []
            total_memory = 0
            total_time = 0
            for task_json in tasks_json:
                task = json.loads(task_json)
                memory_weight = task.get('memory_weight', 0)
                expected_duration = task.get('expected_duration', 0)
                tasks.append({
                    'task_id': task['task_id'],
                    'created_at': task['created_at'],
                    'status': 'queued',
                    'memory_weight': memory_weight,
                    'expected_duration': expected_duration
                })
                total_memory += memory_weight
                total_time += expected_duration
            
            return {
                'node_id': node_id,
                'queue_length': len(tasks),
                'total_memory_usage': total_memory,
                'total_execution_time': total_time,
                'tasks': tasks
            }
        
        try:
            return self._execute_with_retry(_get_info)
        except:
            return {
                'node_id': node_id,
                'queue_length': 0,
                'total_memory_usage': 0,
                'total_execution_time': 0,
                'tasks': []
            }
    
    def get_all_queues_info(self) -> Dict:
        def _get_all():
            nodes_data = self.redis.hgetall(self.nodes_key)
            queues_info = {}
            
            for node_id in nodes_data.keys():
                queues_info[node_id] = self.get_node_queue_info(node_id)
            
            return queues_info
        
        try:
            return self._execute_with_retry(_get_all)
        except:
            return {}

    def process_all_tasks(self, node_ids: List[str] = None):
        if node_ids is None:
            nodes_data = self.redis.hgetall(self.nodes_key)
            node_ids = [node_id for node_id in nodes_data.keys() 
                    if self._is_node_available(node_id)]
        
        print(f"\n=== Processing ALL tasks on nodes: {node_ids} ===")
        total_processed = 0
        
        for node_id in node_ids:
            if not self._is_node_available(node_id):
                print(f"Skipping unavailable node: {node_id}")
                continue
                
            node_processed = 0
            print(f"\nProcessing tasks on node {node_id}:")
            
            while True:
                task = self.get_next_task(node_id)
                if not task:
                    break
                    
                memory = task.get('memory_weight', 0)
                expected_duration = task.get('expected_duration', 0)
                
                print(f"  Processing task {task['task_id']} (memory: {memory}MB, expected: {expected_duration}s)")
                
                # Имитация обработки задачи - используем предсказанное время
                processing_time = min(expected_duration / 10, 0.5)
                time.sleep(processing_time)
                
                # Передаем actual_duration явно
                self.complete_task(task['task_id'], {'result': 'success'}, actual_duration=expected_duration)
                node_processed += 1
                total_processed += 1
            
            print(f"Completed {node_processed} tasks on node {node_id}")
        
        print(f"\n=== TOTAL PROCESSED: {total_processed} tasks ===")
        return total_processed

    def get_total_tasks_count(self) -> int:
        queues_info = self.get_all_queues_info()
        total = 0
        for node_id, info in queues_info.items():
            total += info['queue_length']
        return total

    def clear_all_queues(self):
        def _clear():
            for node_id in ["worker_1", "worker_2", "worker_3"]:
                queue_key = self.get_queue_key(node_id)
                self.redis.delete(queue_key)
            self.redis.delete(self.nodes_key)
            self.redis.delete(self.failed_queues_key)
            print("All queues cleared")
        
        try:
            self._execute_with_retry(_clear)
        except Exception as e:
            print(f"Warning: Failed to clear queues: {e}")

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
    
