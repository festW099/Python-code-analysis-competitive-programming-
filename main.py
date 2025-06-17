import ast
import sys
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
import colorama
from colorama import Fore, Style

colorama.init()

COLORS = {
    'error': Fore.RED,
    'warning': Fore.YELLOW,
    'info': Fore.CYAN,
    'success': Fore.GREEN,
    'highlight': Fore.MAGENTA,
    'reset': Style.RESET_ALL
}

@dataclass
class AnalysisResult:
    time_complexity: str
    space_complexity: str
    optimizations: List[str]
    data_structures: Set[str]
    algorithms: Set[str]
    techniques: Set[str]
    issues: List[str]
    style_issues: List[str]
    potential_bugs: List[str]
    performance_tips: List[str]

class PythonCodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.results: Dict[str, AnalysisResult] = {}
        self.current_function: Optional[str] = None
        self.nested_loops: int = 0
        self.recursion_depth: int = 0
        self.techniques: Set[str] = set()
        self.has_recursion: bool = False
        self.has_dp: bool = False
        self.variables: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.array_accesses: List[Tuple[str, int]] = []
        self.divisions: List[Tuple[str, int]] = []
        self.input_output: List[Tuple[str, int]] = []
        self.magic_numbers: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
        self.uninitialized_vars: Set[str] = set()
        self.loop_invariants: List[Tuple[str, int]] = []
        self.list_comprehensions: List[Tuple[str, int]] = []
        self.deprecated_constructs: List[Tuple[str, int]] = []

    def analyze(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            self._detect_uninitialized_vars(code)
            self._detect_magic_numbers(code)
            self._detect_deprecated(code)
            self.visit(tree)
            self._post_analysis()
            return self._generate_report()
        except SyntaxError as e:
            return f"{COLORS['error']}Ошибка синтаксиса: {e}{COLORS['reset']}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.current_function = node.name
        self.results[node.name] = AnalysisResult(
            time_complexity="O(1)",
            space_complexity="O(1)",
            optimizations=[],
            data_structures=set(),
            algorithms=set(),
            techniques=set(),
            issues=[],
            style_issues=[],
            potential_bugs=[],
            performance_tips=[]
        )
        self.nested_loops = 0
        self.recursion_depth = 0
        
        if not node.returns and all(a.annotation is None for a in node.args.args):
            self.results[node.name].style_issues.append(
                f"Строка {node.lineno}: Отсутствуют аннотации типов"
            )
        
        self.generic_visit(node)
        
        self._update_complexity(node.name)
        self._detect_techniques(node.name)
        self.current_function = None

    def _analyze_loop(self, node: ast.For) -> None:
        if self.current_function:
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == 'range':
                    if len(node.iter.args) == 1:
                        self.results[self.current_function].time_complexity = "O(n)"
                    elif len(node.iter.args) == 2:
                        self.results[self.current_function].time_complexity = "O(n)"
                    
                    if self.nested_loops > 1:
                        self.results[self.current_function].time_complexity = f"O(n^{self.nested_loops})"

    def visit_For(self, node: ast.For) -> None:
        if self.current_function:
            self.nested_loops += 1
            self._analyze_loop(node)
            self._detect_loop_invariants(node)
            self.generic_visit(node)
            self.nested_loops -= 1

    def visit_While(self, node: ast.While) -> None:
        if self.current_function:
            self.nested_loops += 1
            self.results[self.current_function].time_complexity = "O(n)"
            self._detect_loop_invariants(node)
            self.generic_visit(node)
            self.nested_loops -= 1

    def visit_Call(self, node: ast.Call) -> None:
        if not self.current_function:
            self.generic_visit(node)
            return

        if isinstance(node.func, ast.Name) and node.func.id == self.current_function:
            self.has_recursion = True
            self.recursion_depth += 1
            self.results[self.current_function].techniques.add("Рекурсия")

        self._detect_algorithms(node)
        self._detect_data_structures(node)
        self._check_io_optimization(node)

        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        if self.current_function:
            self.list_comprehensions.append((self.current_function, node.lineno))
            self.results[self.current_function].performance_tips.append(
                f"Строка {node.lineno}: List comprehension обычно быстрее цикла for"
            )
            self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self._check_division_by_zero(node)
        if isinstance(node.op, (ast.Add, ast.Mult)):
            self._check_overflow(node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._check_array_bounds(node)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables[self.current_function or 'global'][target.id] = node.lineno
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if (self.current_function and 
                node.id not in self.variables[self.current_function] and 
                node.id not in self.variables['global']):
                self._add_potential_bug(
                    self.current_function or 'global',
                    node.lineno,
                    f"Возможно использование неинициализированной переменной: {node.id}"
                )
        self.generic_visit(node)

    def _detect_algorithms(self, node: ast.Call) -> None:
        func_mapping = {
            'bisect_left': "Бинарный поиск",
            'bisect_right': "Бинарный поиск",
            'gcd': "Алгоритм Евклида",
            'lru_cache': "Динамическое программирование",
            'heappush': "Куча",
            'heappop': "Куча",
            'dfs': "Поиск в глубину",
            'bfs': "Поиск в ширину",
            'dijkstra': "Алгоритм Дейкстры",
            'combinations': "Комбинаторика",
            'permutations': "Перестановки",
            'accumulate': "Префиксные суммы"
        }

        try:
            if isinstance(node.func, ast.Name):
                if node.func.id in func_mapping:
                    self.results[self.current_function].algorithms.add(func_mapping[node.func.id])
                    if func_mapping[node.func.id] == "Динамическое программирование":
                        self.has_dp = True
            
            elif isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                if attr_name in func_mapping:
                    self.results[self.current_function].algorithms.add(func_mapping[attr_name])
                
                if (isinstance(node.func.value, ast.Name) and node.func.value.id in ('itertools', 'heapq')):
                    full_name = f"{node.func.value.id}.{node.func.attr}"
                    if full_name in func_mapping:
                        self.results[self.current_function].algorithms.add(func_mapping[full_name])
                        
        except AttributeError:
            pass

    def _detect_data_structures(self, node: ast.Call) -> None:
        ds_mapping = {
            'append': 'list',
            'pop': 'list',
            'add': 'set',
            'remove': 'set',
            'discard': 'set',
            'keys': 'dict',
            'values': 'dict',
            'items': 'dict',
            'popleft': 'deque',
            'appendleft': 'deque'
        }

        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ds_mapping:
                ds = ds_mapping[node.func.attr]
                self.results[self.current_function].data_structures.add(ds)
                
                if ds == 'list' and node.func.attr == 'insert':
                    self.results[self.current_function].performance_tips.append(
                        f"Строка {node.lineno}: list.insert имеет O(n) сложность, используйте deque для частых вставок"
                    )
                elif ds == 'dict' and node.func.attr == 'keys':
                    self.results[self.current_function].performance_tips.append(
                        f"Строка {node.lineno}: Проверьте, можно ли заменить dict.keys() на прямое использование словаря"
                    )

    def _check_array_bounds(self, node: ast.Subscript) -> None:
        if self.current_function:
            if isinstance(node.slice, ast.Index):
                self.array_accesses.append((self.current_function, node.lineno))
                self._add_potential_bug(
                    self.current_function,
                    node.lineno,
                    "Прямой доступ по индексу - возможен выход за границы списка"
                )

    def _check_division_by_zero(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Div) or isinstance(node.op, ast.FloorDiv):
            if self.current_function:
                self.divisions.append((self.current_function, node.lineno))
                if isinstance(node.right, ast.Constant) and node.right.value == 0:
                    self._add_potential_bug(
                        self.current_function,
                        node.lineno,
                        "Обнаружено деление на ноль"
                    )
                else:
                    self._add_potential_bug(
                        self.current_function,
                        node.lineno,
                        "Потенциальное деление на ноль - нужна проверка знаменателя"
                    )

    def _check_overflow(self, node: ast.BinOp) -> None:
        if self.current_function:
            if isinstance(node.op, (ast.Add, ast.Mult)):
                self._add_potential_bug(
                    self.current_function,
                    node.lineno,
                    "Возможное переполнение при арифметической операции"
                )

    def _check_io_optimization(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id == 'print':
                self.input_output.append((self.current_function or 'global', node.lineno))
                self._add_performance_tip(
                    self.current_function or 'global',
                    node.lineno,
                    "Для вывода больших данных используйте sys.stdout.write() вместо print()"
                )

    def _detect_loop_invariants(self, node: ast.AST) -> None:
        if self.current_function:
            for child in ast.walk(node):
                if isinstance(child, ast.BinOp):
                    self.loop_invariants.append((self.current_function, node.lineno))
                    self._add_performance_tip(
                        self.current_function,
                        node.lineno,
                        "Возможный инвариант цикла - можно вынести за пределы цикла"
                    )

    def _detect_uninitialized_vars(self, code: str) -> None:
        lines = code.split('\n')
        for i, line in enumerate(lines):
            matches = re.finditer(r'(?<!def\s)(?<!class\s)\b(\w+)\s*=\s*', line)
            for match in matches:
                var_name = match.group(1)
                if not var_name.isupper():
                    self.uninitialized_vars.add(var_name)

    def _detect_magic_numbers(self, code: str) -> None:
        lines = code.split('\n')
        for i, line in enumerate(lines):
            numbers = re.findall(r'(?<![\w.])\b(\d+)\b(?![\w.])', line)
            for num in numbers:
                if num not in ('0', '1', '2', '100', '1000'):
                    self.magic_numbers[int(num)].append(('global', i+1))
                    self._add_style_issue(
                        'global',
                        i+1,
                        f"Магическое число: {num}. Рекомендуется использовать именованную константу"
                    )

    def _detect_deprecated(self, code: str) -> None:
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if 'xrange(' in line:
                self.deprecated_constructs.append(('global', i+1))
                self._add_style_issue(
                    'global',
                    i+1,
                    "Использование xrange() устарело в Python 3, используйте range()"
                )
            elif '.iteritems()' in line:
                self.deprecated_constructs.append(('global', i+1))
                self._add_style_issue(
                    'global',
                    i+1,
                    "Использование .iteritems() устарело, используйте .items()"
                )

    def _update_complexity(self, func_name: str) -> None:
        result = self.results[func_name]
        
        if self.nested_loops > 1:
            result.time_complexity = f"O(n^{self.nested_loops})"
        
        if self.has_recursion:
            result.time_complexity = f"O(2^n)" if self.recursion_depth > 3 else "O(n)"
            result.space_complexity = f"O(n)"
        
        if self.has_dp:
            result.time_complexity = "O(n)"
            result.space_complexity = "O(n)"
        
        if self.list_comprehensions:
            result.time_complexity = "O(n)"
            if self.nested_loops > 0:
                result.time_complexity = f"O(n^{self.nested_loops + 1})"

    def _detect_techniques(self, func_name: str) -> None:
        result = self.results[func_name]
        
        if "Бинарный поиск" in result.algorithms:
            result.techniques.add("Двоичный поиск")
        
        if any(ds in {'heapq', 'deque'} for ds in result.data_structures):
            result.techniques.add("Жадные алгоритмы")
        
        if self.has_recursion and self.has_dp:
            result.techniques.add("Динамическое программирование с мемоизацией")
        elif self.has_recursion:
            result.techniques.add("Рекурсивный подход")
        
        if any(tc.startswith("O(n^2)") for tc in [result.time_complexity]):
            result.techniques.add("Полный перебор")
        
        if "dict" in result.data_structures and result.time_complexity == "O(n)":
            result.techniques.add("Хэширование")
        
        if "itertools" in str(result.algorithms):
            result.techniques.add("Комбинаторные методы")

    def _post_analysis(self) -> None:
        for func, result in self.results.items():
            if result.time_complexity.startswith("O(n^2)"):
                result.optimizations.append("Можно оптимизировать до O(n log n) с помощью сортировки")
            
            if "Рекурсия" in result.techniques and self.recursion_depth > 3:
                result.issues.append("Глубокая рекурсия может вызвать переполнение стека - используйте итеративный подход")
            
            if "list" in result.data_structures and "count" in str(result.algorithms):
                result.optimizations.append("Для частых поисков используйте set вместо list")
            
            if "dict" in result.data_structures and "keys" in str(result.algorithms):
                result.performance_tips.append("Для проверки наличия ключа используйте 'key in dict' вместо 'key in dict.keys()'")
            
            if "list" in result.data_structures and "append" in str(result.algorithms):
                result.performance_tips.append("Для частых добавлений в конец списка используйте collections.deque")

    def _add_potential_bug(self, func: str, line: int, message: str) -> None:
        if func in self.results:
            self.results[func].potential_bugs.append(f"Строка {line}: {message}")
        else:
            if 'global' not in self.results:
                self._init_global_result()
            self.results['global'].potential_bugs.append(f"Строка {line}: {message}")

    def _add_performance_tip(self, func: str, line: int, message: str) -> None:
        if func in self.results:
            self.results[func].performance_tips.append(f"Строка {line}: {message}")
        else:
            if 'global' not in self.results:
                self._init_global_result()
            self.results['global'].performance_tips.append(f"Строка {line}: {message}")

    def _add_style_issue(self, func: str, line: int, message: str) -> None:
        if func in self.results:
            self.results[func].style_issues.append(f"Строка {line}: {message}")
        else:
            if 'global' not in self.results:
                self._init_global_result()
            self.results['global'].style_issues.append(f"Строка {line}: {message}")

    def _init_global_result(self) -> None:
        self.results['global'] = AnalysisResult(
            time_complexity="O(1)",
            space_complexity="O(1)",
            optimizations=[],
            data_structures=set(),
            algorithms=set(),
            techniques=set(),
            issues=[],
            style_issues=[],
            potential_bugs=[],
            performance_tips=[]
        )

    def _generate_report(self) -> str:
        report = []
        report.append(f"{COLORS['info']}=== АНАЛИЗ PYTHON КОДА ДЛЯ СПОРТИВНОГО ПРОГРАММИРОВАНИЯ ==={COLORS['reset']}")
        
        for func, result in self.results.items():
            report.append(f"\n{COLORS['highlight']}Функция: {func}{COLORS['reset']}" if func != 'global' else f"\n{COLORS['highlight']}Глобальный анализ{COLORS['reset']}")
            
            report.append(f"\n{COLORS['info']}Сложность:{COLORS['reset']}")
            report.append(f"- Временная: {self._color_complexity(result.time_complexity)}")
            report.append(f"- Пространственная: {self._color_complexity(result.space_complexity)}")
            
            if result.techniques:
                report.append(f"\n{COLORS['info']}Методы решения:{COLORS['reset']}")
                for tech in result.techniques:
                    report.append(f"- {COLORS['highlight']}{tech}{COLORS['reset']}")
            
            items = []
            for algo in result.algorithms:
                items.append(f"Алгоритм: {COLORS['info']}{algo}{COLORS['reset']}")
            for ds in result.data_structures:
                items.append(f"Структура данных: {COLORS['info']}{ds}{COLORS['reset']}")
            if items:
                report.append(f"\n{COLORS['info']}Использовано:{COLORS['reset']}")
                report.extend(items)
            
            if result.potential_bugs:
                report.append(f"\n{COLORS['error']}Потенциальные ошибки:{COLORS['reset']}")
                for bug in result.potential_bugs:
                    report.append(f"- ⚠ {bug}")
            
            if result.performance_tips:
                report.append(f"\n{COLORS['success']}Советы по производительности:{COLORS['reset']}")
                for tip in result.performance_tips:
                    report.append(f"- ✓ {tip}")
            
            if result.optimizations:
                report.append(f"\n{COLORS['warning']}Возможные оптимизации:{COLORS['reset']}")
                for opt in result.optimizations:
                    report.append(f"- ✎ {opt}")
            
            if result.style_issues:
                report.append(f"\n{COLORS['warning']}Проблемы стиля:{COLORS['reset']}")
                for issue in result.style_issues:
                    report.append(f"- ✎ {issue}")

        report.append(f"\n{COLORS['info']}=== ОБЩИЕ РЕКОМЕНДАЦИИ ДЛЯ PYTHON ==={COLORS['reset']}")
        tips = [
            "Для задач с n > 1e5 используйте алгоритмы O(n) или O(n log n)",
            "Используйте sys.stdin.readline() для быстрого ввода",
            "Избегайте глубокой рекурсии - используйте итеративные решения",
            "Для частых поисков используйте set() или dict() вместо списков",
            "Выносите инварианты циклов за их пределы",
            "Используйте collections.deque для частых операций с обоих концов",
            "Замените магические числа на именованные константы",
            "Используйте itertools для эффективных комбинаторных операций",
            "Для работы с большими числами используйте модуль math",
            "Используйте генераторы для экономии памяти"
        ]
        
        for i, tip in enumerate(tips, 1):
            report.append(f"{i:2}. {tip}")

        return '\n'.join(report)

    def _color_complexity(self, complexity: str) -> str:
        if complexity in {"O(1)", "O(log n)"}:
            return f"{COLORS['success']}{complexity}{COLORS['reset']}"
        elif complexity in {"O(n)", "O(n log n)"}:
            return f"{COLORS['warning']}{complexity}{COLORS['reset']}"
        else:
            return f"{COLORS['error']}{complexity}{COLORS['reset']}"

def analyze_code(file_path: Optional[str] = None, code: Optional[str] = None) -> str:
    analyzer = PythonCodeAnalyzer()
    
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    
    if code:
        return analyzer.analyze(code)
    else:
        return f"{COLORS['error']}Не предоставлен ни файл, ни код для анализа.{COLORS['reset']}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(analyze_code(file_path=sys.argv[1]))
    else:
        print("Введите путь к файлу для анализа или используйте функцию analyze_code() с параметром code.")