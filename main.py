import ast
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import colorama
from colorama import Fore, Style

colorama.init()

@dataclass
class AnalysisResult:
    time_complexity: str
    space_complexity: str
    optimizations: List[str]
    data_structures: Set[str]
    algorithms: Set[str]
    techniques: Set[str]
    issues: List[str]

class CompetitiveProgrammingAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.results: Dict[str, AnalysisResult] = {}
        self.current_function: Optional[str] = None
        self.nested_loops: int = 0
        self.recursion_depth: int = 0
        self.techniques: Set[str] = set()
        self.tree_depth: int = 0
        self.has_recursion: bool = False
        self.has_dp: bool = False
        self.has_bitmask: bool = False
        self.has_graph: bool = False

    def analyze(self, code: str) -> None:
        try:
            tree = ast.parse(code)
            self.visit(tree)
            self._post_analysis()
            self._print_pretty_report()
        except SyntaxError as e:
            print(f"{Fore.RED}Ошибка синтаксиса: {e}{Style.RESET_ALL}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.current_function = node.name
        self.results[node.name] = AnalysisResult(
            time_complexity="O(1)",
            space_complexity="O(1)",
            optimizations=[],
            data_structures=set(),
            algorithms=set(),
            techniques=set(),
            issues=[]
        )
        self.nested_loops = 0
        self.recursion_depth = 0
        
        self.generic_visit(node)
        
        self._update_complexity(node.name)
        self._detect_techniques(node.name)
        self.current_function = None

    def visit_For(self, node: ast.For) -> None:
        if self.current_function:
            self.nested_loops += 1
            self._analyze_loop(node)
            self.generic_visit(node)
            self.nested_loops -= 1

    def visit_While(self, node: ast.While) -> None:
        if self.current_function:
            self.nested_loops += 1
            self.results[self.current_function].time_complexity = "O(n)"
            self.generic_visit(node)
            self.nested_loops -= 1

    def visit_Call(self, node: ast.Call) -> None:
        if not self.current_function:
            self.generic_visit(node)
            return

        try:
            if isinstance(node.func, ast.Name) and node.func.id == self.current_function:
                self.has_recursion = True
                self.recursion_depth += 1
                self.results[self.current_function].techniques.add("Рекурсия")

            self._detect_algorithms(node)

            self._detect_data_structures(node)

        except AttributeError:
            pass
            
        self.generic_visit(node)

    def _detect_algorithms(self, node: ast.Call) -> None:
        """Определение используемых алгоритмов"""
        func_mapping = {
            'bisect_left': "Бинарный поиск",
            'bisect_right': "Бинарный поиск",
            'gcd': "Алгоритм Евклида",
            'lru_cache': "Динамическое программирование",
            'heappush': "Куча",
            'heappop': "Куча",
            'dfs': "Поиск в глубину",
            'bfs': "Поиск в ширину",
            'dijkstra': "Алгоритм Дейкстры"
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
                
                if hasattr(node.func, 'value') and isinstance(node.func.value, ast.Name):
                    full_name = f"{node.func.value.id}.{node.func.attr}"
                    if full_name in func_mapping:
                        self.results[self.current_function].algorithms.add(func_mapping[full_name])
                        
        except AttributeError:
            pass

    def _detect_data_structures(self, node: ast.Call) -> None:
        """Определение используемых структур данных"""
        ds_mapping = {
            'append': 'list',
            'pop': 'list',
            'add': 'set',
            'remove': 'set',
            'keys': 'dict',
            'values': 'dict',
            'items': 'dict',
            'popleft': 'deque',
            'appendleft': 'deque'
        }

        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ds_mapping:
                self.results[self.current_function].data_structures.add(ds_mapping[node.func.attr])

    def _analyze_loop(self, node: ast.For) -> None:
        """Анализ циклов для определения сложности"""
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
            if node.iter.func.id == 'range':
                if len(node.iter.args) == 1:
                    self.results[self.current_function].time_complexity = "O(n)"
                elif len(node.iter.args) == 2:
                    self.results[self.current_function].time_complexity = "O(n)"
                
                if self.nested_loops > 1:
                    self.results[self.current_function].time_complexity = f"O(n^{self.nested_loops})"

    def _update_complexity(self, func_name: str) -> None:
        """Обновление метрик сложности"""
        if self.has_recursion:
            self.results[func_name].time_complexity = f"O(2^n)" if self.recursion_depth > 3 else "O(n)"
            self.results[func_name].space_complexity = f"O(n)" 

        if self.has_dp:
            self.results[func_name].time_complexity = "O(n)"
            self.results[func_name].space_complexity = "O(n)"

    def _detect_techniques(self, func_name: str) -> None:
        """Определение методов решения"""
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

    def _post_analysis(self) -> None:
        """Пост-анализ для рекомендаций"""
        for func, result in self.results.items():
            if result.time_complexity.startswith("O(n^2)"):
                result.optimizations.append("Можно оптимизировать до O(n log n) с помощью сортировки")
            
            if "Рекурсия" in result.techniques and self.recursion_depth > 3:
                result.issues.append("Глубокая рекурсия может вызвать переполнение стека")
            
            if "list" in result.data_structures and "count" in dir(ast):
                result.optimizations.append("Для частых поисков используйте set вместо list")

    def _print_pretty_report(self) -> None:
        """Красивый цветной вывод отчета"""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}=== АНАЛИЗ КОДА ДЛЯ СПОРТИВНОГО ПРОГРАММИРОВАНИЯ ==={Style.RESET_ALL}")
        
        for func, result in self.results.items():
            print(f"\n{Fore.YELLOW}Функция: {Fore.WHITE}{func}{Style.RESET_ALL}")
            
            # Сложность
            print(f"  {Fore.GREEN}Сложность:{Style.RESET_ALL}")
            print(f"  - Временная: {self._color_complexity(result.time_complexity)}")
            print(f"  - Пространственная: {self._color_complexity(result.space_complexity)}")
            
            # Методы решения
            if result.techniques:
                print(f"  {Fore.GREEN}Методы решения:{Style.RESET_ALL}")
                for tech in result.techniques:
                    print(f"  - {Fore.MAGENTA}{tech}{Style.RESET_ALL}")
            
            # Алгоритмы и структуры данных
            if result.algorithms or result.data_structures:
                print(f"  {Fore.GREEN}Использовано:{Style.RESET_ALL}")
                for algo in result.algorithms:
                    print(f"  - Алгоритм: {Fore.BLUE}{algo}{Style.RESET_ALL}")
                for ds in result.data_structures:
                    print(f"  - Структура данных: {Fore.BLUE}{ds}{Style.RESET_ALL}")
            
            # Рекомендации
            if result.optimizations:
                print(f"  {Fore.GREEN}Рекомендации по оптимизации:{Style.RESET_ALL}")
                for opt in result.optimizations:
                    print(f"  - {Fore.CYAN}✓ {opt}{Style.RESET_ALL}")
            
            # Проблемы
            if result.issues:
                print(f"  {Fore.GREEN}Потенциальные проблемы:{Style.RESET_ALL}")
                for issue in result.issues:
                    print(f"  - {Fore.RED}⚠ {issue}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}{Style.BRIGHT}=== ОБЩИЕ РЕКОМЕНДАЦИИ ==={Style.RESET_ALL}")
        print(f"{Fore.WHITE}- Для задач с n > 1e5 используйте алгоритмы O(n) или O(n log n)")
        print("- Используйте sys.stdin для быстрого ввода данных")
        print("- Для рекурсивных решений проверяйте глубину рекурсии")
        print(f"- Для задач на графы используйте эффективные представления (списки смежности){Style.RESET_ALL}")

    def _color_complexity(self, complexity: str) -> str:
        """Цветовое выделение сложности"""
        if complexity in {"O(1)", "O(log n)"}:
            return f"{Fore.GREEN}{complexity}{Style.RESET_ALL}"
        elif complexity in {"O(n)", "O(n log n)"}:
            return f"{Fore.YELLOW}{complexity}{Style.RESET_ALL}"
        else:
            return f"{Fore.RED}{complexity}{Style.RESET_ALL}"


def analyze_code(file_path: Optional[str] = None, code: Optional[str] = None) -> None:
    analyzer = CompetitiveProgrammingAnalyzer()
    
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    
    if code:
        analyzer.analyze(code)
    else:
        print(f"{Fore.RED}Не предоставлен ни файл, ни код для анализа.{Style.RESET_ALL}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_code(file_path=sys.argv[1])
    else:
        print("Введите путь к файлу для анализа или используйте функцию analyze_code() с параметром code.")