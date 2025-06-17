import ast
import sys
from collections import defaultdict
from math import log
from typing import Dict, List, Set, Tuple, Optional


class CompetitiveProgrammingAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.issues: Dict[str, List[str]] = defaultdict(list)
        self.function_complexity: Dict[str, int] = {}
        self.current_function: Optional[str] = None
        self.time_complexity: Dict[str, str] = {}
        self.space_complexity: Dict[str, str] = {}
        self.nested_loops: Dict[str, int] = defaultdict(int)
        self.loop_depths: Dict[str, int] = defaultdict(int)
        self.max_loop_depth: int = 0
        self.expensive_operations: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.possible_optimizations: Dict[str, List[str]] = defaultdict(list)
        self.input_output: Dict[str, List[str]] = defaultdict(list)
        self.data_structures: Dict[str, Set[str]] = defaultdict(set)
        self.algorithms_used: Dict[str, Set[str]] = defaultdict(set)
        self.constraints: Dict[str, Tuple[Optional[int], Optional[int]]] = {}

    def analyze(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            self.visit(tree)
            self.post_analysis()
            return self.generate_report()
        except SyntaxError as e:
            return f"Ошибка синтаксиса: {e}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.current_function = node.name
        self.function_complexity[node.name] = 1
        self.nested_loops[node.name] = 0
        self.loop_depths[node.name] = 0
        self.time_complexity[node.name] = "O(1)"
        self.space_complexity[node.name] = "O(1)"
        
        self.generic_visit(node)
        
        self._update_complexity_metrics(node.name)
        self.current_function = None

    def visit_For(self, node: ast.For) -> None:
        if self.current_function:
            self.nested_loops[self.current_function] += 1
            self.loop_depths[self.current_function] += 1
            self.max_loop_depth = max(self.max_loop_depth, self.loop_depths[self.current_function])
            
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == 'range':
                    self._analyze_range_loop(node.iter)
            
            self.generic_visit(node)
            self.loop_depths[self.current_function] -= 1

    def _analyze_range_loop(self, node: ast.Call) -> None:
        """Анализ циклов for с range() для определения сложности"""
        if len(node.args) == 1:
            arg = node.args[0]
            if isinstance(arg, ast.Name) and arg.id in self.constraints:
                self.time_complexity[self.current_function] = f"O({self.constraints[arg.id][0]})"
            else:
                self.time_complexity[self.current_function] = "O(n)"
        elif len(node.args) == 2:
            self.time_complexity[self.current_function] = "O(n)"

    def visit_While(self, node: ast.While) -> None:
        if self.current_function:
            self.nested_loops[self.current_function] += 1
            self.loop_depths[self.current_function] += 1
            self.max_loop_depth = max(self.max_loop_depth, self.loop_depths[self.current_function])
            self.time_complexity[self.current_function] = "O(n)" 
            self.generic_visit(node)
            self.loop_depths[self.current_function] -= 1

    def visit_Call(self, node: ast.Call) -> None:
        if self.current_function:
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                
                if func_name in ('sorted', 'list.sort', 'min', 'max'):
                    self.expensive_operations[self.current_function].append(
                        (f"Использование {func_name}", node.lineno)
                    )
                    self.time_complexity[self.current_function] = "O(n log n)"
                
                if func_name in ('bisect_left', 'bisect_right'):
                    self.algorithms_used[self.current_function].add("Бинарный поиск")
                elif func_name == 'gcd':
                    self.algorithms_used[self.current_function].add("Алгоритм Евклида")
                
                if func_name in ('input', 'print'):
                    self.input_output[self.current_function].append(func_name)
            
            elif isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                if attr_name in ('append', 'pop'):
                    self.data_structures[self.current_function].add("list")
                elif attr_name in ('add', 'remove', 'union', 'intersection'):
                    self.data_structures[self.current_function].add("set")
                elif attr_name in ('keys', 'values', 'items'):
                    self.data_structures[self.current_function].add("dict")
                
                if attr_name == 'sort':
                    self.expensive_operations[self.current_function].append(
                        ("Сортировка списка", node.lineno))
                    self.time_complexity[self.current_function] = "O(n log n)"
                elif attr_name == 'count':
                    self.expensive_operations[self.current_function].append(
                        ("Подсчет элементов", node.lineno))
                    self.time_complexity[self.current_function] = "O(n)"
        
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        if self.current_function:
            self.time_complexity[self.current_function] = "O(n)"
            self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if self.current_function:
            self.function_complexity[self.current_function] += 1
            self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if self.current_function and isinstance(node.op, ast.Pow):
            self.expensive_operations[self.current_function].append(
                ("Возведение в степень", node.lineno))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if (len(node.targets) == 1 and 
            isinstance(node.targets[0], ast.Name) and 
            isinstance(node.value, ast.Constant)):
            var_name = node.targets[0].id
            if isinstance(node.value.value, (int, float)):
                self.constraints[var_name] = (int(node.value.value), None)
        
        self.generic_visit(node)

    def _update_complexity_metrics(self, func_name: str) -> None:
        """Обновление метрик сложности после анализа функции"""
        if self.nested_loops[func_name] > 1:
            self.time_complexity[func_name] = f"O(n^{self.nested_loops[func_name]})"
        
        if "set" in self.data_structures[func_name] or "dict" in self.data_structures[func_name]:
            self.space_complexity[func_name] = "O(n)"
        elif "list" in self.data_structures[func_name]:
            self.space_complexity[func_name] = "O(n)"

    def post_analysis(self) -> None:
        """Пост-анализ для выявления возможных оптимизаций"""
        for func, ops in self.expensive_operations.items():
            for op, lineno in ops:
                if "сортировка" in op.lower():
                    self.possible_optimizations[func].append(
                        f"Строка {lineno}: Используйте counting sort или radix sort для малых диапазонов значений")
                elif "подсчет" in op.lower():
                    self.possible_optimizations[func].append(
                        f"Строка {lineno}: Используйте collections.Counter для эффективного подсчета")
                elif "возведение" in op.lower():
                    self.possible_optimizations[func].append(
                        f"Строка {lineno}: Используйте бинарное возведение в степень для больших степеней")
        
        for func, io_ops in self.input_output.items():
            if len(io_ops) > 3:
                self.possible_optimizations[func].append(
                    "Используйте sys.stdin.read() для быстрого ввода больших данных")
                self.possible_optimizations[func].append(
                    "Используйте sys.stdout.write() вместо print() для быстрого вывода")

    def generate_report(self) -> str:
        report = ["Анализ кода для спортивного программирования"]
        report.append("=" * 50)
        
        # Отчет по сложности
        report.append("\n=== Анализ сложности ===")
        for func, tc in self.time_complexity.items():
            report.append(f"Функция {func}:")
            report.append(f"  - Временная сложность: {tc}")
            report.append(f"  - Пространственная сложность: {self.space_complexity[func]}")
            if self.nested_loops[func] > 2:
                report.append(f"  - Внимание: {self.nested_loops[func]} вложенных цикла!")
        
        # Отчет по оптимизациям
        if any(self.possible_optimizations.values()):
            report.append("\n=== Возможные оптимизации ===")
            for func, optimizations in self.possible_optimizations.items():
                if optimizations:
                    report.append(f"Функция {func}:")
                    for opt in optimizations:
                        report.append(f"  - {opt}")
        
        # Отчет по структурам данных и алгоритмам
        report.append("\n=== Используемые структуры данных и алгоритмы ===")
        for func, ds in self.data_structures.items():
            if ds:
                report.append(f"Функция {func}:")
                report.append(f"  - Структуры данных: {', '.join(ds)}")
            if self.algorithms_used[func]:
                report.append(f"  - Алгоритмы: {', '.join(self.algorithms_used[func])}")
        
        # Общие рекомендации
        report.append("\n=== Общие рекомендации ===")
        if self.max_loop_depth > 3:
            report.append("- Слишком глубокая вложенность циклов. Попробуйте уменьшить.")
        
        if any(tc.startswith("O(n^") for tc in self.time_complexity.values()):
            report.append("- Обнаружены алгоритмы с полиномиальной сложностью. Возможно, есть более оптимальное решение.")
        
        if any("O(n log n)" in tc for tc in self.time_complexity.values()):
            report.append("- Для алгоритмов O(n log n) убедитесь, что n не превышает 1e5-1e6.")
        
        report.append("- Для больших данных (n > 1e6) используйте алгоритмы O(n) или O(1).")
        report.append("- Используйте sys.stdin.read() для быстрого ввода больших данных.")
        report.append("- Для частых запросов используйте префиксные суммы или бинарное дерево отрезков.")
        
        return "\n".join(report)


def analyze_code(file_path: Optional[str] = None, code: Optional[str] = None) -> str:
    analyzer = CompetitiveProgrammingAnalyzer()
    
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
    
    if code:
        return analyzer.analyze(code)
    else:
        return "Не предоставлен ни файл, ни код для анализа."


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(analyze_code(file_path=sys.argv[1]))
    else:
        print("Введите путь к файлу для анализа или используйте функцию analyze_code() с параметром code.")