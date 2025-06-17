from main import analyze_code
code = """
import sys

def solve():
    n = int(sys.stdin.readline())
    return(n)

solve()
"""

print(analyze_code(code=code))