from main import analyze_code
code = """
import sys

def solve():
    n = int(sys.stdin.readline())
    arr = list(map(int, sys.stdin.readline().split()))
    
    # Неоптимальное решение O(n^2)
    for i in range(n):
        for j in range(i+1, n):
            if arr[i] == arr[j]:
                print("YES")
                return
    
    print("NO")

solve()
"""

print(analyze_code(code=code))