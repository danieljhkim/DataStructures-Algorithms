directions = ((0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
directions = ((0, 1), (1, 0), (-1, 0), (0, -1))
R = len(grid)
C = len(grid[0])
memo = [[0] * C for _ in range(R)]

alpha = "abcdefghijklmnopqrstuvwxyz"

0 <= nr < R and 0 <= nc < C

def is_valid(nr, nc, R, C):
    return 0 <= nr < R and 0 <= nc < C

for col in zip(*matrix): 

def find(x):
    if x not in parent:
        parent[x] = x
    else:
        if parent[x] != x:
            parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    rootx = find(x)
    rooty = find(y)
    if rootx != rooty:
        parent[rootx] = parent[rooty]

 dp.cache_clear()