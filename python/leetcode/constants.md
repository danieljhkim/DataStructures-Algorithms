directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
ROW = len(grid)
COL = len(grid[0])
memo = [[0] * COL for _ in range(ROW)]
alpha = "abcdefghijklmnopqrstuvwxyz"