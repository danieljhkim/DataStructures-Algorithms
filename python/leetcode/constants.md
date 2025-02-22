directions = [(-1, 1), (1, -1), (1, 1), (-1, -1)]

directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
ROW = len(grid)
COL = len(grid[0])
memo = [[0] * COL for _ in range(ROW)]

alpha = "abcdefghijklmnopqrstuvwxyz"

0 <= nr < ROW and 0 <= nc < COL

def is_valid(nr, nc, ROW, COL):
    return 0 <= nr < ROW and 0 <= nc < COL

for col in zip(*matrix): 

