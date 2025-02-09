from typing import Optional, List

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

############## operations ##############


def reverse_rows(matrix):
    h = len(matrix)
    w = len(matrix[0])
    for row in range(h):
        for col in range(w // 2):
            matrix[row][col], matrix[row][w - col - 1] = (
                matrix[row][w - col - 1],
                matrix[row][col],
            )
    print(matrix)


def reverse_rows_2(matrix):
    for i in range(len(matrix)):
        matrix[i].reverse()
    print(matrix)


def reverse_col(matrix):
    h = len(matrix)
    w = len(matrix[0])
    for col in range(w):
        for row in range(h // 2):
            matrix[row][col], matrix[h - row - 1][col] = (
                matrix[h - row - 1][col],
                matrix[row][col],
            )
    print(matrix)


def transpose(matrix):
    n = len(matrix)
    for row in range(n):
        for col in range(row, n):
            matrix[row][col], matrix[col][row] = matrix[col][row], matrix[row][col]
    print(matrix)


############## rotates ##############

"""starting
    1 2 3
    4 5 6
    7 8 9
"""


def rotate_90_right(matrix):
    N = len(matrix)

    """ transpose
    1 4 7
    2 5 8
    3 6 9
    """
    for r in range(N):
        for c in range(r, N):
            matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]

    """ reverse reach row
    7 4 1
    8 5 2
    9 6 3
    """
    for row in matrix:
        row.reverse()


def rotate_90_left(matrix):
    transpose(matrix)
    reverse_col(matrix)


def rotate_180_right(matrix):
    reverse_rows(matrix)
    reverse_col(matrix)


############## traversal ##############


def diagonal_traversal(matrix):
    """
    bottom-left to top-right, left to right, diagnonally
    7, 4, 8, 1, 5, 9, 2, 6, 3

    [1, 2, 3]
    [4, 5, 6]
    [7, 8, 9]
    """
    R = len(matrix)
    C = len(matrix[0])
    res = []

    for start in range(R - 1, -1, -1):
        r, c = start, 0
        while r < R and c < C:
            res.append(matrix[r][c])
            r += 1
            c += 1

    for start in range(1, C):
        r, c = 0, start
        while r < R and c < C:
            res.append(matrix[r][c])
            r += 1
            c += 1

    return res


def diagonal_weaving_traverse(matrix: List[List[int]]) -> List[int]:
    """ "
    1, 2, 4, 7, 5, 3, 6, 8, 9

    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
    """
    if not matrix or not matrix[0]:
        return []
    ROW, COL = len(matrix), len(matrix[0])
    result = []
    intermediate = []

    for d in range(ROW + COL - 1):
        intermediate.clear()
        r = 0 if d < COL else d - COL + 1
        c = d if d < COL else COL - 1

        while r < ROW and c >= 0:
            intermediate.append(matrix[r][c])
            r += 1
            c -= 1

        if d % 2 == 0:  # for weaving
            result.extend(intermediate[::-1])
        else:
            result.extend(intermediate)

    return result


def spiralOrder(matrix: List[List[int]]) -> List[int]:
    row = len(matrix)
    col = len(matrix[0])
    dir = 1
    r = 0
    c = -1
    res = []
    while row * col > 0:
        for _ in range(col):  # hortizontal
            c += dir
            res.append(matrix[r][c])
        row -= 1
        for _ in range(row):  # vertical
            r += dir
            res.append(matrix[r][c])
        col -= 1
        dir *= -1
    return res


def spiralOrder(matrix: List[List[int]]) -> List[int]:

    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    rows, cols = len(matrix), len(matrix[0])

    def dfs(r, c, direction):
        seen.add((r, c))
        traverse.append(matrix[r][c])
        for i in range(4):
            new_direction = (direction + i) % 4
            nr = r + dirs[new_direction][0]
            nc = c + dirs[new_direction][1]
            if nr < 0 or nc < 0 or nr >= rows or nc >= cols or (nr, nc) in seen:
                continue
            dfs(nr, nc, new_direction)

    seen = set()
    traverse = []
    dfs(0, 0, 1)
    return traverse
