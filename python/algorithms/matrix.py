import heapq
from typing import Optional, List
import random
from collections import Counter, deque, defaultdict, OrderedDict
import math


matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


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
        # Determine the starting point of the diagonal
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
