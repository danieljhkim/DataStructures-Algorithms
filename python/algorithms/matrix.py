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
    if not matrix or not matrix[0]:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:

        # Traverse from left to right along the top row
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1

        # Traverse from top to bottom along the right column
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1

        if top <= bottom:
            # Traverse from right to left along the bottom row
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1

        if left <= right:
            # Traverse from bottom to top along the left column
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    return result
