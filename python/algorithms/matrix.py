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


class Solutions:

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        ans = []
        r = 0
        c = 0
        R = len(matrix)
        C = len(matrix[0])
        while matrix and len(matrix) > 0:
            # right
            while matrix and c < C:
                num = matrix[r][c]
                if num is not None:
                    ans.append(num)
                    matrix[r][c] = None
                c += 1
            if not matrix:
                return ans
            matrix.pop(0)
            c = C - 1
            r = len(matrix) - 1
            # down
            while matrix and r < len(matrix):
                num = matrix[r][c]
                if num is not None:
                    ans.append(num)
                    matrix[r][c] = None
                r += 1
            r = len(matrix) - 1
            # left
            while matrix and c >= 0:
                num = matrix[r][c]
                if num is not None:
                    ans.append(num)
                    matrix[r][c] = None
                c -= 1
            if not matrix:
                return ans
            matrix.pop(-1)
            c = 0
            r = len(matrix) - 1
            # up
            while matrix and r >= 0:
                num = matrix[r][c]
                if num is not None:
                    ans.append(num)
                    matrix[r][c] = None
                r -= 1
            r = 0
        return ans
