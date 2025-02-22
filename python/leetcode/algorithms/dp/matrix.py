"""Good DP problems
"""

import heapq
import random
import math
import bisect
from typing import *
from math import inf, factorial, gcd, lcm
from functools import lru_cache, cache
from heapq import heapify, heappush, heappop
from itertools import accumulate, permutations, combinations
from collections import Counter, deque, defaultdict, OrderedDict
from sortedcontainers import SortedSet, SortedList, SortedDict


class Solution:

    # 3459. Length of Longest V-Shaped Diagonal Segment
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        R = len(grid)
        C = len(grid[0])
        directions = {
            (-1, 1): (1, 1),
            (1, -1): (-1, -1),
            (1, 1): (1, -1),
            (-1, -1): (-1, 1),
        }

        @cache
        def dp(r, c, turned, dir):
            res = 0
            cur = grid[r][c]
            nxt = 2
            if cur == 2:
                nxt = 0
            nr = dir[0] + r
            nc = dir[1] + c
            if 0 <= nr < R and 0 <= nc < C:
                if nxt == grid[nr][nc]:
                    res = dp(nr, nc, turned, dir)
            if not turned:
                dirs = directions[dir]
                nr = dirs[0] + r
                nc = dirs[1] + c
                if 0 <= nr < R and 0 <= nc < C:
                    if nxt == grid[nr][nc]:
                        res = max(dp(nr, nc, True, dirs), res)

            return res + 1

        res = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    for k in directions.keys():
                        res = max(res, dp(r, c, False, k))
        return res

    # 931 Minimum Falling Path Sum
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        N = len(matrix)
        directions = ((1, 0), (1, 1), (1, -1))

        @cache
        def dp(r, c):
            if r == N - 1:
                return matrix[r][c]
            res = inf
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < N and 0 <= nc < N:
                    res = min(res, dp(nr, nc))
            return res + matrix[r][c]

        best = inf
        for c in range(N):
            best = min(dp(0, c), best)
        return best
