import heapq
from typing import Optional, List
from itertools import accumulate
import random
from collections import Counter, deque, defaultdict, OrderedDict
import math
import bisect
from math import inf
from functools import lru_cache
from sortedcontainers import SortedSet, SortedList, SortedDict


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:

    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        res = []
        R = len(grid)
        C = len(grid[0])
        r = 0
        c = 0
        while r < R:
            while c < C:
                res.append(grid[r][c])
                c += 2
            if c == C + 1:
                c = C - 2
            else:
                c = C - 1
            r += 1
            if r == R:
                return res
            while c >= 0:
                res.append(grid[r][c])
                c -= 2
            if c == -1:
                c = 0
            else:
                c = 1
            r += 1
        return res

    def maximumAmount(self, coins: List[List[int]]) -> int:
        directions = [(0, 1), (1, 0)]
        ROW = len(coins)
        COL = len(coins[0])
        cur = coins[0][0]
        heap = [(-cur, 2, 0, 0)]
        distances = defaultdict(lambda: -inf)
        ans = []
        if coins[0][0] < 0:
            heapq.heappush(heap, (0, 1, 0, 0))
        while heap:
            money, left, r, c = heapq.heappop(heap)
            if r == ROW - 1 and c == COL - 1:
                return -money
            nm = -money
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < ROW and 0 <= nc < COL:
                    ntotal = coins[nr][nc]
                    if (nm + ntotal) > distances[(nr, nc, left)]:
                        distances[(nr, nc, left)] = nm + ntotal
                        heapq.heappush(heap, (-(nm + ntotal), left, nr, nc))
                    if ntotal < 0 and left > 0:
                        if (nm) > distances[(nr, nc, left - 1)]:
                            distances[(nr, nc, left - 1)] = nm + ntotal
                            heapq.heappush(
                                heap,
                                (
                                    -(nm),
                                    (left - 1),
                                    nr,
                                    nc,
                                ),
                            )

    def maximumAmount(self, coins: List[List[int]]) -> int:
        directions = [(0, 1), (1, 0)]
        ROW = len(coins)
        COL = len(coins[0])
        cur = coins[0][0]
        other1 = 0
        if cur < 0:
            other1 = cur
            heap = [(0, 0, 0, other1, 0)]
        else:
            heap = [(-cur, 0, 0, 0, 0)]
        distances = defaultdict(lambda: -inf)
        ans = []
        while heap:
            money, r, c, other, other2 = heapq.heappop(heap)
            if r == ROW - 1 and c == COL - 1:
                if not ans:
                    ans.append(-money)
                elif ans:
                    if -money > max(ans):
                        ans.append(-money)
                    else:
                        return max(ans)
            nm = -money
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < ROW and 0 <= nc < COL:
                    ntotal = coins[nr][nc]
                    if ntotal < 0 and (ntotal < other or ntotal < other2):
                        distances[(nr, nc)] = nm + ntotal
                        small = max(other, other2)
                        heapq.heappush(
                            heap, (-(nm + small), nr, nc, ntotal, min(other, other2))
                        )
                    else:
                        heapq.heappush(heap, (-(nm + ntotal), nr, nc, other, other2))

        return max(ans)

    def maximumAmount(self, coins: List[List[int]]) -> int:
        directions = [(0, 1), (1, 0)]
        ROW = len(coins)
        COL = len(coins[0])
        memo = {}

        def dp(r, c, left):
            if (r, c, left) in memo:
                return memo[(r, c, left)]
            if r == ROW - 1 and c == COL - 1:
                val = coins[r][c]
                return val
            total = 0
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL:
                    val = coins[r][c]
                    if val < 0 and left > 0:
                        total = max(dp(nr, nc, left - 1), total)
                    total = max(dp(nr, nc, left) + val, total)

            memo[(r, c, left)] = total
            return total

        return dp(0, 0, 2)


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
