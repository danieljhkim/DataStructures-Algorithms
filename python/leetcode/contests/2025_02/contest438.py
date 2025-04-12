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

    def hasSameDigits(self, s: str) -> bool:
        arr = []
        for n in s:
            arr.append(int(n))
        while len(arr) > 2:
            arr2 = []
            for i in range(len(arr) - 1):
                f = arr[i]
                s = arr[i + 1]
                t = s + f
                arr2.append(t % 10)
            arr = arr2
        return arr[0] == arr[1]

    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        R = len(grid)
        C = len(grid[0])
        for row in grid:
            heapq.heapify(row)

        @cache
        def dp(r, used):
            if r == R or used == 0:
                return 0
            cur = dp(r + 1, used)
            for i in range(1, used + 1):
                if i > limits[r] or i > C:
                    break
                t = sum(grid[r][:i]) + dp(r + 1, used - i)
                cur = max(t, cur)
            return cur

        return dp(0, k)

    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        heap = []
        for r, row in enumerate(grid):
            for n in row:
                heap.append((-n, r))
        heapq.heapify(heap)
        ans = 0
        while k > 0:
            val, r = heapq.heappop(heap)
            if limits[r] > 0:
                limits[r] -= 1
                ans += -val
                k -= 1
        return ans

    def hasSameDigits(self, s: str) -> bool:
        arr = [int(s[0])]
        N = len(s)
        for n in s[1 : N - 1]:
            arr.append(int(n) * N // 2)
        arr.append(int(s[-1]))
        left = sum(arr)
        ans = (int(s[0]) + left) % 10 == (int(s[-1]) + left) % 10
        return ans


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
