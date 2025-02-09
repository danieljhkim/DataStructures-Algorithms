import heapq
import random
import math
import bisect
from typing import *
from math import inf, factorial
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
    """
    0 1
    1 2

    """

    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        N = len(grid)
        mat = deque()

        for j in range(1, N):
            line = []
            for i in range(N - j):
                r = i
                c = i + j
                line.append(grid[r][c])
            print("1", line)
            line.sort()
            mat.extend(line)

        for j in range(1, N):
            for i in range(N - j):
                r = i
                c = i + j
                grid[r][c] = mat.popleft()

        dq = deque()
        for j in range(N - 1, -1, -1):
            line = []
            for i in range(N - j):
                r = i + j
                c = i
                line.append(grid[r][c])
            print("2", line)
            line.sort(reverse=True)
            dq.extend(line)

        for j in range(N - 1, -1, -1):
            for i in range(N - j):
                r = i + j
                c = i
                grid[r][c] = dq.popleft()
        return grid

    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:

        def is_prime(n):
            if n <= 1:
                return False
            if n <= 3:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True

        N = len(groups)
        table = defaultdict(list)
        etable = {}
        res = [-1] * N
        options = []
        primes = defaultdict(list)
        for i, n in enumerate(groups):
            if is_prime(n):
                primes[n].append(i)
                continue
            table[n].append(i)

        options = []
        eprimes = {}
        for i, n in enumerate(elements):
            if n in primes and n not in eprimes:
                eprimes[n] = i
                continue
            if n not in etable:
                etable[n] = i
                options.append((i, n))

        N2 = len(options)
        options.sort()

        def find(num, idx):
            if idx == N2:
                return -1
            i, n = options[idx]
            res = -1
            if num % n == 0:
                res = i
                return res
            else:
                res = find(num, idx + 1)
            return res

        for k, v in table.items():
            idx = find(k, 0)
            if idx != -1:
                for i in v:
                    res[i] = idx
        for k, v in eprimes.items():
            for i in primes[k]:
                res[i] = v
        return res


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
