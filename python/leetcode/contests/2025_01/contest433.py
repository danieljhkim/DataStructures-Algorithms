import heapq
import random
import math
import bisect
from typing import *
from math import inf, factorial
from functools import lru_cache, cache
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

    def subarraySum(self, nums: List[int]) -> int:
        res = []

        for i, n in enumerate(nums):
            start = max(0, i - n)
            total = 0
            for j in range(start, i + 1):
                total += nums[j]
            res.append(total)
        return sum(res)

    def minMaxSums(self, nums: List[int], k: int) -> int:
        """ "
        5 * 4 +
        5 * 3
        5 * 2

        5 (4 + 3 + 2 + 1)
        4 * 3

        2
        """

        def score(n, size, kk):
            total = 0
            for j in range(kk + 1):
                tem = 0
                for i in range(size, j, -1):
                    tem += i
                total *= n
            return total

        MOD = 10**9 + 7
        nums.sort()
        total = 0

        N = len(nums)
        for i, n in enumerate(nums):
            total += score(n, N - i - 1, i) % MOD
        for i, n in enumerate(reversed(nums)):
            total += score(n, N - i - 1) % MOD
        total += sum(nums) * 2 % MOD
        return total % MOD

    def minCost(self, n: int, cost: List[List[int]]) -> int:
        """ "
        cost = n x 3
        cost[i][j] = cost of paiting ouse i with color j + 1
        cost = [c1, c2, c3]

        no two adjacent same color
        no same color equadist -> h0 + h5 == h1 + h4

        c1, c2, c1, c2 c1
        1   2   3   4  5  6
        x   y   x   y  x  y

        middle = cheap
        """
        nn = n // 2
        idx = 0
        right = n - 1
        while idx <= nn:
            leftc = cost[idx]
            rightc = cost[right]


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
