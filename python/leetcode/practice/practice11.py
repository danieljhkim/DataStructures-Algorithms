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

    # 2021. Brightest Position on Street
    def brightestPosition(self, lights: List[List[int]]) -> int:
        table = SortedDict()
        for pos, r in lights:
            table[pos + r + 1] = table.get(pos + r + 1, 0) - 1
            table[pos - r] = table.get(pos - r, 0) + 1
        best = idx = total = 0
        for k, v in table.items():
            total += v
            if total > best:
                idx = k
                best = total
        return idx


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
