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

    def maxDifference(self, s: str) -> int:
        # odd - even
        counts = Counter(s)
        evens = []
        odds = []

        for k, v in counts.items():
            if v % 2 == 1:
                odds.append(v)
            else:
                evens.append(v)
        odds.sort(reverse=True)
        evens.sort()
        if evens[0] >= odds[0]:
            return odds[-1] - evens[0]
        return odds[0] - evens[0]

    def maxDistance(self, s: str, k: int) -> int:
        """ ""
        -1 -1 -1 -1 -1 -1 -1 1 1 1
        """
        ups = 0
        sides = 0
        counts = defaultdict(int)
        best = 0

        for i, w in enumerate(s):
            counts[w] += 1
            if w == "N":
                ups += 1
            elif w == "S":
                ups -= 1
            elif w == "E":
                sides += 1
            elif w == "W":
                sides -= 1
            total = abs(ups) + abs(sides)
            changes = 0

            if ups < 0:
                changes += counts["N"]
            elif ups > 0:
                changes += counts["S"]
            else:
                changes += counts["S"]

            if sides < 0:
                changes += counts["E"]
            elif sides > 0:
                changes += counts["W"]
            else:
                changes += counts["W"]

            if total + min(k, changes) * 2 > best:
                best = total + min(k, changes) * 2

        return best

    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        """ "
        target 10
        num 5
        """

        ans = 0
        table = defaultdict(lambda: inf)
        target = set(target)
        ntarget = target.copy()
        target = list(target)
        target.sort()
        for i in range(len(target) - 1):
            n1 = target[i]
            for j in range(i + 1, len(target)):
                n2 = target[j]
                if n2 % n1 == 0:
                    if n1 in ntarget:
                        ntarget.remove(n1)
                        break

        target = list(ntarget)
        for n in nums:
            for i, t in enumerate(target):
                if t >= n:
                    diff2 = min(table[t], n % t, t - n)
                    table[t] = diff2
                else:
                    if (n - t + n) % t == 0:
                        table[t] = min(table[t], n - t)

        return sum(table.values())


def test_solution():
    s = Solution()
    ss = "NWSE"
    s.maxDistance(ss, 1)


if __name__ == "__main__":
    test_solution()
