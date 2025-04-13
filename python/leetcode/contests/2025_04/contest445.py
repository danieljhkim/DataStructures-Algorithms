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

    def findClosest(self, x: int, y: int, z: int) -> int:
        first = abs(z - x)
        second = abs(z - y)
        if first == second:
            return 0
        if first < second:
            return 1
        return 2

    def smallestPalindrome(self, s: str) -> str:
        N = len(s)
        alpha = "abcdefghijklmnopqrstuvwxyz"
        counts = Counter(s)
        res = [""] * N
        left, right = 0, N - 1
        mid = N // 2
        is_odd = (N % 2) == 1
        while left <= right:
            for a in alpha:
                if a in counts:
                    if counts[a] >= 2:
                        res[left] = a
                        res[right] = a
                        counts[a] -= 2
                        left += 1
                        right -= 1
                        break
            if is_odd and left == right:
                break
        if is_odd:
            for k, v in counts.items():
                if v == 1:
                    res[mid] = k
                    break
        return "".join(res)

    def smallestPalindrome(self, s: str, k: int) -> str:  # MLE
        N = len(s)
        if N == 1 and k == 1:
            return s
        counts, cands = Counter(s), []
        mid, half, total = None, N // 2, 1

        for kk, v in counts.items():
            if v > 1:
                h = v // 2
                total *= half * h
                diff = v % 2
                if diff == 1:
                    mid = kk
                cands.extend([kk] * (v // 2))
            elif v == 1:
                mid = kk

        if total + 1 < k:
            return ""
        cands.sort()
        perms = list(permutations(cands))
        nset = set()
        ans = None
        for p in perms:
            nset.add(p)
            if len(nset) == k:
                ans = "".join(p)
                break
        if not ans:
            return ""
        res = ans
        rev = res[::-1]
        if mid:
            res += mid
        res += rev
        return res


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
