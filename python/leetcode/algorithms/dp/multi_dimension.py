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

    # 1092. Shortest Common Supersequence
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        N1, N2 = len(str1), len(str2)

        @lru_cache(maxsize=None)
        def find_occurrences(sub, start):
            indices = []
            pos = str2.find(sub, start, N2)
            while pos != -1:
                indices.append(pos)
                pos = str2.find(sub, pos + len(sub), N2)
            return tuple(indices)

        @cache
        def dp(idx1, idx2):
            if idx1 == N1 and idx2 == N2:
                return 0, ""
            elif idx1 == N1:
                return N2 - idx2, str2[idx2:]
            elif idx2 == N2:
                return N1 - idx1, str1[idx1:]

            res = inf
            res_str = ""
            for i in range(idx1, N1):
                cur = str1[idx1 : i + 1]
                diff = i - idx1 + 1
                for found in find_occurrences(cur, idx2):
                    res2, res2_str = dp(i + 1, found + diff)
                    res2 += found - idx2 + diff
                    if res2 < res:
                        res_str = str2[idx2:found] + cur + res2_str
                        res = res2

            res3, res3_str = dp(idx1 + 1, idx2)
            res3 += 1
            if res3 < res:
                res_str = str1[idx1] + res3_str
                res = res3
            return res, res_str

        res, ans = dp(0, 0)
        return ans

    # 1092. Shortest Common Supersequence
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        N1, N2 = len(str1), len(str2)

        @cache
        def dp(i, j):
            if i == N1:
                return str2[j:]
            if j == N2:
                return str1[i:]
            if str1[i] == str2[j]:
                return str1[i] + dp(i + 1, j + 1)
            else:
                candidate1 = str1[i] + dp(i + 1, j)
                candidate2 = str2[j] + dp(i, j + 1)
                return candidate1 if len(candidate1) <= len(candidate2) else candidate2

        return dp(0, 0)

    # 1092. Shortest Common Supersequence
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        N1, N2 = len(str1), len(str2)

        @cache
        def dp(i, j):
            if i == N1:
                return N2 - j
            if j == N2:
                return N1 - i
            if str1[i] == str2[j]:
                return 1 + dp(i + 1, j + 1)
            else:
                return 1 + min(dp(i + 1, j), dp(i, j + 1))

        @cache
        def rec(i, j):
            if i == N1:
                return str2[j:]
            if j == N2:
                return str1[i:]
            if str1[i] == str2[j]:
                return str1[i] + rec(i + 1, j + 1)
            else:
                if dp(i + 1, j) < dp(i, j + 1):
                    return str1[i] + rec(i + 1, j)
                else:
                    return str2[j] + rec(i, j + 1)

        return rec(0, 0)
