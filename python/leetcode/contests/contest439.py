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

    def largestInteger(self, nums: List[int], k: int) -> int:
        cand = defaultdict(int)
        table = Counter(nums[0:k])
        for n, v in table.items():
            if v >= 1:
                cand[n] += 1
        left = 0

        for right in range(k, len(nums)):
            table[nums[left]] -= 1
            table[nums[right]] += 1
            for n, v in table.items():
                if v >= 1:
                    cand[n] += 1
            left += 1

        ans = []
        for n, v in cand.items():
            if v == 1:
                ans.append(n)
        ans.sort()
        if not ans:
            return -1
        return ans[-1]

    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        N = len(s)

        @cache
        def dp(left, right, n):
            if left == right:
                return 1
            if left > right:
                return 0
            cnt = 0
            if s[left] == s[right]:

                cnt = dp(left + 1, right - 1, n) + 2
            else:
                cnt = max(dp(left + 1, right, n), dp(left, right - 1, n))
                if n > 0:
                    lw = ord(s[left]) - ord("a")
                    rw = ord(s[right]) - ord("a")
                    diff1 = abs(lw - rw)
                    diff2 = (26 - max(lw, rw) + min(lw, rw)) % 26
                    if diff1 <= n:
                        cnt = max(cnt, dp(left + 1, right - 1, n - diff1) + 2)
                    if diff2 <= n:
                        cnt = max(cnt, dp(left + 1, right - 1, n - diff2) + 2)
            return cnt

        return dp(0, N - 1, k)

    def maxSum(self, nums: List[int], k: int, m: int) -> int:
        N = len(nums)
        S = N // k

        @cache
        def dp(idx, n):
            if n == 0:
                return 0
            if idx > N - (m * (n)):
                return -inf
            if n == 1:
                return sum(nums[idx:])
            total = sum(nums[idx : idx + m])
            res = dp(idx + m, n - 1) + total
            t = total
            for i in range(1, S - m + 1):
                if idx + i >= N:
                    break
                t += nums[idx + i]
                res = max(dp(idx + m + i, n - 1) + t, res)
            return res

        return dp(0, k)


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
