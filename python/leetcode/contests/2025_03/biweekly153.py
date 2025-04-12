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

    def reverseDegree(self, s: str) -> int:
        total = 0
        for i, n in enumerate(s):
            total += abs((ord(n) - ord("a")) - 26) * (i + 1)
            print((ord(n) - ord("a")) - 26)
        return total

    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        N = len(s)
        total = s.count("1")

        @cache
        def dp(idx):
            if idx == N:
                return total
            if s[idx] == "1":
                return dp(idx + 1)
            res = total
            i = idx
            cnt = 0
            while i < N and s[i] == "0":
                i += 1
            m = i
            while m < N and s[m] == "1":
                m += 1
                cnt += 1
            r = m
            if m < N and s[m] == "0":
                res = max(dp(m), res)
            else:
                return res
            while r < N and s[r] == "0":
                r += 1
            res = max(r - idx + total - cnt, res, dp(r))
            return res

        return dp(0)

    """"
    (1 + 2 + cnt * k) * (c1 + c2)
    (1 + 2 + 3 + cnt+1 * k) * (c3)
    
    c3 + 2c3 + 3c3 + c3(cnt + 1)*k
    """

    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        N = len(nums)
        nprefix = [nums[0]]
        for i in range(1, N):
            nprefix.append(nprefix[-1] + nums[i])

        @cache
        def dp(idx, cnt):
            if idx == N:
                return 0
            ctotal = 0
            res = inf
            bigcnt = 0
            for i in range(idx, N):
                if bigcnt > (cnt * k):
                    break
                ctotal += cost[i]
                total = (nprefix[i] + k * cnt) * (ctotal)
                out = dp(i + 1, cnt + 1) + total
                if out + total < res:
                    res = out + total
                else:
                    bigcnt += 1
            return res

        res = dp(0, 1)
        dp.cache_clear()
        return res


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
