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

    # 3494. Find the Minimum Amount of Time to Brew Potions

    # kinda DP approach: BARELY PASSING (11885ms)
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        N = len(skill)
        M = len(mana)
        tskill = sum(skill)
        total = sum(mana[i] * tskill for i in range(M))
        dp = [0] * (N + 1)

        for j in range(1, N + 1):
            dp[j] = dp[j - 1] + skill[j - 1] * mana[0]

        overlaps = 0
        for i in range(1, M):
            cur = 0
            gap = float("-inf")
            for j in range(N):
                gap = max(gap, dp[j + 1] - cur)
                cur += skill[j] * mana[i]
                if j == N - 1:
                    overlaps += dp[N] - gap
                dp[j + 1] = cur
        return total - overlaps

    # kinda DP approach 2: MLE
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        N = len(skill)
        M = len(mana)
        tskill = sum(skill)
        total = sum(mana[i] * tskill for i in range(M))
        dp = [[0] * (N + 1) for _ in range(M)]

        for j in range(1, N + 1):
            dp[0][j] = dp[0][j - 1] + skill[j - 1] * mana[0]

        overlaps = 0
        for i in range(1, M):
            cur = 0
            gap = float("-inf")
            for j in range(N):
                gap = max(gap, dp[i - 1][j + 1] - cur)
                cur += skill[j] * mana[i]
                dp[i][j + 1] = cur
            overlaps += dp[i - 1][N] - gap

        return total - overlaps

    # binary Search approach: TLE
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        N, M = len(skill), len(mana)

        dp = [0] * N
        dp[0] = skill[0] * mana[0]
        for i in range(1, N):
            dp[i] = dp[i - 1] + skill[i] * mana[0]

        def calc(time, m):
            start = time
            for i, s in enumerate(skill):
                if start < dp[i]:
                    return False
                start += s * m
            return True

        for i in range(1, M):
            low, high = dp[0], dp[-1]
            while low <= high:
                mid = (low + high) // 2
                if calc(mid, mana[i]):
                    high = mid - 1
                else:
                    low = mid + 1

            dp[0] = skill[0] * mana[i] + low
            for j in range(1, N):
                dp[j] = dp[j - 1] + skill[j] * mana[i]

        return dp[-1]

    # binary Search approach: TLE
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        N, M = len(skill), len(mana)
        tskill = sum(skill)

        def calc(stime, prev_time, j):
            for i, s in enumerate(skill):
                prev_time += s * mana[j - 1]
                if stime < prev_time:
                    return False
                stime += s * mana[j]
            return True

        start = 0
        end = tskill * mana[0]
        for i in range(1, M):
            low, high = start, end
            while low <= high:
                mid = (low + high) // 2
                if calc(mid, start, i):
                    high = mid - 1
                else:
                    low = mid + 1
            start = low
            end = start + tskill * mana[i]

        return end

    def minTime(self, skill: List[int], mana: List[int]) -> int:
        time = [0] * (len(skill))
        prev = 0
        for i in range(len(mana)):
            t = time[0] + mana[i] * skill[0]
            for j in range(1, len(skill)):
                t = max(t, time[j]) + mana[i] * skill[j]

            for j in range(len(skill) - 1, -1, -1):
                time[j] = t
                t -= mana[i] * skill[j]
        return time[-1]


def test_solution():
    s = Solution()
    s.minTime([], [])


if __name__ == "__main__":
    test_solution()
