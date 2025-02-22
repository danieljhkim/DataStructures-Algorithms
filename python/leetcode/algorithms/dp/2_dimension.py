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

    # 673. Number of Longest Increasing Subsequence
    def findNumberOfLIS(self, nums: List[int]) -> int:
        N = len(nums)

        @cache
        def dp(idx, prev):
            if idx == N:
                return 0, 1
            cnt = res = 0
            ln2, cnt2 = dp(idx + 1, prev)
            if nums[idx] > prev:
                ln3, cnt3 = dp(idx + 1, nums[idx])
                res = 1 + ln3
                cnt = cnt3
            if res > ln2:
                return res, cnt
            elif res < ln2:
                return ln2, cnt2
            else:
                return res, cnt2 + cnt

        ln, cnt = dp(0, -inf)
        return cnt
