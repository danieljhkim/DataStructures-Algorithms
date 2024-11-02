import heapq
from typing import Optional, List
import random
from collections import Counter, deque, defaultdict
import math
from functools import lru_cache


def fn(arr):
    BASE_CASE = True

    def dp(i):
        if BASE_CASE:
            return 0
        if i in memo:
            return memo[i]
        ans = dp(i - 2) + dp(i - 1)
        memo[i] = ans
        return ans

    memo = {}
    return dp(10)


class Solution:
    pass
