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

    def minimumPairRemoval(self, nums: List[int]) -> int:
        cnt = 0
        while True:
            idx = -1
            small = inf
            n = len(nums)
            go = False
            for i in range(1, n):
                if nums[i - 1] > nums[i]:
                    go = True
                    break
            if not go:
                return cnt
            for i in range(1, n):
                total = nums[i - 1] + nums[i]
                if total < small:
                    small = total
                    idx = i
            if idx == -1:
                return cnt
            nums[idx] = small
            nums.pop(idx - 1)
            cnt += 1

    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:  # MLE
        N = len(nums)
        has_zero = 0 in nums

        @cache
        def dp(cnt, idx, total, prod):
            res = -1
            if total == k and cnt > 0:
                if prod <= limit:
                    res = prod
                else:
                    if not has_zero:
                        return res
            if idx >= N:
                return res
            i = idx
            is_even = (cnt) % 2 == 0
            ntotal = total
            if is_even:
                ntotal += nums[i]
            else:
                ntotal -= nums[i]
            res = max(res, dp(cnt, i + 1, total, prod))
            nprod = prod * nums[i]
            if not has_zero and nprod > limit:
                return res
            res = max(res, dp(cnt + 1, i + 1, ntotal, nprod))
            return res

        res = dp(0, 0, 0, 1)
        dp.cache_clear()
        return res


class Router:

    def __init__(self, memoryLimit: int):
        self.limit = memoryLimit
        self.packets = deque()
        self.seen = set()
        self.table = defaultdict(deque)

    def addPacket(self, source: int, destination: int, timestamp: int) -> bool:
        key = f"{source}:{destination}:{timestamp}"
        if key in self.seen:
            return False
        if len(self.packets) >= self.limit:
            out = self.packets.popleft()
            key2 = f"{out[0]}:{out[1]}:{out[2]}"
            self.seen.discard(key2)
            self.table[out[1]].popleft()
        self.packets.append([source, destination, timestamp])
        self.table[destination].append(timestamp)
        self.seen.add(key)
        return True

    def forwardPacket(self) -> List[int]:
        if self.packets:
            out = self.packets.popleft()
            key2 = f"{out[0]}:{out[1]}:{out[2]}"
            self.seen.discard(key2)
            self.table[out[1]].popleft()
            return out
        return []

    def getCount(self, destination: int, startTime: int, endTime: int) -> int:
        """ "
        0 1 1 1 2 3
        """
        dq = self.table[destination]
        if not dq:
            return 0
        low = 0
        high = len(dq) - 1
        while low <= high:
            mid = (low + high) // 2
            if dq[mid] >= startTime:
                high = mid - 1
            else:
                low = mid + 1
        s = low
        low = 0
        high = len(dq) - 1
        while low <= high:
            mid = (low + high) // 2
            if dq[mid] > endTime:
                high = mid - 1
            else:
                low = mid + 1
        e = high
        return e - s + 1


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
