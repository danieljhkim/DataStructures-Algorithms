from typing import *
from collections import deque
from functools import cache
from math import inf

"""
Fist all problem solved (AK)
"""


class Solution:
    # 3833. Count Dominant Indices
    def dominantIndices(self, nums: List[int]) -> int:
        N, total, res = len(nums), sum(nums), 0
        cnt = N
        for n in nums[:-1]:
            total -= n
            cnt -= 1
            avg = total / cnt
            if n > avg:
                res += 1
        return res

    # 3834. Merge Adjacent Equal Elements
    def mergeAdjacent(self, nums: List[int]) -> List[int]:
        stack = []
        for i, n in enumerate(nums):
            cur = n
            while stack and stack[-1] == cur:
                out = stack.pop()
                cur += out
            stack.append(cur)
        return stack

    # 3835. Count Subarrays With Cost Less Than or Equal to K
    def countSubarrays(self, nums: List[int], k: int) -> int:
        """
        1. when max change
        2. when min change
        3. when r - l big
        """
        sdq = deque()
        bdq = deque()
        res = left = 0
        for right, n in enumerate(nums):
            while sdq and nums[sdq[-1]] > n:
                out = sdq.pop()
            sdq.append(right)
            while bdq and nums[bdq[-1]] < n:
                out = bdq.pop()
            bdq.append(right)
            while (
                left < right and (nums[bdq[0]] - nums[sdq[0]]) * (right - left + 1) > k
            ):
                if bdq[0] == left:
                    bdq.popleft()
                if sdq[0] == left:
                    sdq.popleft()
                left += 1
            res += right - left + 1
        return res

    # 3836. Maximum Score Using Exactly K Pairs
    def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
        M, N = len(nums1), len(nums2)

        @cache
        def dp(m, n, rem):
            if rem == 0:
                return 0
            if m == M or n == N:
                return -inf
            cur = nums1[m] * nums2[n]
            res1 = dp(m + 1, n + 1, rem - 1) + cur
            res2 = max(dp(m + 1, n, rem), dp(m, n + 1, rem))
            return max(res2, res1)

        res = dp(0, 0, k)
        dp.cache_clear()
        return res
