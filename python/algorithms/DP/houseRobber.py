from typing import Optional, List


class Solution:

    def rob(self, nums: List[int]) -> int:
        memo = {}

        def recurs(i):
            if i >= len(nums):
                return 0
            if i in memo:
                return memo[i]

            skip = recurs(i + 1)
            cur = nums[i] + recurs(i + 2)
            chosen = max(skip, cur)
            memo[i] = chosen
            return chosen

        return recurs(0)
