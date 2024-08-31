from ast import List
from typing import Optional


class Solution:

    # 228. Summary Ranges
    def summaryRanges(self, nums: List[int]) -> List[str]:
        ans = []
        i = 0
        n = len(nums)
        while i < n:
            start = i
            while i + 1 < n and nums[i + 1] == nums[i] + 1:
                i += 1
            if i == start:
                ans.append(str(nums[i]))
            else:
                ans.append(f"{nums[start]}->{nums[i]}")
            i += 1
        return ans
