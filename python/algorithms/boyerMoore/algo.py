from typing import Optional, List

"""
How to find majority elements in constant time?

Boyer Moore Algo Basis
- There can be at most one majority element which is more than ⌊n/2⌋ times.
- There can be at most two majority elements which are more than ⌊n/3⌋ times.
- There can be at most three majority elements which are more than ⌊n/4⌋ times.

"""


class Solution:

    def majorityElement(self, nums: List[int]) -> List[int]:
        # more than n/3

        candidate1 = None
        candidate2 = None
        count1 = 0
        count2 = 0

        for n in nums:
            if n == candidate1:
                count1 += 1
            elif n == candidate2:
                count2 += 1
            elif count1 == 0:
                candidate1 = n
                count1 += 1
            elif count2 == 0:
                candidate2 = n
                count2 += 1
            else:
                count1 -= 1
                count2 -= 1

        limit = len(nums) // 3
        ans = []
        for c in [candidate1, candidate2]:
            if nums.count(c) > limit:
                ans.append(c)
        return ans
