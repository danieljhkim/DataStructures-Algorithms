from typing import *
import heapq
from collections import deque

"""862. Shortest Subarray with Sum at Least K

You are given an array of integers nums (may contain negatives) and an integer k.
Return the length of the shortest, non-empty subarray of nums with sum at least k.
If no such subarray exists, return -1.

Constraints:
    - 1 <= nums.length <= 105
    - -105 <= nums[i] <= 105
    - 1 <= k <= 109
    
Lessons learned:
    - beware of using sliding window when nagative numbers are present.
    - Removing a positive may unlock removal of a negative
"""


class Solution:
    """
    My first intuition was to use sliding-window. And it was a good learning experience.

    Failed test case: nums = [84,-37,32,40,95], k = 167
    """

    def shortestSubarray_wrong(self, nums: List[int], k: int) -> int:
        # when 0 or negative
        if k <= 0:
            for n in nums:
                if n >= k:
                    return 1
            return -1

        # when positive
        cur = left = 0
        best = float("inf")
        for right, n in enumerate(nums):
            cur += n
            while left <= right and cur <= 0:
                cur -= nums[left]
                left += 1
            while left <= right and cur - nums[left] >= k:
                cur -= nums[left]
                left += 1
            if cur >= k and right >= left:
                best = min(best, right - left + 1)
            while left <= right and cur >= k:
                best = min(best, right - left + 1)
                cur -= nums[left]
                left += 1

        return best if best != float("inf") else -1

    # heap approach: O(N logN)
    def shortestSubarray_heap(self, nums: List[int], k: int) -> int:
        heap = []
        total, best = 0, float("inf")

        for right, n in enumerate(nums):
            total += n
            if total >= k:
                best = min(right + 1, best)
            while heap and total - heap[0][0] >= k:
                val, idx = heapq.heappop(heap)
                best = min(best, right - idx)

            heapq.heappush(heap, (total, right))
        return -1 if best == float("inf") else best

    # deque approach: O(N)
    def shortestSubarray_dq(self, nums: List[int], k: int) -> int:
        """
        little hard to grasp the mechanics but I think it goes something like:
            - when we have a negative number, the prefix_sum dips, and when it dips, its not even needed for consideration.
            - slope should goes up only, and that gives us all info we need: diff from high to low
            - and we shrink and find valid candidates
        """
        prefix = [0]
        for n in nums:
            prefix.append(prefix[-1] + n)

        dq = deque()
        best = float("inf")
        for right, n in enumerate(prefix):
            # collect candidate answers
            while dq and n - prefix[dq[0]] >= k:
                left = dq.popleft()
                best = min(best, right - left)
            # remove the dip
            while dq and prefix[dq[-1]] >= n:
                dq.pop()
            dq.append(right)
        return -1 if best == float("inf") else best
