from typing import Optional, List
from collections import deque


class Solution:
    """_summary_
    - top numbers in windows size of k
    """

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """_summary_
        [1, 2, 3], 2, 4, 6  [3]
        1, [2, 3, 2], 4, 6
        1, 2, [3, 2, 4], 6  [3]
        1, 2, 3, [2, 4, 6]
        """
        dq = deque()
        ans = []
        for i in range(k):
            while dq and nums[dq[-1]] <= nums[i]:
                dq.pop()
            dq.append(i)

        ans.append(nums[dq[0]])
        for i in range(k, len(nums)):
            if dq and dq[0] == i - k:
                dq.popleft()
            while dq and nums[dq[-1]] <= nums[i]:
                dq.pop()
            dq.append(i)
            ans.append(nums[dq[0]])
        return ans
