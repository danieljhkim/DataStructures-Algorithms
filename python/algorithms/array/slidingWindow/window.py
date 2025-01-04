from collections import Counter, deque
from typing import Optional, List


class Window:

    def minWindow(self, s: str, t: str) -> str:
        """find the min window where it contains all characters of t
        - not space optimal
        - coordinates = [(1, a), (2, b)...]
        """
        required = Counter(t)
        coordinates = []

        for i, w in enumerate(s):
            if w in required:
                coordinates.append((i, w))
        queue = deque()
        good = 0
        min_range = float("inf")
        start = 0
        end = 0
        cur = Counter()
        for idx, w in coordinates:
            cur[w] += 1
            queue.append((idx, w))
            if cur[w] == required[w]:
                good += 1
            if good == len(required):
                while queue and cur[queue[0][1]] > required[queue[0][1]]:
                    sidx, nw = queue.popleft()
                    cur[nw] -= 1
                cur_range = idx - queue[0][0] + 1
                if cur_range < min_range:
                    min_range = cur_range
                    start = queue[0][0]
                    end = idx
        if min_range != float("inf"):
            return s[start : end + 1]
        return ""

    # 3364
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        """minimum sum of the subarrays of length l - r
        - tricky
        """
        ans = float("inf")
        prefix = [0]
        for n in nums:
            prefix.append(prefix[-1] + n)
        for right in range(l, len(prefix)):
            start = max(0, right - r)
            end = right - l + 1
            for left in range(start, end):
                total = prefix[right] - prefix[left]
                ans = min(total, ans)
        if ans == float("inf"):
            return -1
        return ans

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        """
        - return min length of subarray that has sum greater than equal to target
        - return 0 if not found
        """
        left = 0
        total = 0
        ans = float("inf")
        for right in range(len(nums)):
            total += nums[right]
            if total >= target:
                while left < right and total - nums[left] >= target:
                    total -= nums[left]
                    left += 1
                length = right - left + 1
                ans = min(length, ans)
        if ans == float("inf"):
            return 0
        return ans

    # 1438
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        """
        - longest continuous subarray with abs diff less than or equal to limit
        """
        dec_dq = deque()
        inc_dq = deque()
        max_len = 0
        left = 0
        N = len(nums)
        for right in range(N):
            cur = nums[right]
            while dec_dq and dec_dq[-1] < cur:
                dec_dq.pop()
            while inc_dq and inc_dq[-1] > cur:
                inc_dq.pop()
            dec_dq.append(cur)
            inc_dq.append(cur)
            while dec_dq and inc_dq and dec_dq[0] - inc_dq[0] > limit:
                if inc_dq[0] == nums[left]:
                    inc_dq.popleft()
                if dec_dq[0] == nums[left]:
                    dec_dq.popleft()
                left += 1
            max_len = max(max_len, right - left + 1)
        return max_len
