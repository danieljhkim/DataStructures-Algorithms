import heapq
from typing import Optional, List
import random
from collections import Counter, deque, defaultdict, OrderedDict
import math
import bisect
from functools import lru_cache


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:

    # def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
    #     prefix = [0]
    #     for n in nums:
    #         prefix.append(n + prefix[-1])
    #     left = 0
    #     right = 1
    #     ans = float("inf")
    #     drange = set()
    #     for i in range(l, r + 1):
    #         drange.add(i)

    #     diff = r - l + 1
    #     left = 0
    #     for right in range(len(nums)):
    #         c = diff
    #         while left
    #     return ans

    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        prefix = [0]
        for n in nums:
            prefix.append(n + prefix[-1])
        left = 0
        right = 1
        ans = float("inf")

        while right < len(prefix):
            diff = right - left
            if diff > r:
                while right - left - 1 > l:
                    left += 1
                total = prefix[right] - prefix[left]
                if total > 0:
                    ans = min(ans, total)
            elif l <= diff <= r:
                total = prefix[right] - prefix[left]
                if total > 0:
                    ans = min(ans, total)
            right += 1

        if ans == float("inf"):
            return -1
        return ans

    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]

        ans = float("inf")

        # Sliding window approach
        for right in range(l, n + 1):
            for left in range(max(0, right - r), right - l + 1):
                total = prefix[right] - prefix[left]
                ans = min(ans, total)

        return ans if ans != float("inf") else -1

    def isPossibleToRearrange(self, s: str, t: str, k: int) -> bool:
        splits = defaultdict(int)
        left = 0
        size = len(s) // k
        for right in range(k, len(s) + size, size):
            if right > len(s):
                break
            word = s[left:right]
            splits[word] += 1
            left = right
        left = 0
        for right in range(k, len(t) + size, size):
            if right > len(t):
                break
            word = t[left:right]
            if word in splits and splits[word] >= 1:
                splits[word] -= 1
            else:
                return False
        return True

    def minimumSumSubarray2(self, nums: List[int], l: int, r: int) -> int:
        """_summary_
        1, 2, 3, 4
           [      ]
        0, 1, 3, 6, 10
          [          ]
        l=2
        r=4
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
                if total > 0:
                    ans = min(total, ans)
        if ans == float("inf"):
            return -1
        return ans

    def minimumSumSubarray2(self, nums: List[int], l: int, r: int) -> int:
        """_summary_
        [1,2,3,4,5]
            [     ]
        [  ]

        [1,2,3,4]
        [2,3,4] - queue2 [1]
        [3,4] - queue2 [1,2]
        [3,4,5] - queue2 [2]
        [3,4]   - queue2 [2,3]


        1-3

        [1,2,3,4,5,6,7,8,9,10]
        []

        """
        ans = float("inf")

        def calc(total):
            nonlocal ans
            if 0 < total < ans:
                ans = total

        queue = deque(nums[:l])
        lqueue = deque()
        total = sum(nums[:l])
        calc(total)
        ltotal = 0
        for n in nums[l:]:
            out = queue.popleft()
            total -= out
            total += n
            ltotal += out
            queue.append(n)
            lqueue.append(out)
            calc(total)
            if len(lqueue) + len(queue) <= r:
                calc(ltotal + total)
            elif r > l:
                trash = lqueue.popleft()
                ltotal -= trash
                calc(ltotal + total)
                for i in range(l):
                    out = lqueue.popleft()
                    ltotal -= out
                    lqueue.append(out)
                    calc(ltotal + total)
        if r > l:
            while len(lqueue) > l:
                out = lqueue.popleft()
                ltotal -= out
                calc(ltotal + total)
                if r == len(nums):
                    calc(ltotal)

        if ans == float("inf"):
            return -1
        return ans

        # push back into queue


def test_solution():
    s = Solution()
    n = [-13, -21, 24, -3]
    s.minimumSumSubarray2(n, 1, 4)


if __name__ == "__main__":
    test_solution()
    print("hello world")
