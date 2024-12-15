import heapq
from typing import Optional, List
from itertools import accumulate
import random
from collections import Counter, deque, defaultdict, OrderedDict
import math
import bisect
from math import inf
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

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        """ "
        1 2 3
            [3] 4 5 6
        """
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        N1 = len(nums1)
        N2 = len(nums2)
        median = (N1 + N2 + 1) // 2
        low = 0
        high = N1
        while low <= high:
            mid = (high + low) // 2
            mid2 = median - mid
            left1 = -inf if mid < 1 else nums1[mid - 1]
            right1 = inf if mid >= N1 else nums1[mid]
            left2 = -inf if mid2 < 1 else nums2[mid2 - 1]
            right2 = inf if mid >= N2 else nums2[mid2]
            if left1 <= right2 and left2 <= right1:
                if (N1 + N2) % 2 == 1:
                    return max(left1, left2)
                res = (max(left2, left1) + min(right1, right2)) / 2
                return res
            elif left1 > right2:
                high = mid - 1
            else:
                low = mid + 1

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        N1 = len(nums1)
        N2 = len(nums2)

        def search(median, low1, high1, low2, high2):
            if high1 < low1:
                return nums2[median - low1]
            if high2 < low2:
                return nums1[median - low2]
            mid1 = (high1 + low1) // 2
            mid2 = (high2 + low2) // 2
            val1 = nums1[mid1]
            val2 = nums2[mid2]
            if mid1 + mid2 < median:
                if val1 < val2:
                    return search(median, mid1 + 1, high1, low2, high2)
                return search(median, low1, high1, mid2 + 1, high2)
            else:
                if val1 > val2:
                    return search(median, low1, mid1 - 1, low2, high2)
                return search(median, low1, high1, low2, mid2 - 1)

        median = (N1 + N2) // 2
        if (N1 + N2) % 2 == 1:
            return search(median, 0, N1 - 1, 0, N2 - 1)
        else:
            return (
                search(median, 0, N1 - 1, 0, N2 - 1)
                + search(median - 1, 0, N1 - 1, 0, N2 - 1)
            ) / 2

    def trap(self, height: List[int]) -> int:
        N = len(height)
        ans = 0
        left = 0
        right = N - 1
        max_left = 0
        max_right = 0
        while left <= right:
            cur_left = height[left]
            cur_right = height[right]
            max_left = max(cur_left, max_left)
            max_right = max(cur_right, max_right)
            if max_left < max_right:
                ans += max_left - cur_left
                left += 1
            else:
                ans += max_right - cur_right
                right -= 1

        return ans

    def trap(self, height: List[int]) -> int:
        if not height:
            return 0

        left = 0
        right = len(height) - 1
        left_max = height[left]
        right_max = height[right]
        ans = 0

        while left < right:
            if height[left] < height[right]:
                left += 1
                left_max = max(left_max, height[left])
                ans += max(0, left_max - height[left])
            else:
                right -= 1
                right_max = max(right_max, height[right])
                ans += max(0, right_max - height[right])

        return ans

    def maxAverageRatio(self, classes: List[List[int]], extraStudents: int) -> float:
        ratios = []
        fulls = []
        for p, t in classes:
            if t == p:
                fulls.append((t, p))
            else:
                ratios.append((t, p))
        heapq.heapify(ratios)
        total = 0
        ext = extraStudents
        while ext > 0:
            cur_t, cur_p = heapq.heappop(ratios)
            cur_t += 1
            cur_p += 1
            heapq.heappush(ratios, (cur_t, cur_p))
            ext -= 1

        for t, p in ratios:
            total += p / t
        if fulls:
            for t, p in fulls:
                total += p / t
        return total / len(classes)

    def maxAverageRatio(self, classes: List[List[int]], extraStudents: int) -> float:
        def potential_gain(p, t):
            return (p + 1) / (t + 1) - p / t

        gains = []
        for p, t in classes:
            pg = potential_gain(p, t)
            gains.append((-pg, p, t))

        heapq.heapify(gains)
        total = 0
        ext = extraStudents
        while ext > 0:
            g, cur_p, cur_t = heapq.heappop(gains)
            cur_t += 1
            cur_p += 1
            heapq.heappush(gains, (-potential_gain(cur_p, cur_t), cur_p, cur_t))
            ext -= 1

        for g, p, t in gains:
            total += p / t

        return total / len(classes)


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
