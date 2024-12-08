import heapq
from itertools import accumulate
from typing import Optional, List
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

    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        """_summary_
        0, 1, 2, 3
           3  2   1
        """
        N = len(nums)
        result = [0] * N
        for i, n in enumerate(nums):
            if n > 0:
                idx = (i + n) % (N)
                result[i] = nums[idx]
            elif n < 0:
                idx = i + n
                if idx < 0:
                    idx = abs(idx) % (N)
                    result[i] = nums[-idx]
                else:
                    result[i] = nums[idx]
            else:
                result[i] = nums[i]
        return result

    def maxRectangleArea(self, points: List[List[int]]) -> int:
        """ "
        x, y
        c1(x1, y1), c2(x1, y2)
        c3(x2, y1), c4(x2, y2)

        small x1, big x2
        small y2, big y3
        x1, y1       x1, y2


        x2,y1        x2, y2
        """

        def check(x1, y1, x2, y2):
            smallx = min(x1, x2)
            bigx = max(x1, x2)
            smally = min(y1, y2)
            bigy = max(y1, y2)
            ysets = set([n for n in range(smally + 1, bigy)])
            xsets = set([n for n in range(smallx + 1, bigx)])
            for x in range(smallx, bigx + 1):
                ys = adjx[x].intersection(ysets)
                if len(ys) > 0:
                    return False
            for y in range(smally, bigy + 1):
                xs = adjy[y].intersection(xsets)
                if len(xs) > 0:
                    return False
            return True

        adjx = defaultdict(set)
        adjy = defaultdict(set)
        top = -1
        for x, y in points:
            adjx[x].add(y)
            adjy[y].add(x)
        for x1, y1 in points:
            candx2 = adjy[y1]
            for x2 in candx2:
                candy2 = adjx[x1].intersection(adjx[x2])
                for y2 in candy2:
                    area = abs(x1 - x2) * abs(y1 - y2)
                    if area > top:
                        if check(x1, y1, x2, y2):
                            top = area
        if top == 0:
            return -1
        return top

    def maxRectangleArea(self, xCoord: List[int], yCoord: List[int]) -> int:
        @lru_cache
        def check(small, big):
            smallx = small[0]
            bigx = big[0]
            smally = small[1]
            bigy = big[1]
            ysets = set([n for n in range(smally + 1, bigy)])
            xsets = set([n for n in range(smallx + 1, bigx)])
            for x in range(smallx, bigx + 1):
                ys = adjx[x].intersection(ysets)
                if len(ys) > 0:
                    return False
            for y in range(smally, bigy + 1):
                xs = adjy[y].intersection(xsets)
                if len(xs) > 0:
                    return False
            return True

        adjx = defaultdict(set)
        adjy = defaultdict(set)
        top = -1
        N = len(xCoord)
        points = sorted(zip(xCoord, yCoord))
        for x, y in points:
            adjx[x].add(y)
            adjy[y].add(x)

        for x1, y1 in points:
            candx2 = adjy[y1]
            for x2 in candx2:
                candy2 = adjx[x1].intersection(adjx[x2])
                ttop = -1
                for y2 in candy2:
                    area = abs(x1 - x2) * abs(y1 - y2)
                    if ttop > top:
                        smallx = min(x1, x2)
                        bigx = max(x1, x2)
                        smally = min(y1, y2)
                        bigy = max(y1, y2)
                        if check((smallx, smally), (bigx, bigy)):
                            ttop = area
                if ttop > -1:
                    return ttop
        if top == 0:
            return -1
        return top

    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        """ "
        1, 2, 3

        0 1 3 6
        0 1 2 3
        """
        N = len(nums)
        prefix_sums = list(accumulate(nums, initial=0))
        prefix = [0]
        for n in nums:
            prefix.append(prefix[-1] + n)
        total = float("-inf")
        i = 0
        prev = 0
        while i <= N:
            idx = i + k
            cur = prefix[i]
            while idx <= N:
                t = prefix[idx] - cur
                total = max(t, total)
                total = max(t, total)
                idx += k
        return total


def test_solution():
    s = Solution()
    print(-6 % 5)


if __name__ == "__main__":
    test_solution()
