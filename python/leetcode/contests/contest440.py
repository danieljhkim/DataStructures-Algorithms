import heapq
import random
import math
import bisect
from typing import *
from math import inf, factorial, gcd, lcm
from functools import lru_cache, cache
from heapq import heapify, heappush, heappop
from itertools import accumulate, permutations, combinations
from collections import Counter, deque, defaultdict, OrderedDict
from sortedcontainers import SortedSet, SortedList, SortedDict


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

    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        ans = 0
        N = len(fruits)
        for f in fruits:
            for i, b in enumerate(baskets):
                if b >= f:
                    baskets[i] = -1
                    break
                if i == N - 1:
                    ans += 1
        return ans

    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        N = len(nums1)
        ans = [0] * N
        arr1 = []

        for i, n in enumerate(nums1):
            arr1.append((n, i))
        arr1.sort()

        out = []
        i = 0
        total = 0
        while i < N:
            n, idx = arr1[i]
            if i + 1 < N and arr1[i + 1][0] == n:
                sames = []
                tmp = total
                while i < N and arr1[i][0] == n:
                    nn, id2 = arr1[i]
                    if len(out) < k:
                        heapq.heappush(out, nums2[id2])
                        tmp += nums2[id2]
                    else:
                        if out[0] < nums2[id2]:
                            tmp -= out[0]
                            tmp += nums2[id2]
                            heapq.heapreplace(out, nums2[id2])
                    sames.append(id2)
                    i += 1
                for j in sames:
                    ans[j] = total
                total = tmp
            else:
                ans[idx] = total
                if len(out) < k:
                    heapq.heappush(out, nums2[idx])
                    total += nums2[idx]
                else:
                    if out[0] < nums2[idx]:
                        total -= out[0]
                        total += nums2[idx]
                        heapq.heapreplace(out, nums2[idx])
                i += 1
        return ans

    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        N = len(fruits)
        small = min(baskets)
        big = max(baskets)
        ans = 0
        buckets = [deque() for _ in range(big - small + 1)]
        for i, n in enumerate(baskets):
            pos = n - small
            buckets[pos].append(i)

        N2 = len(buckets)
        # stuffs = deque()
        # for i in range(N2 - 1, -1, -1):
        #     if buckets[i]:
        #         stuffs.extendleft(buckets[i])

        for i, n in enumerate(fruits):
            pos = n + small
            found = False
            if pos < N2:
                for j in range(pos, N2):
                    if buckets[j]:
                        buckets[j].popleft()
                        found = True
                        break
            if not found:
                ans += 1
        return ans


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
