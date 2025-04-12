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

    def maxSum(self, nums: List[int]) -> int:
        wset = set()
        ans = 0
        for n in nums:
            if n > 0:
                if n not in wset:
                    ans += n
                    wset.add(n)
        if ans == 0:
            nums.sort()
            return nums[-1]
        return ans

    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        N = len(nums)
        ans = []
        counts = defaultdict(list)
        for i, n in enumerate(nums):
            counts[n].append(i)
            counts[n].append(i + N)
            counts[n].append(i - N)

        for v in counts.values():
            v.sort()

        for i in queries:
            cur = nums[i]
            arr = counts[cur]
            if len(arr) == 3:
                ans.append(-1)
            else:
                left = bisect.bisect_left(arr, i)
                right = bisect.bisect_left(arr, i + N)
                mid = bisect.bisect_left(arr, i - N)
                leng = len(arr)
                options = []
                if left > 0:
                    options.append(arr[left - 1])
                options.append(arr[right - 1])
                if (right + 1) < leng:
                    options.append(arr[(right + 1)])
                if (left + 1) < leng:
                    options.append(arr[(left + 1)])
                if (mid + 1) < leng:
                    options.append(arr[(mid + 1)])
                if (mid - 1) >= 0:
                    options.append(arr[(mid - 1)])
                small = inf
                for o in options:
                    if o != i and o != i + N and o != i - N:
                        diff = abs(o - i)
                        diff2 = abs(o - (N + i))
                        diff3 = abs(o - (i - N))
                        small = min(diff, diff2, diff3)
                ans.append(small)
        return ans

    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        # queries[i] = [li, ri, vali]
        N = len(nums)
        table = defaultdict(list)
        for i, n in enumerate(nums):
            if n > 0:
                table[n].append(i)

        total = sum(nums)
        k = 0
        for l, r, v in queries:
            if v not in table:
                continue
            out = table[v]
            left = bisect.bisect_left(out, l)
            right = bisect.bisect_left(out, r)
            keep = []
            for i in range(left):
                keep.append(i)
            for i in range(right + 1, len(out)):
                keep.append(i)
            if keep:
                table[v] = keep
            if len(keep) != len(out):
                total -= (right - left + 1) * v
                if total == 0:
                    return k + 1
            k += 1

        return -1


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
