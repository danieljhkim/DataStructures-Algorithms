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


class Solution:

    def minOperations(self, nums: List[int], k: int) -> int:
        total = sum(nums)
        diff = total % k
        return diff

    def uniqueXorTriplets(self, nums: List[int]) -> int:  # unable to solve
        """ "
        1 2 3 4 5 6 7
        """
        N = len(nums)
        n = N
        nset = set()
        top = 0
        for i in range(1, N + 1):
            xn = n ^ i
            top = max(xn, top)
        for i in range(1, N + 1):
            nset.add(i ^ top)
            nset.add(0 ^ i)
        print(nset)
        nset.update(nums)
        return len(nset)

    ########## upsolve ###########

    # 3513. Number of Unique XOR Triplets I
    def uniqueXorTriplets(self, nums: List[int]) -> int:
        N = len(nums)
        nset = set()
        top = 0
        for i in range(1, N + 1):
            xn = N ^ i
            top = max(xn, top)
        start = 0 if N >= 3 else 1
        for i in range(start, N + 1):
            nset.add(i ^ top)
        nset.update(nums)
        res = len(nset)
        if N >= 3 and 0 not in nset:
            res += 1
        return res

    # 3514. Number of Unique XOR Triplets II
    def uniqueXorTriplets(self, nums: List[int]) -> int:
        nset, xset = set(nums), set()
        arr = list(nset)
        N = len(arr)
        zero = False
        for i in range(N):
            for j in range(i + 1, N):
                xval = arr[i] ^ arr[j]
                xset.add(xval)

        for n in xset:
            for n2 in arr:
                xval = n ^ n2
                nset.add(xval)
        return len(nset)


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
