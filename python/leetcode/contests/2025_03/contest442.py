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

    def maxContainers(self, n: int, w: int, maxWeight: int) -> int:
        ans = 0
        cur = 0
        for i in range(n * n):
            if cur + w <= maxWeight:
                cur += w
                ans += 1
            else:
                break
        return ans

    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        R = len(properties)
        C = len(properties[0])
        pset = []
        for p in properties:
            pset.append(set(p))

        def intersect(i, j):
            return len(pset[i].intersection(pset[j]))

        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            else:
                if parent[x] != x:
                    parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rootx = find(x)
            rooty = find(y)
            if rootx != rooty:
                parent[rootx] = parent[rooty]

        ans = set()
        for i in range(R - 1):
            for j in range(i + 1, R):
                if intersect(i, j) >= k:
                    root = union(i, j)

        for i in range(R):
            root = find(i)
            ans.add(root)
        return len(ans)

    def minTime(self, skill: List[int], mana: List[int]) -> int:
        N = len(skill)
        M = len(mana)
        tskill = sum(skill)
        total = sum(mana[i] * tskill for i in range(M))
        track = [[0] * (N + 1) for _ in range(M)]
        diffs = 0
        for j in range(1, N + 1):
            track[0][j] = skill[j - 1] * mana[0] + track[0][j - 1]

        for i in range(1, M):
            tmp = 0
            diff = 0
            for j in range(N):
                cur = skill[j] * mana[i]
                tmp += cur
                diff = max(diff, track[i - 1][j + 1] - tmp)
                track[i][j] = tmp
            track[i][N] = tmp
            diffs += diff
        return total - diffs


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
