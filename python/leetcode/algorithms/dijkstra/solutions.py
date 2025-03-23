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

    # 2737. Find the Closest Marked Node
    def minimumDistance(
        self, n: int, edges: List[List[int]], s: int, marked: List[int]
    ) -> int:
        marked = set(marked)
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((w, v))

        distances = [inf] * n
        distances[s] = 0
        heap = [(0, s)]
        while heap:
            dist, cur = heapq.heappop(heap)
            if cur in marked:
                return dist
            for w, nei in adj[cur]:
                ndist = w + dist
                if ndist < distances[nei]:
                    distances[nei] = ndist
                    heapq.heappush(heap, (ndist, nei))
        return -1

    # 1976. Number of Ways to Arrive at Destination
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        MOD = 10**9 + 7
        times = [inf] * n
        counts = [0] * n
        adj = [[] for _ in range(n)]
        for u, v, w in roads:
            adj[u].append((w, v))
            adj[v].append((w, u))

        times[0] = 0
        counts[0] = 1
        heap = [(0, 0)]
        shortest = inf
        while heap:
            time, cur = heapq.heappop(heap)
            if cur == n - 1 and time <= shortest:
                shortest = time
                continue
            if time >= shortest:
                continue
            for w, nei in adj[cur]:
                ntime = w + time
                if ntime <= shortest and times[nei] > ntime:
                    times[nei] = ntime
                    counts[nei] = counts[cur]
                    heapq.heappush(heap, (ntime, nei))
                elif times[nei] == ntime:
                    counts[nei] = (counts[nei] + counts[cur]) % MOD
        return counts[n - 1] % MOD
