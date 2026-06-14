"""
Problem: Earliest Arrival With Nonnegative Weights

You are given a directed, weighted graph with N nodes labeled 0..N-1 and a list of edges:

edges[i] = [u, v, w] meaning there is an edge from u to v with travel time w, where w >= 0.

Given src and dst, return the minimum total travel time from src to dst.
If dst is unreachable, return -1.
"""

from typing import Optional, List
from collections import defaultdict, deque
import math


def shortest_path(edges, src, dst):
    adj = defaultdict(list)
    small = math.inf
    for s, d, w in edges:
        small = min(small, w)
        adj[s].append((w, d))

    distances = defaultdict(lambda: math.inf)
    distances[src] = 0
    dq = deque([(src, 0, 0)])

    while dq:
        cur, val, holds = dq.popleft()
        if holds > 0:
            if distances[cur] == val:
                dq.append((cur, val, holds - 1))
            continue
        if cur == dst:
            return val
        for ndist, nei in adj[cur]:
            nw = ndist + val
            if distances[nei] > nw:
                distances[nei] = nw
                if ndist - small == 0:
                    dq.appendleft((nei, nw, 0))
                else:
                    dq.append((nei, nw, ndist - small - 1))
    return -1


edges = [[0, 1, 2], [0, 2, 5], [1, 2, 1], [1, 3, 2], [2, 3, 1], [3, 4, 3]]
src = 0
dst = 4


def networkDelayTime(self, edges: List[List[int]], n: int, src: int) -> int:
    adj = defaultdict(list)
    small = math.inf
    for s, d, w in edges:
        small = min(small, w)
        adj[s].append((w, d))

    distances = defaultdict(lambda: math.inf)
    distances[src] = 0
    dq = deque([(src, 0, 0)])

    while dq:
        cur, val, holds = dq.popleft()
        if holds > 0:
            if distances[cur] == val:
                dq.append((cur, val, holds - 1))
            continue
        if len(distances) == n:
            return max(distances.values())
        for ndist, nei in adj[cur]:
            nw = ndist + val
            if nei not in distances:
                distances[nei] = nw
                if ndist - small == 0:
                    dq.appendleft((nei, nw, 0))
                else:
                    dq.append((nei, nw, ndist - small - 1))
    return -1
