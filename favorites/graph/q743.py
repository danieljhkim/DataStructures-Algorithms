from typing import Optional, List
from collections import defaultdict, deque
import math

"""743. Network Delay Time

Simple shortest path problem, but important lesson to be learned here from the constraints, 
and how it allows us to use different approaches.

Problem:

You are given a network of n nodes, labeled from 1 to n. 
You are also given times, a list of travel times as directed edges times[i] = (ui, vi, wi), 
where ui is the source node, vi is the target node, 
and wi is the time it takes for a signal to travel from source to target.

We will send a signal from a given node k. 
Return the minimum time it takes for all the n nodes to receive the signal. 
If it is impossible for all the n nodes to receive the signal, return -1.

Constraints:
    1 <= k <= n <= 100
    1 <= times.length <= 6000
    1 <= ui, vi <= n
    ui != vi
    0 <= wi <= 100
    All the pairs (ui, vi) are unique. (i.e., no multiple edges.)
"""


class Solution:
    """bfs approach
    - the small constraints allow us to use bfs here
    """

    def networkDelayTime(self, edges: List[List[int]], n: int, src: int) -> int:
        adj = [[] for _ in range(n)]
        small = math.inf
        for s, d, w in edges:
            small = min(small, w)
            adj[s - 1].append((w, d))

        visited = [math.inf] * n
        dq = deque([(src, 0, 0)])  # node, distance, steps

        while dq:
            cur, val, holds = dq.popleft()
            if holds > 0 or visited[cur - 1] != math.inf:
                if visited[cur - 1] == math.inf:
                    dq.append((cur, val, holds - 1))
                continue
            visited[cur - 1] = val
            for ndist, nei in adj[cur - 1]:
                nw = ndist + val
                if visited[nei - 1] == math.inf:
                    if ndist - small == 0:
                        dq.appendleft((nei, nw, 0))
                    else:
                        dq.append((nei, nw, ndist - small - 1))

        if math.inf not in visited:
            return max(visited)
        return -1
