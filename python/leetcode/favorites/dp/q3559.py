from typing import *
from functools import cache
from collections import deque, defaultdict


# 3559. Number of Ways to Assign Edge Weights II
# DP, LCA, BFS, undirected tree, graph
class Solution:
    """_summary_
    There is an undirected tree with n nodes labeled from 1 to n, rooted at node 1.
    The tree is represented by a 2D integer array edges of length n - 1, where edges[i] = [ui, vi]
    Initially, all edges have a weight of 0. You must assign each edge a weight of either 1 or 2.
    The cost of a path between any two nodes u and v is the total weight of all edges in the path connecting them.

    You are given a 2D integer array queries. For each queries[i] = [ui, vi], determine the number of ways to assign weights to edges in the path such that the cost of the path between ui and vi is odd.
    Return an array answer, where answer[i] is the number of valid assignments for queries[i].
    """

    def assignEdgeWeights(
        self, edges: List[List[int]], queries: List[List[int]]
    ) -> List[int]:

        MOD = 10**9 + 7
        adj, ans = defaultdict(list), []
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        index, parent = {1: 0}, {}
        dq = deque([(1, 0)])
        while dq:
            cur, dist = dq.popleft()
            for nei in adj[cur]:
                if nei not in index:
                    parent[nei] = cur
                    index[nei] = dist + 1
                    dq.append((nei, dist + 1))

        @cache
        def dp(idx, is_even): # equivalent to: idx**2
            if idx == 0:
                if not is_even:
                    return 1
                return 0
            elif idx < 0:
                return 0
            res = dp(idx - 1, not is_even) + dp(idx - 1, is_even)
            return res % MOD

        for u, v in queries:
            dist = 0
            
            # finding LCA 
            while index[u] > index[v]:
                u = parent[u]
                dist += 1
            while index[v] > index[u]:
                v = parent[v]
                dist += 1
            while v != u:
                v = parent[v]
                u = parent[u]
                dist += 2
            # (dist - 1)**2
            ans.append((dp(dist - 1, False) + dp(dist - 1, True)) % MOD)
        return ans
