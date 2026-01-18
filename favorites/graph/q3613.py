from typing import *

"""3613. Minimize Maximum Component Cost

You are given an undirected connected graph with n nodes labeled from 0 to n - 1 and a 2D integer array edges where edges[i] = [ui, vi, wi] denotes an undirected edge between node ui and node vi with weight wi, and an integer k.

You are allowed to remove any number of edges from the graph such that the resulting graph has at most k connected components.

The cost of a component is defined as the maximum edge weight in that component. If a component has no edges, its cost is 0.

Return the minimum possible value of the maximum cost among all components after such removals.
"""


# MST
class Solution:
    def minCost(self, n: int, edges: List[List[int]], k: int) -> int:
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
                return True
            else:
                return False

        edges.sort(key=lambda x: x[2])
        options = []
        for u, v, w in edges:
            if union(u, v):  # found a connecting point
                options.append(w)

        options.sort(reverse=True)
        idx = len(options) - k
        if idx < 0:
            return 0
        return options[idx]
