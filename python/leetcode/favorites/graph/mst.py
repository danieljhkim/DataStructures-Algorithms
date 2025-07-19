from typing import *

"""
MST - minimum spanning tree
"""


class Solution:

    # 3613. Minimize Maximum Component Cost
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
