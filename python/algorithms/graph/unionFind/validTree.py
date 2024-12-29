from collections import defaultdict
from typing import Optional, List


class UnionFind:

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootx = self.find(x)
        rooty = self.find(y)
        if rootx != rooty:
            self.parent[rootx] = rooty


class Solution:
    """_summary_
    - return True if the graph is a valid tree - i.e no cycles, all are connected

    * 1. Tree has n-1 edges
    * 2. Graph with less than n-1 edges is definitely not connected
    * 3. Graph with more than n-1 edges definitely has cycle
    """

    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges) != n - 1:
            return False  # more or less -> cycle or unconnected
        uf = UnionFind(n)

        for edge in edges:
            x = edge[0]
            y = edge[1]
            rootx = uf.find(x)
            rooty = uf.find(y)
            if rootx == rooty:
                # this means one node already had a parent -> double connections
                return False
            uf.union(x, y)

        root = uf.find(0)
        for i in range(1, n):
            if uf.find(i) != root:
                # found a loner
                return False
        return True

    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges) != n - 1:
            return False

        adj_table = defaultdict(list)
        for x, y in edges:
            adj_table[x].append(y)
            adj_table[y].append(x)

        seen = set()
        stack = [(0, -1)]  # (current node, parent node)

        while stack:
            cur, parent = stack.pop()
            if cur in seen:
                return False
            seen.add(cur)
            for neighbor in adj_table[cur]:
                if neighbor != parent:  # Ignore the edge leading back to the parent
                    stack.append((neighbor, cur))

        return len(seen) == n  # Ensure all nodes are visited

    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        """_summary_
        - turn the graph into a valid tree by removing an edge
        """
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

        for x, y in edges:
            rootx = find(x)
            rooty = find(y)
            if rootx == rooty:
                return [x, y]
            union(x, y)
        return []
