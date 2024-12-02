from typing import List, Optional
from collections import deque, defaultdict


class Solution:

    # 2097
    def validArrangement(self, pairs: List[List[int]]) -> List[List[int]]:
        indegree = defaultdict(int)
        outdegree = defaultdict(int)
        adj = defaultdict(deque)

        for start, end in pairs:
            indegree[end] += 1
            outdegree[start] += 1
            adj[start].append(end)

        start_node = None
        for k, v in outdegree.items():
            if indegree[k] + 1 == v:
                start_node = k
                break
        if start_node is None:
            start_node = pairs[0][0]

        # dfs
        paths = []

        def dfs(node):
            while adj[node]:
                next_node = adj[node].popleft()
                dfs(next_node)
            paths.append(node)

        dfs(start_node)
        paths.reverse()
        result = []
        for i in range(len(paths) - 1):
            result.append([paths[i], paths[i + 1]])

        return result

    # 332
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        adj = defaultdict(list)
        for start, end in tickets:
            adj[start].append(end)
        for k, v in adj.items():
            v.sort(reverse=True)

        paths = []

        def dfs(node):
            while adj[node]:
                dst = adj[node].pop()
                dfs(dst)
            paths.append(node)

        start_node = "JFK"
        dfs(start_node)
        return paths[::-1]
