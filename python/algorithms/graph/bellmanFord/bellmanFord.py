"""Bellman-Ford Algorithm
    - single source shortest path
    - slower than djkra but good when there is negative weights
    - time: O(V*E)
    - space: O(V)
"""

import heapq
from typing import Dict, List
from math import inf


class BellmanFord:

    def bellman_ford(self, vertices: int, edges: list, src):
        dist = [inf] * vertices
        dist[src] = 0
        for i in range(vertices - 1):
            for u, v, wt in edges:
                if dist[u] != inf and dist[u] + wt < dist[v]:
                    dist[v] = dist[u] + wt

        for u, v, wt in edges:
            if dist[u] + wt < dist[v]:
                return "negative cycle"

        return dist

    # 787
    def findCheapestPrice(
        self, n: int, flights: List[List[int]], src: int, dst: int, k: int
    ) -> int:
        costs = [float("inf")] * n

        costs[src] = 0
        for _ in range(k + 1):
            temp = costs.copy()
            for start, end, price in flights:
                if costs[start] != float("inf"):
                    temp[end] = min(costs[start] + price, temp[end])
            costs = temp
        return costs[dst] if costs[dst] != float("inf") else -1
