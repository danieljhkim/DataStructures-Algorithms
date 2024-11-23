"""Bellman-Ford Algorithm
    - slower than djkra but good when there is negative weights
"""

import heapq
from typing import Dict, List
from math import inf


class BellmanFord:

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
