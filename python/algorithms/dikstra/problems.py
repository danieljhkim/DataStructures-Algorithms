import heapq
from typing import Dict, List
from math import inf
from collections import defaultdict


class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        """_summary_
        - 0: empty
        - 1: building
        - 2: obstacle

        [0 0 0 1 0]
        [1 0 2 0 1]
        [0 0 1 0 0]
        """

    def findCheapestPrice(
        self, n: int, flights: List[List[int]], src: int, dst: int, k: int
    ) -> int:
        """_summary_
        flights[i] = [from, to, price]
        """

        adj = defaultdict(list)
        for fr, to, pr in flights:
            adj[fr].append((to, pr))

        # (cost, current_node, stops)
        heap = [(0, src, 0)]
        while heap:
            cost, node, stops = heapq.heappop(heap)
            if node == dst:
                return cost
            if stops <= k:
                for neighbor, price in adj[node]:
                    heapq.heappush(heap, (cost + price, neighbor, stops + 1))

        return -1
