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

    # 787
    def findCheapestPrice(
        self, n: int, flights: List[List[int]], src: int, dst: int, k: int
    ) -> int:
        """
        TLE
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

    # 787
    def findCheapestPrice(
        self, n: int, flights: List[List[int]], src: int, dst: int, k: int
    ) -> int:
        """_summary_
        [from, to, price]
        """
        visited = {}
        adj = defaultdict(list)
        for fr, to, price in flights:
            adj[fr].append((price, to))

        # cost,  where, # trips,
        heap = [(0, src, 0)]

        while heap:
            cost, city, trips = heapq.heappop(heap)
            if city == dst:
                return cost
            if trips > k:
                continue
            for new_cost, new_dst in adj[city]:
                new_price = new_cost + cost
                if (
                    new_dst not in visited
                    or visited[new_dst][0] > new_price
                    or visited[new_dst][1] > trips + 1
                ):
                    visited[new_dst] = (new_price, trips + 1)
                    heapq.heappush(heap, (new_price, new_dst, trips + 1))
        return -1

    # 2290
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        """mininum obstacles (1's) removed to reach botton right
        [0 1 1 0]
        [1 0 1 0]
        [0 0 1 0]
        """
        ROW = len(grid)
        COL = len(grid[0])
        distances = [[inf] * COL for _ in range(ROW)]
        distances[0][0] = 0
        heap = [(0, (0, 0))]
        directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]

        while heap:
            cur_dist, cur_pos = heapq.heappop(heap)
            r, c = cur_pos
            if cur_pos == (ROW - 1, COL - 1):
                return cur_dist
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < ROW and 0 <= nc < COL:
                    cost = grid[nr][nc] + cur_dist
                    if cost < distances[nr][nc]:
                        heapq.heappush(heap, (cost, (nr, nc)))
                        distances[nr][nc] = cost
        # not found
        return distances[ROW - 1][COL - 1]
