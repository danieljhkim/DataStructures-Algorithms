from math import inf
from collections import defaultdict, deque
from typing import List, Tuple, Optional
import heapq
import math


class DP:

    def maxScore(
        self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]
    ) -> int:
        """
        stayScore[day][city] = points
        travelScore[cur city][next city] = points
        """
        dp = [[0] * n for _ in range(k + 1)]

        for day in range(1, k + 1):
            for city in range(n):
                stay_points = dp[day - 1][city] + stayScore[day - 1][city]
                max_travel_points = stay_points
                for dest in range(n):
                    # here we want to find the points from other city -> current city
                    if city == dest:
                        continue
                    points_from_other_city_to_cur = (
                        dp[day - 1][dest] + travelScore[dest][city]
                    )
                    max_travel_points = max(
                        max_travel_points, points_from_other_city_to_cur
                    )
                dp[day][city] = max_travel_points
        return max(dp[-1])

    def knapsack(weights, values, W):
        n = len(weights)
        dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
        for i in range(1, n + 1):
            for w in range(1, W + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(
                        dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1]
                    )
                else:
                    dp[i][w] = dp[i - 1][w]

        return dp[n][W]
