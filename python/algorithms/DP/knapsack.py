from math import inf
from collections import defaultdict, deque
from typing import List, Tuple, Optional
import math


class KnapSack:

    def knapsack(self, weights: list[int], values: list[int], W: int):
        n = len(weights)
        dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
        for i in range(1, n + 1):
            for w in range(1, W + 1):
                if weights[i - 1] <= w:
                    max_val_of_remaining_w = dp[i - 1][w - weights[i - 1]]
                    val_of_including_cur = values[i - 1] + max_val_of_remaining_w
                    val_of_excluding_cur = dp[i - 1][w]
                    dp[i][w] = max(val_of_excluding_cur, val_of_including_cur)
                else:
                    dp[i][w] = dp[i - 1][w]
                print(dp)

        return dp[n][W]

    def fractional_knapsack(weights, values, W):
        n = len(weights)
        items = [(values[i] / weights[i], weights[i], values[i]) for i in range(n)]
        items.sort(reverse=True, key=lambda x: x[0])  # Sort by value-to-weight ratio
        total_value = 0
        for ratio, weight, value in items:
            if W >= weight:
                W -= weight
                total_value += value
            else:
                total_value += ratio * W
                break
        return total_value

    def unbounded_knapsack(weights, values, W):
        n = len(weights)
        dp = [0] * (W + 1)
        for w in range(W + 1):
            for i in range(n):
                if weights[i] <= w:
                    dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
        return dp[W]

    def bounded_knapsack(weights, values, quantities, W):
        n = len(weights)
        dp = [0] * (W + 1)

        for i in range(n):
            for _ in range(quantities[i]):
                for w in range(W, weights[i] - 1, -1):
                    dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

        return dp[W]


if __name__ == "__main__":
    weights = [5, 10, 15, 20]
    values = [7, 10, 20, 30]
    W = 30
    dp = KnapSack()
    dp.knapsack(weights, values, W)
