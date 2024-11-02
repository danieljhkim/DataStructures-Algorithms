from typing import Optional, List
from functools import lru_cache


class Solution:

    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float("inf")] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != float("inf") else -1

    def coinChange(self, coins: List[int], amount: int) -> int:
        cache = {}

        def dfs(rem):
            if rem in cache:
                return cache[rem]
            if rem < 0:
                return -1
            if rem == 0:
                return 0
            min_cost = float("inf")
            for coin in coins:
                res = dfs(rem - coin)
                if res != -1:
                    min_cost = min(min_cost, res + 1)
            output = min_cost if min_cost != float("inf") else -1
            cache[rem] = output
            return output

        return dfs(amount)
