from typing import *
from functools import cache


# 1931. Painting a Grid With Three Different Colors
# DP, masking
class Solution:
    """
    You are given two integers m and n. Consider an m x n grid where each cell is initially white.
    You can paint each cell red, green, or blue. All cells must be painted.
    Return the number of ways to color the grid with no two adjacent cells having the same color.
    Since the answer can be very large, return it modulo 109 + 7.
    """

    # BOTTOM-UP
    def colorTheGrid(self, R: int, C: int) -> int:
        states = []
        MOD = 10**9 + 7

        def dfs(row, prev, mask):
            if row == R:
                states.append(mask)
                return
            for color in range(3):
                if color != prev:
                    dfs(row + 1, color, mask * 3 + color)

        dfs(0, -1, 0)
        S = len(states)
        combs = [[] for _ in range(S)]
        for i in range(S):
            for j in range(S):
                if i == j:
                    continue
                good = True
                n_i, n_j = states[i], states[j]
                for r in range(R):
                    if n_i % 3 == n_j % 3:  # same color
                        good = False
                        break
                    n_i //= 3
                    n_j //= 3
                if good:
                    combs[i].append(j)

        dp = [[0] * S for _ in range(C)]
        for c in range(C):
            if c == 0:
                for i in range(S):
                    dp[0][i] = 1
                continue
            for i in range(S):
                for j in combs[i]:
                    dp[c][j] = (dp[c - 1][i] + dp[c][j]) % MOD
        return sum(dp[-1]) % MOD

    # TOP-DOWN
    def colorTheGrid(self, R: int, C: int) -> int:
        states = []
        MOD = 10**9 + 7

        def dfs(row, prev, mask):
            if row == R:
                states.append(mask)
                return
            for color in range(3):
                if color != prev:
                    dfs(row + 1, color, mask * 3 + color)

        dfs(0, -1, 0)
        S = len(states)
        combs = [[] for _ in range(S)]
        for i in range(S):
            for j in range(S):
                if i == j:
                    continue
                good = True
                n_i, n_j = states[i], states[j]
                for r in range(R):
                    if n_i % 3 == n_j % 3:
                        good = False
                        break
                    n_i //= 3
                    n_j //= 3
                if good:
                    combs[i].append(j)

        @cache
        def dp(prev, c):
            if c == C:
                return 1
            res = 0
            for i, state in enumerate(states):
                if prev == -1 or i in combs[prev]:
                    res = (res + dp(i, c + 1)) % MOD
            return res % MOD


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
