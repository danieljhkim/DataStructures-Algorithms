"""TOP DOWN
Memoization
"""


def fn(arr):
    BASE_CASE = True

    def dp(i):
        if BASE_CASE:
            return 0
        if i in memo:
            return memo[i]
        ans = dp(i - 2) + dp(i - 1)
        memo[i] = ans
        return ans

    memo = {}
    return dp(10)


"""BOTTOM UP
Tabulation
"""
