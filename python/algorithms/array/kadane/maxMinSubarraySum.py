"""Kadane's Algorithm
    - finds the largest sum of a contiguous subarray
"""


def max_kadane(arr):
    max_sum = float("-inf")
    current_sum = 0

    for num in arr:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum


def min_kadane(arr):
    min_sum = float("inf")
    current_sum = 0

    for num in arr:
        current_sum = min(num, current_sum + num)
        min_sum = min(min_sum, current_sum)

    return min_sum


def dynamic_p(arr):
    N = len(arr)
    dp = [0] * N
    dp[0] = arr[0]
    top = arr[0]
    for i in range(1, N):
        num = arr[i]
        dp[i] = max(dp[i - 1] + num, num)
        top = max(top, dp[i])
    return top


def maxSubarraySumCircular(nums) -> int:
    """
    - circular array
    """
    cur_max = 0
    cur_min = 0
    max_sum = nums[0]
    min_sum = nums[0]
    N = len(nums)
    for n in nums:
        cur_max = max(cur_max + n, n)
        max_sum = max(cur_max, max_sum)
        cur_min = min(cur_min + n, n)
        min_sum = min(cur_min, min_sum)
    total = sum(nums)
    if min_sum == total:
        return max_sum
    return max(max_sum, total - min_sum)
