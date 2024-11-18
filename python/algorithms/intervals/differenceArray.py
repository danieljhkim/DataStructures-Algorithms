"""Difference Array algo

- given a list of intervals [start, end], and nums array, create a prefix sum arr in a linear time
    1. psums[start] + value
    2. psums[end] - value
    3. psum[i] += psum[i - 1]

"""


def create_prefix_sums_value(n, intervals: list):
    """_summary_
    - intervals = [start, end, value]
    """
    psums = [0] * n
    for l, r, v in intervals:
        psums[l] += v
        if r + 1 < n:
            psums[r + 1] -= v

    for i in range(1, len(psums)):
        psums[i] += psums[i - 1]
    return psums


def create_prefix_sums_1(n, intervals):
    """_summary_
    - intervals = [start, end]
    - add 1 from start_i to end_i
    """
    psums = [0] * n
    for l, r in intervals:
        psums[l] += 1
        if r + 1 < n:
            psums[r + 1] -= 1

    for i in range(1, len(psums)):
        psums[i] += psums[i - 1]
    return psums
