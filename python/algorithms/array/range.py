def maxElementsWithinRange(nums: list[int], k: int) -> int:
    """ "
    - numbers within range of (n - k, n + k)
    """
    if len(nums) == 1:
        return 1

    max_value = max(nums)
    count = [0] * (max_value + 1)

    for num in nums:  # line sweep
        count[max(num - k, 0)] += 1
        if num + k + 1 <= max_value:
            count[num + k + 1] -= 1

    max_count = 0
    current_sum = 0

    for val in count:
        current_sum += val
        max_count = max(max_count, current_sum)

    return max_count
