from typing import Optional, List, OrderedDict

table = OrderedDict()


def next_greater_permutation(nums: List[int]) -> None:
    """_summary_
    1 2 3 4 -> 1 2 4 3

    4 3 2 1 -> 1 2 3 4

    1 2 4 3 -> 1 3 4 2
    """

    n = len(nums)
    dec_idx = None
    # traverse from right, and find first decreasing
    for i in range(len(nums) - 1, 0, -1):
        prev = nums[i - 1]
        cur = nums[i]
        if prev < cur:
            dec_idx = i - 1
            break
    if dec_idx is None:  # all are increasing, so reverse
        nums.reverse()
        return
    i = n - 1
    while nums[dec_idx] >= nums[i]:  # find the first bigger
        i -= 1
    nums[dec_idx], nums[i] = nums[i], nums[dec_idx]
    nums[dec_idx + 1 :] = reversed(nums[dec_idx + 1 :])
