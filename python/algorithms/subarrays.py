from typing import Optional, List, OrderedDict


def subarraySum_connections(nums: List[int], k: int) -> int:
    """_summary_
    number of k running sum connections
    """
    ans = 0
    found = 1
    total = 0
    for i in range(len(nums)):
        total += nums[i]
        if total == k:
            ans += found
            found += 1
    return ans
