from typing import Optional, List, OrderedDict
from collections import defaultdict


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


def subarrays_sums_multiples_of_k(self, nums: List[int], k: int) -> bool:
    """_summary_
    2, 3, 7, 3 - 5
    3, 2, 2, 3
    3, 5, 7, 10

    1, 2, 3, 4, 5 - 5
    1, 2, 3, 4, 0

    1, 3, 6, 10, 10
    """
    p_sums = defaultdict(list)
    p_sums[0].append(-1)
    total = 0
    result = []
    for i, n in enumerate(nums):
        rem = n % k
        total += rem
        p_sums[total].append(i + 1)
        if total % k in p_sums:
            result.append([p_sums[total - k], i])

    return result


def subarrays_sums_multiples_of_k(nums: List[int], k: int) -> List[List[int]]:
    """
    - 1 2 3 4 5
    sums: 1 3 6 10 15 - 5
    rems: 1 3 1

    -
    """
    p_sums = defaultdict(list)
    p_sums[0].append(-1)
    total = 0
    result = []

    for i, n in enumerate(nums):
        total += n
        rem = total % k
        if rem in p_sums:
            for start_index in p_sums[rem]:
                result.append([start_index + 1, i])
        p_sums[rem].append(i)
    print(result)
    return result
