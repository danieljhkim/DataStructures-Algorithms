import heapq
from typing import Optional, List, OrderedDict
import random
from collections import Counter, deque, defaultdict
import math


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


def subarraySum_from_beginning(self, nums: List[int], k: int) -> int:
    """
    Find the number of subarrays starting from the beginning that sum up to k.
    """
    ans = 0
    total = 0
    for i in range(len(nums)):
        total += nums[i]
        if total == k:
            ans += 1
    return ans
