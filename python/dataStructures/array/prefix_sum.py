from typing import List, Tuple, Optional
from collections import defaultdict


class PrefixSum:

    def __init__(self, nums: List[int]):
        self.prefix_sum = []
        current_sum = 0
        for num in nums:
            current_sum += num
            self.prefix_sum.append(current_sum)

    def get_sum(self, start: int, end: int) -> int:
        if start == 0:
            return self.prefix_sum[end]
        return self.prefix_sum[end] - self.prefix_sum[start - 1]

    def contains_sum(self, target_sum: int) -> bool:
        sums = {0}
        for sum_item in self.prefix_sum:
            if sum_item - target_sum in sums:
                return True
            sums.add(sum_item)

        return False

    def contains_sum(self, target_sum: int) -> Optional[Tuple[int, int]]:
        sums = {
            0: -1
        }  # Initialize with 0 sum at index -1 to handle subarrays starting from index 0
        for i, sum_item in enumerate(self.prefix_sum):
            if sum_item - target_sum in sums:
                start_index = sums[sum_item - target_sum] + 1
                end_index = i
                return start_index, end_index
            sums[sum_item] = i
        return None

    def total_subarray_of_sum_k(self, nums: List[int], k: int) -> int:
        """_summary_
        number of subarrays that sum up to k
        """
        prefix_sum_count = defaultdict(int)
        prefix_sum_count[0] = 1
        total = 0
        ans = 0

        for num in nums:
            total += num
            if total - k in prefix_sum_count:
                ans += prefix_sum_count[total - k]
            prefix_sum_count[total] += 1

        return ans
