from collections import defaultdict, deque
from typing import List, Tuple, Optional


class CombinationSum:
    """
    - we have a list of distinct intergers
    - find all combinations where sum equals target
    """

    def more_than_one_element(self, nums: List[int], target: int) -> List[int]:
        """
        - an element can be used multiple times
        - time: O(n^target/min(n))
        """
        ans = []

        def backtrack(arr: list[int], prev_idx: int, total: int) -> None:
            if total == target:
                ans.append(arr[:])
                return
            elif total > target:
                return
            for i in range(prev_idx, len(nums)):
                arr.append(nums[i])
                backtrack(arr, i, total + nums[i])
                arr.pop()

        backtrack([], 0, 0)
        return ans

    def subset_single_element(self, nums: list, target: int):
        """
        - an element can be used just once
        - this should ideally be solved this via sorting and sliding window
        - time: O(2^n)
        """
        ans = []

        def backtrack(arr: list, prev_idx: int, total: int):
            if total == target:
                ans.append(arr[:])
            elif total > target:  # if all positive numbers
                return
            for i in range(prev_idx, len(nums)):
                arr.append(nums[i])
                backtrack(arr, i + 1, total + nums[i])
                arr.pop()

        backtrack([], 0, 0)
        return ans

    def subset_single_element_size_limit(self, nums: list, target: int, size: int):
        """
        - subsets with a size limit: less than equal to size
        - time: O(n^k)
        """
        ans = []

        def backtrack(arr: list, total: int, prev_idx: int):
            if len(arr) > size:
                return
            if total == target:
                ans.append(arr[:])
            elif total > target:
                return
            for i in range(prev_idx, len(nums)):
                arr.append(nums[i])
                backtrack(arr, target + nums[i], i + 1)
                arr.pop()

        backtrack([], 0, 0)
        return ans


if __name__ == "__main__":
    combsum = CombinationSum()
