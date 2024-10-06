from typing import Optional, List
import math


class Solution:

    # 704. Binary Search
    def search(self, nums: List[int], target: int) -> int:
        high = len(nums) - 1
        low = 0
        while high >= low:
            mid = low + (high - low) // 2
            cur = nums[mid]
            if cur < target:
                low = mid + 1
            elif cur > target:
                high = mid - 1
            else:
                return mid
        return -1

    # 74. Search a 2D Matrix
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        row = len(matrix)
        col = len(matrix[0])
        low = 0
        high = row * col - 1
        while low <= high:
            mid = low + (high - low) // 2
            cur = matrix[mid // col][mid % col]
            if cur < target:
                low = mid + 1
            elif cur > target:
                high = mid - 1
            else:
                return True
        return False

    # 875. Koko Eating Bananas
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        def munch_munch(k):
            time = 0
            for banany in piles:
                time += math.ceil(banany / k)
            if time <= h:
                return True
            return False

        low = 1
        high = sum(piles)
        while low < high:
            mid = (high + low) // 2
            if not munch_munch(mid):
                low = mid + 1
            else:
                high = mid
        return low

    # 153. Find Minimum in Rotated Sorted Array
    def findMin(self, nums: List[int]) -> int:
        """_summary_
        7 6 5 4 3 2 1 0
        """
        n = len(nums)
        if n == 1:
            return nums[0]
        if nums[0] < nums[-1]:
            return nums[0]
        else:
            low, high = 0, n - 1
            while low < high:
                mid = (high + low) // 2
                if nums[mid] > nums[high]:
                    low = mid + 1
                else:
                    high = mid
        return nums[low]
