"""
    low < high or low <= high
        - if you discard mid (i.e. low = mid+1 or high = mid-1) then (low <= high).
        - if you keep mid (i.e. low = mid or high = mid) then (llow < high)
        
    math.ceil(n)
        - low + (high - low + 1) // 2
"""


class BinarySearch:

    def binary_search(self, arr, target):
        low = 0
        high = len(arr) - 1
        while low <= high:
            mid = (high - low) // 2 + low
            if target < arr[mid]:
                high = mid - 1
            elif target > arr[mid]:
                low = mid + 1
            else:
                return mid
        return -1

    def find_first_occurrence(self, nums, target):
        low, high = 0, len(nums) - 1
        while low < high:
            mid = low + (high - low) // 2
            if nums[mid] < target:
                low = mid + 1
            else:
                high = mid
        return low if low < len(nums) and nums[low] == target else -1

    def find_largest_less_than_or_equal(self, nums, target):
        low, high = 0, len(nums) - 1
        while low < high:
            mid = (
                high - low + 1
            ) // 2 + low  # Use (high - low + 1) to avoid infinite loop
            if nums[mid] <= target:
                low = mid
            else:
                high = mid - 1
        # return high for largest less than or equal
        return low if nums[low] <= target else -1

    def find_smallest_greater_than_target(self, nums, target):
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = (high - low) // 2 + low
            if nums[mid] <= target:
                low = mid + 1
            else:
                high = mid - 1
        return low if low < len(nums) and nums[low] > target else -1

    def find_smallest_greater_or_equal(self, nums, target):
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = (high - low) // 2 + low
            if nums[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        # also could return high + 1
        # high would be the largest element less than the target
        return low if low < len(nums) and nums[low] >= target else -1

    def binary_search_insert_position(self, arr, target):
        # useful for when inserting
        low, high = 0, len(arr)
        while low < high:
            mid = (low + high) // 2
            if arr[mid] < target:
                low = mid + 1
            else:
                high = mid
        return low

    def recursive_search(self, arr, target, low, high):
        if low <= high:
            mid = (high - low) // 2 + low
            if target == arr[mid]:
                return mid
            elif target < arr[mid]:
                high = mid - 1
                return self.recursive_search(arr, target, low, high)
            else:
                low = mid + 1
                return self.recursive_search(arr, target, low, high)
        else:
            return -1

    def recursive_search_2(self, arr, target):
        if len(arr) == 1:
            if arr[0] == target:
                return 0
            else:
                return -1
        else:
            mid = (len(arr) - 1) // 2
            if target == arr[mid]:
                return mid
            elif target < arr[mid]:
                high = mid
                return self.recursive_search_2(arr[:high], target)
            else:
                low = mid + 1
                return_index = self.recursive_search_2(arr[low:], target)
                if return_index >= 0:
                    return mid + return_index + 1
                else:
                    return return_index
