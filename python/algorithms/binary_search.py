"""
    low <= high:
        - Typical Use Case: 
            when searching for a specific target value in a sorted array.
        
    low < high:
        - Typical Use Case: 
            where you are looking for a boundary or a specific condition rather than a specific value. 
            i.e., finding the smallest or largest index that satisfies a certain condition.


    high = mid
    low = mid + 1
        - when you are looking for a boundary condition, 
        - i.e. finding the smallest element that is greater than or equal to the target.
        - i.e. finding the first occurence
        - In this case, you do not want to exclude the middle element from the search range because it might be the boundary you are looking for.
        
    low = mid
    high = mid - 1
        - when you are looking for a boundary condition, 
        - i.e. finding the largest element that is less than or equal to the target
        - i.e. finding the last occurence
        - In this case, you do not want to exclude the middle element from the search range because it might be the boundary you are looking for.
    
    
    math.ceil(n)
        - low + (high - low + 1) // 2
"""


def binary_search(arr, target):
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


def find_first_occurrence(nums, target):
    low, high = 0, len(nums) - 1
    while low < high:
        mid = low + (high - low) // 2
        if nums[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low if low < len(nums) and nums[low] == target else -1


def find_largest_less_than_or_equal(nums, target):
    low, high = 0, len(nums) - 1
    while low < high:
        mid = (high - low + 1) // 2 + low  # Use (high - low + 1) to avoid infinite loop
        if nums[mid] <= target:
            low = mid
        else:
            high = mid - 1
    return low if nums[low] <= target else -1


def binary_search_insert_position(arr, target):
    # useful for when inserting
    low, high = 0, len(arr)
    while low < high:
        mid = (low + high) // 2
        if arr[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low


def recursive_search(arr, target, low, high):

    if low <= high:
        mid = (high - low) // 2 + low
        if target == arr[mid]:
            return mid
        elif target < arr[mid]:
            high = mid - 1
            return recursive_search(arr, target, low, high)
        else:
            low = mid + 1
            return recursive_search(arr, target, low, high)
    else:
        return -1


def recursive_search_2(arr, target):

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
            return recursive_search_2(arr[:high], target)
        else:
            low = mid + 1
            return_index = recursive_search_2(arr[low:], target)
            if return_index >= 0:
                return mid + return_index + 1
            else:
                return return_index


a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(recursive_search_2(a, 11))
