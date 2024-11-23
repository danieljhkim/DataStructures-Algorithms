import heapq
from typing import Optional, List
from collections import Counter


def partition(arr: list[int], low: int, high: int) -> int:
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] >= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quick_select_idx(arr: list[int], position: int) -> int:
    low, high = 0, len(arr) - 1
    while low <= high:
        if low > len(arr) - 1 or high < 0:
            return -1
        pivot_index = partition(arr, low, high)
        if pivot_index == position - 1:
            return arr[pivot_index]
        elif pivot_index > position - 1:
            high = pivot_index - 1
        else:
            low = pivot_index + 1
    return -1


def quick_select(self, nums: List[int], k: int) -> int:
    pivot = nums[-1]
    left = []
    right = []
    pivots = []
    for n in nums:
        if n < pivot:
            left.append(n)
        elif n > pivot:
            right.append(n)
        else:
            pivots.append(n)
    if len(right) >= k:
        return quick_select(right, k)
    if len(right) + len(pivots) < k:
        return quick_select(left, k - len(right) - len(pivots))

    return pivot


def counting_sort(nums: List[int], k: int) -> int:
    min_val = min(nums)
    max_val = max(nums)
    diff = max_val - min_val
    slots = [0] * (diff + 1)
    for n in nums:
        pos = n - min_val
        slots[pos] += 1
    for i in range(len(slots) - 1, -1, -1):
        cur = slots[i]
        k -= cur
        if k <= 0:
            return min_val + i


def topKFrequent_count_sort(nums, k):
    counts = Counter(nums)
    freq = list(counts.values())
    max_freq = max(freq)
    min_freq = min(freq)
    slots = [[] for _ in range(max_freq - min_freq + 1)]
    for key, val in counts.items():
        idx = val - min_freq
        slots[idx].append(key)

    ans = []
    for slot in reversed(slots):
        for n in slot:
            ans.append(n)
            if len(ans) == k:
                return ans
    return ans


def topKFrequent_bucket_sort(nums, k):
    buckets = [[] for _ in range(len(nums) + 1)]
    counts = Counter(nums).items()
    for num, freq in counts:
        buckets[freq].append(num)
    result = []
    for bucket in reversed(buckets):
        for n in bucket:
            result.append(n)
            if len(result) == k:
                return result
    return result
