# Search

### Binary Search
I think binary search algo is such a fundemental algorithm that all swe should master. Although it seems simple, it took me some time to truly grasp the concept, especially the lower and upper bound search.

```python
def find_lower_bound(nums, target):
    low = 0 
    high = len(nums) - 1

    while low <= high:
        mid = (low + high) // 2
        # if we do "nums[mid] <= target" here, we will get the upper bound, 
        # because we will move the low up when nums[mid] == target
        if nums[mid] < target: 
            low = mid + 1
        else:
            high = mid
    return low
```

This example helps me to visualize when doing a boundary search.

```python
target = 3

[1, 2, 3, 3, 3, 4, 5]
 l        m        h

[1, 2, 3, 3, 3, 4, 5]
 l  m     h

[1, 2, 3, 3, 3, 4, 5]
    l  m  h

[1, 2, 3, 3, 3, 4, 5]   
    l  h
```

### Kth Largest/Smallest Element in Array

##### Solution 1: Quick Select

```python
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
```

##### Solution 2: Bcuket Sort

```python
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

```

##### Solution 3: Count Sort

```python
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

```