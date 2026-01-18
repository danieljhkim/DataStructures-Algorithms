# Prefix Sum

---

How to find the sum of elements in a given range?

Prefix sum is a technique used to find the sum of elements in a given range in constant time.

```python
def prefixSum_find_sum_of_range(nums, start, end):
    N = len(nums)
    prefix = [nums[0]]
    
    for n in nums[1:]:
        prefix.append(prefix[-1] + n)

    if start == 0:
        return prefix[end]
    else:
        return prefix[end] - prefix[start - 1]
    
    return prefix
```

This is useful when we need to query the sum of elements in a given range multiple times - reducing the time complexity from O(n) to O(1).
