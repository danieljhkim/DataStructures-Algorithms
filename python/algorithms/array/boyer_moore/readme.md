# Boyer Moore Algorithm

--- 

### Majority Element

How to find majority elements in constant time?

Boyer Moore Algo Basis

- There can be at most one majority element which is more than ⌊n/2⌋ times.
- There can be at most two majority elements which are more than ⌊n/3⌋ times.
- There can be at most three majority elements which are more than ⌊n/4⌋ times.

```python
def majorityElement(nums):
    count = 0
    candidate = None
    
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1
    
    return candidate
```