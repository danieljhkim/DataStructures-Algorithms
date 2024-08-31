# Leetcode Patterns

---

## Table of Contents

1. [Prefix Sum](#prefix-sum)
2. [Two Pointers](#two-pointers)
3. [Sliding Window](#sliding-window)
4. [Fast and Slow Pointers](#fast-and-slow-pointers)
5. [Reversing Linked List](#reversing-linked-list)
6. [Finding Number of Subarrays that Fit a Criteria](#finding-number-of-subarrays-that-fit-a-criteria)
7. [Monotonic Increasing Stack](#monotonic-increasing-stack)
8. [Finding Top k Elements with Heap](#finding-top-k-elements-with-heap)
9. [Binary Search](#binary-search)
10. [Backtracking](#backtracking)
11. [Dynamic Programming: Top-Down Memo](#dynamic-programming-top-down-memo)
12. [Build a Trie](#build-a-trie)
13. [Dijkstra's Algorithm](#dijkstras-algorithm)

---

## Prefix Sum

```python
def prefix_sum(arr):
    if not arr:
        return []
    sums = [arr[0]]
    for i in range(1, len(arr)):
        total = sums[i-1] + arr[i] 
        sums.append(total)
    return sums
```

- Sum in place

```python
def prefix_sum(arr):
    for i in range(1, len(arr)):
        arr[i] += arr[i-1]
    return arr

def sub_arr_sums(sums, i, j):
    return sums[j] - sums[i-1]
```

---

## Two Pointers

- one input, opposite ends 
```python
def two_pointers(arr):
    start = 0
    end = len(arr) - 1
    while start < end:
        if CONDITION: 
            some logic
```

- two inputs, exhaust both

```python
def fn(arr1, arr2):
    i = j = ans = 0

    while i < len(arr1) and j < len(arr2):
        # do some logic here
        if CONDITION:
            i += 1
        else:
            j += 1
    
    while i < len(arr1):
        # do logic
        i += 1
    
    while j < len(arr2):
        # do logic
        j += 1
    
    return ans
```

- 3Sum
- Two Sum II - Input Array is Sorted
- Container with Most water


---

## Sliding Window

```python
def sliding_window():
    left = ans = curr = 0
    for right in range(len(arr)):
        # do logic here to add arr[right] to curr
        curr += arr[right]
        while WINDOW_CONDITION_BROKEN:
            # remove arr[left] from curr
            curr -= arr[left]
            left += 1
        # update ans
    return ans
```

- subarray of size k with max sum

```python
def max_sub_arr(arr, k):
    curr = sum(arr[:k])
    ans = curr
    for right in range(k, len(arr)):
        curr += arr[right] - arr[right - k]
        ans = max(curr, ans)
    return ans
```

- maximum average subarray
- longest substring without repeating characters
- minimum window substring


---

## Fast and slow pointers

```python
def fn(head):
    slow = head
    fast = head
    ans = 0

    while fast and fast.next:
        # do logic
        slow = slow.next
        fast = fast.next.next
    
    return ans
```

- Finding middle

```python
def find_middle(head):
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

---

## Reversing Linked List

```python
def fn(head):
    curr = head
    prev = None
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node 
    return prev
```

---

## Finding # of subarrays that fit a criteria

```python
from collections import defaultdict

def fn(arr, k):
    counts = defaultdict(int)
    counts[0] = 1
    ans = curr = 0

    for num in arr:
        # do logic to change curr
        ans += counts[curr - k]
        counts[curr] += 1
    
    return ans
```

---

## Monotonic increasing stack

```python
def fn(arr):
    stack = []
    ans = 0
    for num in arr:
        # for monotonic decreasing, just flip the > to <
        while stack and stack[-1] > num:
            # do logic
            stack.pop()
        stack.append(num)
    return ans
```
---

## Finding top k elements with heap

```python
import heapq

def fn(arr, k):
    heap = []
    for num in arr:
        # do some logic to push onto heap according to problem's criteria
        heapq.heappush(heap, (CRITERIA, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for num in heap]
```

---

## Binary search

```python
def fn(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            # do something
            return
        if arr[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    
    # left is the insertion point
    return left
```

-  duplicate elements, left-most insertion point

```python
def fn(arr, target):
    left = 0
    right = len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] >= target:
            right = mid
        else:
            left = mid + 1
    return left
```

- duplicate elements, right-most insertion point

```python
def fn(arr, target):
    left = 0
    right = len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] > target:
            right = mid
        else:
            left = mid + 1

    return left
```

- Greedy: minimum

```python
def fn(arr):
    def check(x):
        # this function is implemented depending on the problem
        return BOOLEAN

    left = MINIMUM_POSSIBLE_ANSWER
    right = MAXIMUM_POSSIBLE_ANSWER
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            right = mid - 1
        else:
            left = mid + 1
    
    return left
```

- Greedy: Maximum

```python
def fn(arr):
    def check(x):
        # this function is implemented depending on the problem
        return BOOLEAN

    left = MINIMUM_POSSIBLE_ANSWER
    right = MAXIMUM_POSSIBLE_ANSWER
    while left <= right:
        mid = (left + right) // 2
        if check(mid):
            left = mid + 1
        else:
            right = mid - 1
    
    return right
```

---

## Backtracking

```python
def backtrack(curr, OTHER_ARGUMENTS...):
    if (BASE_CASE):
        # modify the answer
        return
    
    ans = 0
    for (ITERATE_OVER_INPUT):
        # modify the current state
        ans += backtrack(curr, OTHER_ARGUMENTS...)
        # undo the modification of the current state
    return ans
```

---

## Dynamic Programming: top-down memo

```python
def fn(arr):
    def dp(STATE):
        if BASE_CASE:
            return 0
        
        if STATE in memo:
            return memo[STATE]
        
        ans = RECURRENCE_RELATION(STATE)
        memo[STATE] = ans
        return ans

    memo = {}
    return dp(STATE_FOR_WHOLE_INPUT)
```

---

## Build a trie

```python
# note: using a class is only necessary if you want to store data at each node.
# otherwise, you can implement a trie using only hash maps.
class TrieNode:
    def __init__(self):
        # you can store data at nodes if you wish
        self.data = None
        self.children = {}

def fn(words):
    root = TrieNode()
    for word in words:
        curr = root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        # at this point, you have a full word at curr
        # you can perform more logic here to give curr an attribute if you want
    
    return root
```

---

## Dijkstra's algorithm

```python
from math import inf
from heapq import *

distances = [inf] * n
distances[source] = 0
heap = [(0, source)]

while heap:
    curr_dist, node = heappop(heap)
    if curr_dist > distances[node]:
        continue
    
    for nei, weight in graph[node]:
        dist = curr_dist + weight
        if dist < distances[nei]:
            distances[nei] = dist
            heappush(heap, (dist, nei))
```
