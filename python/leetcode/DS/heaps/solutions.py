from ast import List
from typing import Optional, Tuple
import heapq


class ListNode:
    def __init__(self, val=0, next=None, prev=None):
        self.val = val
        self.next = next
        self.prev = prev


class Solution:

    def findKthLargest(self, nums: List[int], k: int) -> int:
        # O(n 2logk)
        heap = nums[:k]
        heapq.heapify(heap)  # O(k)
        for num in nums[k:]:
            if num > heap[0]:
                heapq.heappop(heap)  # O(logk)
                heapq.heappush(heap, num)  # O(logk)
        return heap[0]

    # 3275. K-th Nearest Obstacle Queries
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        # time limit exceeded

        def insertion(arr: List[Tuple[int, int]], num: Tuple[int, int]):
            if len(arr) > k:
                arr.pop()
            lo = 0
            hi = len(arr) - 1
            target = num[1]
            while lo <= hi:
                mid = (hi + lo) // 2
                if arr[mid][1] > target:
                    hi = mid - 1
                elif arr[mid][1] < target:
                    lo = mid + 1
                else:
                    arr.insert(mid, num)
                    lo = -1
                    break
            if lo >= 0:
                arr.insert(lo, num)

        distances = []
        ans = []
        for i, v in enumerate(queries):
            dist = abs(v[0]) + abs(v[1])
            if i + 1 < k:
                insertion(distances, (i, dist))
                ans.append(-1)
            else:
                if len(distances) < k or distances[k - 1][1] > dist:
                    insertion(distances, (i, dist))
                ans.append(distances[k - 1][1])
        return ans

    # 3275. K-th Nearest Obstacle Queries
    def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
        heap = []
        ans = []
        for i, v in enumerate(queries):
            dist = -(abs(v[0]) + abs(v[1]))
            if len(heap) < k:
                heapq.heappush(heap, dist)
            elif heap[0] < dist:
                heapq.heappop(heap)
                heapq.heappush(heap, dist)
            if i < k - 1:
                ans.append(-1)
            else:
                ans.append(-heap[0])
        return ans
