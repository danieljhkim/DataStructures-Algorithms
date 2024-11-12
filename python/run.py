from math import inf
from collections import Counter, defaultdict, deque
from typing import List, Tuple, Optional
import heapq
import math


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Node:

    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    
    def smallestNumber(self, n: int, t: int) -> int:
        while True:
            prod = 1
            temp = n
            while temp > 0:
                prod *= temp % 10
                temp //= 10
            diff = prod % t
            if diff == 0:
                return n
            n += 1

    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        """_summary_
        2, 3, 1, 1, 1, 20, 20, 20, 20
        
        k = 10
        n = 2
        """
        counts = Counter(nums)
        sort_arr = [(v, k) for k,v in counts.items()]
        sort_arr.sort(key=lambda x:[0])
        top_f = sort_arr[-1][0]
        top_val = sort_arr[-1][1]
        while numOperations:
            tops = len(sort_arr) - 1
            top_f = sort_arr[tops]
            top_val = sort_arr[tops]
            while tops > 0:
    
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        """
        1,2,3, 2, 2,3,4
        """
        freqs = []
        freq = 1
        for i in range(1, len(nums)):
            prev = nums[i - 1]
            cur = nums[i]
            if cur > prev:
                freq += 1
                if freq > k:
                    freqs.append(freq - k)
                    freqs.append(k)
                else:
                    freqs.append(freq)
            else:
                if freq > k:
                    freqs.append(freq - k)
                    freqs.append(k)
                else:
                    freqs.append(freq)

                freq = 1
        for i in range(1, len(freqs)):
            prev = freqs[i - 1]
            cur = freqs[i]
            if cur >= k and prev >= k:
                return True
        return False

def test_solution():
    s = Solution()



if __name__ == "__main__":
    test_solution()
