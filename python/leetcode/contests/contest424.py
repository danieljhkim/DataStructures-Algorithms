import heapq
from typing import Optional, List
import random
from collections import Counter, deque, defaultdict, OrderedDict
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

    def countValidSelections(self, nums: List[int]) -> int:
        cur = []
        for i, n in enumerate(nums):
            if n == 0:
                cur.append(i)

        def check(i, isLeft, nums):
            cur = i

            while cur < len(nums) and cur >= 0:
                if nums[cur] == 0:
                    if isLeft:
                        cur -= 1
                    else:
                        cur += 1
                elif nums[cur] > 0:
                    nums[cur] -= 1
                    isLeft = not isLeft
                    if isLeft:
                        cur -= 1
                    else:
                        cur += 1
            if sum(nums) == 0:
                return True
            return False

        ans = 0
        for i in cur:
            left = check(i, True, nums[:])
            if left:
                ans += 1
            right = check(i, False, nums[:])
            if right:
                ans += 1
        return ans

    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        # TLE
        table = {}
        for i, n in enumerate(nums):
            if n > 0:
                table[i] = n
        small = min(table.keys())
        large = max(table.keys())
        q_sum = [0] * len(nums)

        for l, r in queries:
            for i in range(max(l, small), min(r, large) + 1):
                if i in table and table[i] > 0:
                    table[i] -= 1

        count = len(table)
        for k, v in table.items():
            if q_sum[k] >= v:
                count -= 1

        if count > 0:
            return False
        return True

    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        # TLE
        non_zeros = {}
        for i, v in enumerate(nums):
            if v > 0:
                non_zeros[i] = v

        def can_do(k):
            table = defaultdict(int)
            for l, r, v in queries[:k]:
                for i in range(l, r + 1):
                    table[i] += v
            for k, v in non_zeros.items():
                if table[k] < v:
                    return False
            return True

        low = 0
        high = len(queries)

        while low <= high:
            mid = (high + low) // 2
            if can_do(mid):
                high = mid - 1
            else:
                low = mid + 1

        if low <= len(queries) and can_do(low):
            return low
        return -1

    ################## passing solutions ###################

    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:

        def create_prefix_sums():
            psums = [0] * len(nums)
            for l, r in queries:
                psums[l] += 1
                if r + 1 < len(nums):
                    psums[r + 1] -= 1

            for i in range(1, len(psums)):
                psums[i] += psums[i - 1]
            return psums

        psums = create_prefix_sums()
        for i, n in enumerate(nums):
            if n > psums[i]:
                return False
        return True

    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:

        def create_prefix_sums(k):
            psums = [0] * len(nums)
            for l, r, v in queries[:k]:
                psums[l] += v
                if r + 1 < len(nums):
                    psums[r + 1] -= v

            for i in range(1, len(psums)):
                psums[i] += psums[i - 1]
            return psums

        def check(psums):
            for i, n in enumerate(nums):
                if n > psums[i]:
                    return False
            return True

        low = 0
        high = len(queries)
        while low <= high:
            mid = (high + low) // 2
            psums = create_prefix_sums(mid)
            if check(psums):
                high = mid - 1
            else:
                low = mid + 1

        if low <= len(queries):
            return low
        return -1


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
