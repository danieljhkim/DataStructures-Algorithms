from ast import List
from math import inf
from typing import Optional
from collections import defaultdict
from typing import List
import heapq


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

    def getSneakyNumbers(self, nums: List[int]) -> List[int]:
        n_map = defaultdict(int)
        ans = []
        for i, num in enumerate(nums):
            n_map[num] += 1
            if n_map[num] > 1:
                ans.append(num)
                if len(ans) == 2:
                    break
        return ans

    def maxScore(self, a: List[int], b: List[int]) -> int:
        dp = [-inf] * 5
        dp[0] = 0
        for x in b:
            for i in range(4, 0, -1):
                dp[i] = max(dp[i], dp[i - 1] + a[i - 1] * x)
        return dp[-1]

    def maxScore(self, a, b):
        # Initialize dp array for the 4 selections, starting with the worst case (-inf)
        dp = [-float("inf")] * 4

        # Traverse each element in b
        for i in range(len(b)):
            # Update dp array in reverse to avoid overwriting during this iteration
            for j in range(3, -1, -1):
                if j == 0:
                    # For the first selection, just choose a[0] * b[i]
                    dp[0] = max(dp[0], a[0] * b[i])
                else:
                    # For the subsequent selections, consider previous best dp[j-1]
                    dp[j] = max(dp[j], dp[j - 1] + a[j] * b[i])
                print(dp)

        # The maximum score after choosing 4 elements will be stored in dp[3]
        return dp[3]


def test_solution():
    s = Solution()
    a = [3, 2, 5, 6]
    b = [2, -6, 4, -5, -3, 2, -7]
    print(s.maxScore(a, b))


if __name__ == "__main__":
    test_solution()
