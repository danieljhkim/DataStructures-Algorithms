import heapq
from typing import Optional, List
from itertools import accumulate
import random
from collections import Counter, deque, defaultdict, OrderedDict
import math
import bisect
from math import inf
from functools import cache, lru_cache
from sortedcontainers import SortedSet, SortedList


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

    def maxLength(self, nums: List[int]) -> int:
        """ "
        lcm * gcd = prod(arr)
        lcm = abs(a*b) / gcd * gcd
        abs(a*b) = prod
        """

        N = len(nums)
        arr = [nums[0]]
        size = 1
        for i in range(N - 1):
            arr = [nums[i]]
            prod = nums[i]
            for j in range(i + 1, N):
                arr.append(nums[j])
                prod *= nums[j]
                if math.lcm(*arr) * math.gcd(*arr) == prod:
                    size = max(size, j - i + 1)
        return size

    def calculateScore(self, s: str) -> int:
        rev = []
        for i in range(26):
            rev.append(chr(ord("a") + i))
        rev.reverse()
        score = 0
        table = defaultdict(list)
        marked = set()

        for i, w in enumerate(s):
            idx = rev[ord(w) - ord("a")]
            if w in table and table[w]:
                j = table[w].pop()
                score += i - j
                marked.add(i)
                marked.add(j)
            else:
                table[idx].append(i)
        return score

    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        """_summary_
        [l, r, c]
        l to r contains c coins
        """

        coins.sort()
        ans = 0
        N = len(coins)
        right = 0
        points = 0
        start_pos = coins[0][0]
        cur_pos = start_pos
        while right < N:
            l, r, c = coins[right]
            if cur_pos < l:
                cur_pos = l
            dist = cur_pos - start_pos + 1
            if dist <= k:
                diff = k - dist
                res = min(r - l + 1, diff)
                points += c * res
                dist += res
                cur_pos += res
                ans = max(points, ans)
                if dist == k:
                    points = 0
            else:
                points = 0
                start_pos = l
                cur_pos = l
                continue
            right += 1
        return ans


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
    a = [1, 2]
    print(math.lcm(*a))
