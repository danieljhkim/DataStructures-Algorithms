import heapq
import random
import math
import bisect
from typing import *
from math import inf, factorial, gcd, lcm
from functools import lru_cache, cache
from heapq import heapify, heappush, heappop
from itertools import accumulate, permutations, combinations
from collections import Counter, deque, defaultdict, OrderedDict
from sortedcontainers import SortedSet, SortedList, SortedDict


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

    # 1415. The k-th Lexicographical String of All Happy Strings of Length n
    def getHappyString(self, n: int, k: int) -> str:
        options = "abc"
        ans = []

        def backtrack(arr):
            if len(arr) == n:
                ans.append("".join(arr))
                return
            if len(ans) >= k:
                return

            for i in range(3):
                if not arr or (arr and arr[-1] != options[i]):
                    arr.append(options[i])
                    backtrack(arr)
                    arr.pop()

        backtrack([], 0)
        if len(ans) < k:
            return ""
        return ans[k - 1]

    # 2375. Construct Smallest Number From DI String
    def smallestNumber(self, pattern: str) -> str:
        N = len(pattern)
        self.ans = inf
        avail = deque([i for i in range(1, 10)])

        def backtrack(arr, idx, avail):
            if self.ans != inf:
                return
            if idx == N:
                self.ans = min(self.ans, int("".join(arr)))
                return
            cur = pattern[idx]
            prev = int(arr[-1])
            size = len(avail)
            for _ in range(size):
                n = avail.popleft()
                if (cur == "D" and prev > n) or (cur == "I" and prev < n):
                    arr.append(str(n))
                    backtrack(arr, idx + 1, avail)
                    arr.pop()
                avail.append(n)

        for i in range(9, 0, -1):
            avail.remove(i)
            backtrack([str(i)], 0, avail)
            if self.ans != inf:
                return str(self.ans)
            avail.insert(i - 1, i)

    def numTilePossibilities(self, tiles: str) -> int:
        N = len(tiles)
        ans = set()

        def backtrack(arr, used):
            if arr:
                ans.add("".join(arr))
            if len(used) == N:
                return
            for i in range(N):
                if i not in used:
                    w = tiles[i]
                    arr.append(w)
                    used.add(i)
                    backtrack(arr, used)
                    arr.pop()
                    used.remove(i)

        backtrack([], set())
        return len(ans)

    # 1718. Construct the Lexicographically Largest Valid Sequence
    def constructDistancedSequence(self, x: int) -> List[int]:
        ans, options = [], []
        for i in range(1, x + 1):
            options.append(i)
        N = (x - 1) * 2 + 1
        arr = [0] * N
        options.reverse()
        used = [0] * x

        def backtrack(arr, idx, used):
            if sum(used) == x:
                ans.append(tuple(arr))
                return
            if ans:
                return
            if arr[idx] != 0:
                backtrack(arr, idx + 1, used)
                return

            for i, n in enumerate(options):
                if n == 1 and used[n] == 0:
                    arr[idx] = n
                    used[n] = 1
                    backtrack(arr, idx + 1, used)
                    arr[idx] = 0
                    used[n] = 0
                else:
                    dist = idx + n
                    if dist >= N:
                        continue
                    if arr[dist] == 0 and used[n] == 0:
                        arr[dist] = n
                        arr[idx] = n
                        used[n] = 1
                        backtrack(arr, idx + 1, used)
                        arr[idx] = 0
                        arr[dist] = 0
                        used[n] = 0

        backtrack(arr, 0, used)
        ans.sort(reverse=True)
        return list(ans[0])

    # 1049. Last Stone Weight II
    def lastStoneWeightII(self, stones: List[int]) -> int:
        counts = Counter(stones)
        arr = []
        for k, v in counts.items():
            if v % 2 == 1:
                arr.append(k)

        N = len(arr)
        self.ans = inf

        def backtrack(idx, arr):
            if idx >= N - 1:
                self.ans = min(sum(arr), self.ans)
                return
            if arr[idx] == 0:
                backtrack(idx + 1, arr)
                return
            val = arr[idx]
            arr[idx] = 0
            for i in range(idx + 1, len(arr)):
                if arr[i] != 0:
                    prev = arr[i]
                    nval = abs(val - arr[i])
                    arr[i] = nval
                    backtrack(idx + 1, arr)
                    arr[i] = prev
            arr[idx] = val

        backtrack(0, arr)
        return self.ans

    # 1980. Find Unique Binary String
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        nset = set(nums)
        self.ans = None
        N = len(nums)

        def backtrack(arr):
            if len(arr) == N:
                num = "".join(arr)
                if num not in nset:
                    self.ans = num
                return
            if self.ans:
                return

            arr.append("0")
            backtrack(arr)
            arr.pop()

            arr.append("1")
            backtrack(arr)
            arr.pop()

        backtrack([])
        return self.ans


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
