import heapq
from typing import Optional, List
from itertools import accumulate
import random
from collections import Counter, deque, defaultdict, OrderedDict
import math
import bisect
from math import inf
from functools import lru_cache
from sortedcontainers import SortedSet


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
    def minimumOperations(self, nums: List[int]) -> int:
        def check(arr):
            if len(set(arr)) != len(arr):
                return False
            return True

        numsq = deque(nums)
        ans = 0
        while not check(numsq):
            ans += 1
            k = 3
            while numsq and k > 0:
                k -= 1
                numsq.popleft()
        return ans

    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        intervals = [(n - k, n + k) for n in nums]
        intervals.sort()
        count = 0
        used = set()
        small = intervals[0][0]
        for i in range(len(intervals)):
            start, end = intervals[i]
            start = max(small, start)
            while start < end and start in used:
                start += 1
            if start not in used:
                count += 1
            small = start
            used.add(small)
        return count

    def minLength(self, s: str, numOps: int) -> int:
        left = 0
        right = 0
        big = 0
        start = 0
        end = 0
        while right < len(s) and left < len(s):
            while right < len(s) and s[left] == s[right]:
                right += 1
            if right - left > big:
                big = right - left
                start = left
                end = right - 1
            left = right
        c = s[end]
        S = start
        E = end + 1
        while end < len(s) - 1 and s[end + 1] != c:
            end += 1
        c = s[start]
        while start > 0 and s[start - 1] != c:
            start -= 1
        word = s[start : end + 1]
        N = len(word)

        lcache = {}

        def longest(st):
            if st in lcache:
                return lcache[st]
            ans = 1
            left = 0
            right = 0
            while right < E and left < E:
                while right < E and st[left] == st[right]:
                    right += 1
                ans = max(right - left, ans)
                left = right
            lcache[st] = ans
            return ans

        cache = {}
        ans = {}

        def recurs(word, count, i):
            if (word, count, i) in cache:
                return cache[(word, count, i)]
            if count == 0 or i == E:
                return longest(word)
            ans = recurs(word, count, i + 1)
            if word[i] == "0":
                word2 = word[:i] + "1" + word[i + 1 :]
            else:
                word2 = word[:i] + "0" + word[i + 1 :]
            other = recurs(word2, count - 1, i + 1)

            res = min(ans, other)
            cache[(word, count, i)] = res
            return res

        res = recurs(word, numOps, 0)
        return res

    def minLength(self, s: str, numOps: int) -> int:
        left = 0
        right = 0
        big = 0
        start = 0
        end = 0
        while right < len(s) and left < len(s):
            while right < len(s) and s[left] == s[right]:
                right += 1
            if right - left > big:
                big = right - left
                start = left
                end = right - 1
            left = right

        word = list(s[start : end + 1])
        N = len(word)

        def longest(st):
            ans = 1
            left = 0
            right = 0
            while right < N and left < N:
                while right < N and st[left] == st[right]:
                    right += 1
                ans = max(right - left, ans)
                left = right
            return ans

        def recurs(word, count, i):
            if count == 0 or i == N:
                return longest(word)
            ans = recurs(word, count, i + 1)
            if word[i] == "0":
                word[i] = "1"
                prev = "0"
            else:
                word[i] = "0"
                prev = "1"
            other = recurs(word, count - 1, i + 1)
            word[i] = prev
            return min(ans, other)

        return min(big, recurs(word, numOps, 0))

    def minLength(self, s: str, numOps: int) -> int:

        word = s
        N = len(word)
        lcache = {}

        def longest(st):
            if st in lcache:
                return lcache[st]
            ans = 1
            left = 0
            right = 0
            while right < N and left < N:
                while right < N and st[left] == st[right]:
                    right += 1
                ans = max(right - left, ans)
                left = right
            lcache[st] = ans
            return ans

        cache = {}

        def recurs(word, count, i):
            if (word, count) in cache:
                return cache[(word, count)]
            if count == 0 or i == N:
                return longest(word)
            ans = recurs(word, count, i + 1)
            if word[i] == "0":
                word2 = word[:i] + "1" + word[i + 1 :]
            else:
                word2 = word[:i] + "0" + word[i + 1 :]
            other = recurs(word2, count - 1, i + 1)
            res = min(ans, other)
            cache[(word, count)] = res
            return res

        return recurs(word, numOps, 0)


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
