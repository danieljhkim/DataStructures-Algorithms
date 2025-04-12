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

    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        N = len(s)
        if N == 1 and k == 1:
            return True
        for i in range(len(s) - k):
            wo = s[i : i + k]
            w = s[i]
            if wo.count(w) == len(wo):
                if i > 0:
                    prev = s[i - 1]
                    if prev == w:
                        continue
                if i + k <= N - 1:
                    nxt = s[i + k]
                    if nxt == w:
                        continue
                return True
        return False

    def maxWeight(self, pizzas: List[int]) -> int:

        N = len(pizzas)
        D = N // 4
        evens = 0
        odds = 0
        for i in range(1, D + 1):
            if i % 2 == 0:
                evens += 1
            else:
                odds += 1

        ans = 0
        pizzas.sort()
        min_dq = deque(pizzas[: N // 2])
        max_dq = deque(pizzas[N // 2 :])

        while odds > 0:
            i = 0
            while min_dq and i < 3:
                min_dq.popleft()
                i += 1
            if max_dq:
                ans += max_dq.pop()
            else:
                ans += min_dq.pop()
            odds -= 1

        while evens > 0:
            i = 0
            while i < 2:
                if min_dq:
                    min_dq.popleft()
                else:
                    max_dq.popleft()
                i += 1
            print(evens, max_dq, min_dq)
            max_dq.pop()
            ans += max_dq.pop()
            evens -= 1
        return ans

    def maxSubstringLength(self, s: str, k: int) -> bool:

        count = Counter(s)

        N = len(s)
        i = 0
        seen = set()
        wset = set()
        good = set()
        while i < N:
            cur = s[i]
            if cur in seen:
                i += 1
            cnt = 0
            idx = i
            while idx < N and count[cur] > cnt:
                if s[idx] == cur:
                    cnt += 1
                idx += 1
            ww = s[i:idx]
            if ww not in wset:
                wset.add(ww)
                good.add(ww)
            else:
                if ww in good:
                    good.remove(ww)
            i = idx
            seen.add(cur)

        if len(good) >= k:
            return True
        return False


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
