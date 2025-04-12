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


class Solution:

    def minCosts(self, cost: List[int]) -> List[int]:
        N = len(cost)
        res = cost.copy()
        arr = []
        for i, n in enumerate(cost):
            arr.append((n, i))
        arr.sort()
        i = 0
        while i < N:
            val, idx = arr[i]
            res[idx] = min(val, res[idx])
            for j in range(i + 1, N):
                v2, i2 = arr[j]
                if i2 > idx:
                    res[i2] = min(res[i2], val)
            i += 1
        return res

    def longestPalindrome(self, s: str, t: str) -> int:  # absolute FAIL

        @cache
        def palindrome(word, left, right):
            while left >= 0 and right < len(word):
                if word[left] != word[right]:
                    break
                left -= 1
                right += 1
            return word[left + 1 : right], left, right

        def find(who, other: str):
            res = 1
            rev = other[::-1]
            for right in range(len(who)):
                if right > 0 and who[right] == who[right - 1]:
                    w1, idx, ridx = palindrome(who, right - 1, right)
                    if w1 in other:
                        res = max(len(w1) * 2, res)
                    else:
                        res = max(res, len(w1))
                        if who == s:
                            iii = idx
                            while idx >= 0 and who[idx : iii + 1] in rev:
                                idx -= 1
                                res = max(res, len(w1) + iii - idx + 1)
                        else:
                            iii = ridx
                            while ridx < len(other) and who[iii : ridx + 1] in rev:
                                ridx += 1
                                res = max(res, len(w1) + ridx - iii + 1)
                else:
                    w1, idx, ridx = palindrome(who, right, right)
                    if w1 in other:
                        res = max(len(w1) * 2, res)
                    else:
                        res = max(res, len(w1))
                        iii = idx
                        if who == s:
                            while idx >= 0 and who[idx : iii + 1] in rev:
                                idx -= 1
                                res = max(res, len(w1) + iii - idx + 1)
                        else:
                            iii = ridx
                            while ridx < len(other) and who[iii : ridx + 1] in rev:
                                ridx += 1
                                res = max(res, len(w1) + ridx - iii + 1)
                if who == s:
                    for left in range(0, right + 1):
                        w3 = who[left : right + 1]
                        if w3 in rev:
                            res = max(res, len(w3) * 2)
                        elif w3[:-1] in rev:
                            res = max(res, (len(w3) - 1) * 2 + 1)
                        if w3 in other:
                            ii = other.rfind(w3)
                            idx = ii - 1
                            if idx < 0:
                                res = max(len(w3) * 2, res)
                            elif idx > 0:
                                while idx > 0 and other[idx] == other[idx - 1]:
                                    idx -= 1
                                res = max(len(w3) * 2 + ii - idx, res)
                            else:
                                res = max(len(w3) * 2 + 1, res)

            return res

        return max(find(s, t), find(t, s))

    def longestPalindrome(self, s: str, t: str) -> int:
        res = 1
        for i in range(len(s)):
            for j in range(i, len(s) + 1):
                for l in range(len(t)):
                    for r in range(l, len(t) + 1):
                        if s[i:j] + t[l:r] == (s[i:j] + t[l:r])[::-1]:
                            res = max(res, j - i + r - l)
        return res


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
