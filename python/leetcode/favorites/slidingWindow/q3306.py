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

    # 3306. Count of Substrings Containing Every Vowel and K Consonants II
    def countOfSubstrings(self, word: str, k: int) -> int:
        N = len(word)
        right = ans = vcnt = ccnt = left = 0
        counts = {"a": 0, "e": 0, "i": 0, "o": 0, "u": 0}
        while right < N:
            w = word[right]
            if w not in counts:
                ccnt += 1
            else:
                counts[w] += 1
                if counts[w] == 1:
                    vcnt += 1

            if ccnt == k and vcnt == 5:
                right_cnt = 1
                finished = True
                tmp = right
                while left < right and word[left] in counts:
                    if counts[word[left]] == 1:
                        finished = False
                        break
                    counts[word[left]] -= 1
                    left += 1
                    right_cnt += 1
                if right == N - 1 or word[right + 1] not in counts:
                    ans += right_cnt
                    right += 1
                    ccnt += 1
                    finished = True
                    add = False
                    if word[left] not in counts:
                        ccnt -= 1
                        left += 1
                        add = True
                    while left < right and word[left] in counts:
                        if counts[word[left]] == 1:
                            vcnt -= 1
                        counts[word[left]] -= 1
                        left += 1
                        if add and vcnt == 5:
                            ans += 1
                    left += 1
                    ccnt -= 1
                else:
                    while right < N:
                        ans += right_cnt
                        right += 1
                        if right == N or word[right] not in counts:
                            right -= 1
                            break
                        if finished:
                            counts[word[right]] += 1
                if not finished:
                    counts[word[left]] -= 1
                    vcnt -= 1
                    left += 1
                    right = tmp
            elif ccnt > k:
                while left < right and word[left] in counts:
                    if counts[word[left]] == 1:
                        vcnt -= 1
                    counts[word[left]] -= 1
                    left += 1
                left += 1
                ccnt -= 1
            right += 1

        while vcnt == 5 and k <= ccnt:
            if word[left] not in counts:
                ccnt -= 1
            else:
                counts[word[left]] -= 1
                if counts[word[left]] == 0:
                    break
            if ccnt == k:
                ans += 1
            left += 1
        return ans
