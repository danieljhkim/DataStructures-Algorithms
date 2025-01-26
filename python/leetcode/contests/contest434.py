import heapq
import random
import math
import bisect
from typing import *
from math import inf, factorial
from functools import lru_cache, cache
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

    def countPartitions(self, nums: List[int]) -> int:
        ans = 0
        prefix = [nums[0]]
        for n in nums[1:]:
            prefix.append(prefix[-1] + n)

        if len(prefix) == 2:
            if abs(nums[0] - nums[1]) % 2 == 0:
                return 1
            else:
                return 0

        for i in range(len(prefix) - 1):
            left = prefix[i]
            right = prefix[-1] - prefix[i]
            if abs(left - right) % 2 == 0:
                ans += 1
        return ans

    def countMentions(self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        N = numberOfUsers
        mentions = [0] * N
        message = []
        offline = []

        for n in events:
            if n[0] == "MESSAGE":
                entry = (int(n[1]), n[2])
                message.append(entry)
            else:
                entry = (int(n[1]), n[2])
                offline.append(entry)

        message.sort()
        message = deque(message)
        offline.sort()
        totals = defaultdict(int)
        users = defaultdict(deque)
        for t, id in offline:
            users[id].append((t, t + 60 - 1))

        while message:
            mtime, ids = message.popleft()
            if ids == "ALL":
                for i in range(N):
                    totals[str(i)] += 1

            elif ids == "HERE":
                cand = set([str(i) for i in range(N)])
                for k, v in users.items():
                    while v and mtime > v[0][1]:
                        v.popleft()
                    if v and v[0][0] <= mtime <= v[0][1]:
                        if k in cand:
                            cand.remove(k)
                for c in cand:
                    totals[c] += 1
            else:
                splits = ids.split(" ")
                for s in splits:
                    key = s[2:]
                    totals[key] += 1

        for i in range(N):
            key = str(i)
            mentions[i] = totals[key]
        return mentions

    def maxFrequency(self, nums: List[int], k: int) -> int:

        N = len(nums)
        ans = 0
        freq_table = defaultdict(list)
        for i, n in enumerate(nums):
            freq_table[n].append(i)

        ans = 0
        for i in range(N):
            cur = nums[i]
            cnt = len(freq_table[cur]) - bisect.bisect_right(freq_table[cur], i) + 1
            bkcnt = bisect.bisect_right(freq_table[k], i) + 1
            kcnt = len(freq_table[k]) - bkcnt
            ans = max(ans, cnt - kcnt + bkcnt)

        return ans


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
