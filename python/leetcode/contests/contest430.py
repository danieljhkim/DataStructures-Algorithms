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
    def minimumOperations(self, grid: List[List[int]]) -> int:

        table = defaultdict(list)
        ROW = len(grid)
        COL = len(grid[0])
        for c in range(COL):
            for r in range(ROW):
                table[c].append(grid[r][c])
        count = 0
        for c in range(COL):
            vals = table[c]
            for i in range(len(vals) - 1):
                cur = vals[i]
                nxt = vals[i + 1]
                diff = cur - nxt
                if diff >= 0:
                    nxt += diff + 1
                    count += diff + 1
                vals[i + 1] = nxt
        return count

    def answerString(self, word: str, numFriends: int) -> str:
        if numFriends == 1:
            return word
        N = len(word)
        table = defaultdict(list)
        for i, w in enumerate(word):
            table[w].append(i)

        inorder = sorted(list(table.keys()), reverse=True)
        tops = table[inorder[0]]
        size = N - numFriends + 1
        candidates = []
        for i in tops:
            j = i
            limit = size
            while j < N and limit > 0:
                j += 1
                limit -= 1

            candidates.append(word[i:j])
        candidates.sort()
        return candidates[-1]

    def numberOfSubsequences(self, nums: List[int]) -> int:
        """ "
        p, q, r, s
        x     x
        nums[p] * nums[r] == nums[q] * nums[s]
        - at least 1 element inbtw

        """
        self.ans = 0
        N = len(nums)

        def backtrack(idx, pr, qs):
            if len(pr) == 2 and len(qs) == 2:
                lsome = 1
                for i in pr:
                    lsome *= nums[i]
                rsome = 1
                for i in qs:
                    rsome *= nums[i]
                if lsome == rsome:
                    self.ans += 1
                return
            if idx >= N:
                return
            if len(pr) == 0:
                for i in range(idx, N):
                    pr.append(i)
                    backtrack(i + 1, pr, qs)

    def numberOfSubsequences(self, nums: List[int]) -> int:
        N = len(nums)
        left = []
        ans = 0
        for p in range(N - 6):
            for q in range(p + 2, N - 4):
                left.append((p, q))
        for p, q in left:
            qv = nums[q]
            pv = nums[p]
            if qv == 0 or pv == 0:
                # q = 0 (s any) // p not zeor, so r == 0
                if qv == 0 and pv != 0:
                    for r in range(q + 2, N - 2):
                        if nums[r] == 0:
                            for s in range(r + 2, N):
                                ans += 1
                elif qv != 0 and pv == 0:
                    for r in range(q + 2, N - 2):
                        for s in range(r + 2, N):
                            if nums[s] == 0:
                                ans += 1
                elif pv == 0 and qv == 0:
                    for r in range(q + 2, N - 2):
                        if nums[r] == 0:
                            for s in range(r + 2, N):
                                ans += 1
                        else:
                            for s in range(r + 2, N):
                                if nums[s] == 0:
                                    ans += 1

            elif pv == qv:
                for r in range(q + 2, N - 2):
                    for s in range(r + 2, N):
                        if nums[s] == nums[r]:
                            ans += 1
            else:
                for r in range(q + 2, N - 2):
                    if nums[r] == qv:
                        for s in range(r + 2, N):
                            if nums[s] == pv:
                                ans += 1

        return ans


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
