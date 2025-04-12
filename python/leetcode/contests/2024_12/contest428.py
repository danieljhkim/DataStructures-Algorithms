import heapq
from typing import Optional, List
from itertools import accumulate
import random
from collections import Counter, deque, defaultdict, OrderedDict
import math
import bisect
from math import inf
from functools import lru_cache


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

    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        ans = defaultdict(list)
        mx = 0
        for i in range(len(events) - 1):
            cur = events[i]
            nxt = events[i + 1]
            idx = nxt[0]
            diff = nxt[1] - cur[1]
            mx = max(diff, mx)
            ans[diff].append(idx)
        idx, start = events[0]
        mx = max(start, mx)
        ans[start].append(idx)
        arr = ans[mx]
        arr.sort()
        return arr[0]

    def maxAmount(
        self,
        initialCurrency: str,
        pairs1: List[List[str]],
        rates1: List[float],
        pairs2: List[List[str]],
        rates2: List[float],
    ) -> float:
        """ "
        day1
            pairs1[i] = [startCurreny, targetCurreny] -> can convert at rate1[i] -> start * rate -> targetcur
        day2
            pairs2[i] = [startCurreny, targetCurreny] -> can convert at rate12i]

        targetCurrency -convert-> startCur => 1/rate
        """

        start = initialCurrency
        day1 = []
        day2 = []
        adj1 = defaultdict(list)
        adj2 = defaultdict(list)

        for i, v in enumerate(pairs1):
            rate = rates1[i]
            cur = v[0]
            dst = v[1]
            adj1[cur].append((rate, dst))
            adj1[dst].append((1 / rate, cur))

        for i, v in enumerate(pairs2):
            rate = rates2[i]
            cur = v[0]
            dst = v[1]
            adj2[cur].append((rate, dst))
            adj2[dst].append((1 / rate, cur))

        visited = {}

        def dfs1(cur, total):
            if cur in visited:
                if visited[cur] >= total:
                    return
            day1.append((total, cur))
            visited[cur] = total
            for rate, nei in adj1[cur]:
                dfs1(nei, rate * total)

        dfs1(start, 1)

        ans = []
        for dist, start in day1:
            heap = [(-dist, start)]
            distances = {}
            while heap:
                dist, cur = heapq.heappop(heap)
                if cur == initialCurrency:
                    ans.append(-dist)
                    break
                for rate, nei in adj2[cur]:
                    nd = dist * rate
                    if nei in distances:
                        if distances[nei] > nd:
                            distances[nei] = nd
                        else:
                            continue
                    else:
                        distances[nei] = nd
                    heapq.heappush(heap, (nd, nei))
        ans.sort()
        return ans[-1]

    def beautifulSplits(self, nums: List[int]) -> int:
        """ "
        num1 prefix of num2
        num2 prefix of num3
        """
        N = len(nums)
        ans = 0
        idx = 0
        while idx <= N - 1:
            start = idx
            while idx < N - 1 and nums[idx] == nums[idx + 1]:
                idx += 1
                ans += 1
            idx = start + 1
        idx = 1
        while idx <= N - 1:
            start = idx
            while idx < N - 1 and nums[idx] == nums[idx + 1]:
                idx += 1
                ans += 1
            idx = start + 1
        return ans


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
