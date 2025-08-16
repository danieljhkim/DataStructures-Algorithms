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

    # 3634. Minimum Removals to Balance Array
    def minRemoval(self, nums: List[int], k: int) -> int:  # MLE
        nums.sort()

        @cache
        def dp(lidx, ridx):
            if lidx == ridx:
                return 0
            if nums[lidx] * k < nums[ridx]:
                res = min(dp(lidx + 1, ridx), dp(lidx, ridx - 1)) + 1
                return res
            return 0

        res = dp(0, len(nums) - 1)
        dp.cache_clear()
        return res

    # 1198. Find Smallest Common Element in All Rows
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        rtable = defaultdict(set)
        table = defaultdict(int)
        R, C = len(mat), len(mat[0])
        for c in range(C):
            for r in range(R):
                val = mat[r][c]
                if val not in rtable[r]:
                    rtable[r].add(val)
                    table[val] += 1
                    if table[val] == R:
                        return val
        return -1

    # 61. Rotate List
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if k == 0:
            return head
        dq = deque()
        cur = head

        while cur:
            dq.append(cur)
            cur = cur.next
        if len(dq) == 0:
            return head

        n = k % len(dq)
        for _ in range(n):
            dq.appendleft(dq.pop())

        dummy = ListNode(-1)
        cur = dummy
        while dq:
            cur.next = dq.popleft()
            cur = cur.next
        cur.next = None
        return dummy.next

    # 2942. Find Words Containing Character
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        arr = []
        for i, w in enumerate(words):
            if x in w:
                arr.append(i)
        return arr

    # 2943. Maximize Area of Square Hole in Grid
    def maximizeSquareHoleArea(
        self, n: int, m: int, hBars: List[int], vBars: List[int]
    ) -> int:
        hBars.sort()
        vBars.sort()

        hlen, mxh = 2, 2
        for i in range(1, len(hBars)):
            if hBars[i - 1] + 1 == hBars[i]:
                hlen += 1
                mxh = max(hlen, mxh)
            else:
                hlen = 2

        vlen, mxv = 2, 2
        for i in range(1, len(vBars)):
            if vBars[i - 1] + 1 == vBars[i]:
                vlen += 1
                mxv = max(vlen, mxv)
            else:
                vlen = 2

        res = min(mxv, mxh)
        return res * res

    # 2944. Minimum Number of Coins for Fruits
    def minimumCoins(self, prices: List[int]) -> int:
        N = len(prices)

        @cache
        def dp(idx):
            if idx >= N:
                return 0
            cost = prices[idx]
            free = idx + idx
            if free >= N:
                return cost
            res = dp(free + 1)
            for i in range(idx + 1, free + 1):
                res = min(res, dp(idx + i + 1))
            return res + cost

        return dp(0)

    # 869. Reordered Power of 2
    def reorderedPowerOf2(self, n: int) -> bool:
        if n == 1:
            return True
        all_nums = ["1", "2"]
        start, nmax, snum = 2, 10**9, str(n)
        cands, nums, N = [], Counter(snum), len(snum)

        while start <= nmax:
            start *= 2
            all_nums.append(str(start))

        for nn in all_nums:
            if len(nn) == N:
                cands.append(nn)

        for cand in cands:
            cnts = Counter(cand)
            found = True
            for k, v in cnts.items():
                if nums[k] != v:
                    found = False
                    break
            if found:
                return True
        return False

    def reverseSubmatrix(
        self, grid: List[List[int]], x: int, y: int, k: int
    ) -> List[List[int]]:
        flips = []
        for r in range(x, x + k):
            row = []
            for c in range(y, y + k):
                row.append(grid[r][c])
            flips.append(row)

        for r in range(x, x + k):
            row = flips.pop()
            idx = 0
            for c in range(y, y + k):
                grid[r][c] = row[idx]
                idx += 1
        return grid

    def maximum69Number(self, num: int) -> int:
        target, snum, d = 0, 0, 1
        while num > 0:
            rem = num % 10
            num //= 10
            if rem == 6:
                target = d
            snum += rem * d
            d *= 10
        snum += 3 * target
        return snum

    # 3648. Minimum Sensors to Cover Grid
    def minSensors(self, n: int, m: int, k: int) -> int:
        jump = k * 2 + 1
        r = n // jump
        c = m // jump
        if jump * r < n:
            r += 1
        if jump * c < m:
            c += 1
        return r * c

    # 3650. Minimum Cost Path with Edge Reversals
    def minCost(self, n: int, edges: List[List[int]]) -> int:
        adj, adj2 = defaultdict(list), defaultdict(list)
        distances = [inf] * n
        distances[0] = 0
        heap = [(0, 0)]

        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w * 2))

        while heap:
            dist, cur = heapq.heappop(heap)
            if cur == n - 1:
                return dist
            if dist > distances[cur]:
                continue
            for dest, w in adj[cur]:
                if distances[dest] > w + dist:
                    distances[dest] = w + dist
                    heapq.heappush(heap, (dist + w, dest))
            for dest, w in adj2[cur]:
                if distances[dest] > w + dist:
                    distances[dest] = w + dist
                    heapq.heappush(heap, (dist + w, dest))
        return -1


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
