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

    def punishmentNumber(self, n: int) -> int:
        memo = {}

        def dp(snum, cur, idx):
            if idx == len(snum):
                if cur == 0:
                    return True
                return False
            if cur < 0:
                return False
            if (snum, cur, idx) in memo:
                return memo[(snum, cur, idx)]
            res = False
            for i in range(idx, len(snum)):
                n = int(snum[idx : i + 1])
                res = dp(snum, cur - n, i + 1) or res
                if res:
                    break
            memo[(snum, cur, idx)] = res
            return res

        ans = 0
        for i in range(1, n):
            num = str(i * i)
            res = dp(num, i, 0)
            if res:
                ans += i * i
        return ans

    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        covered = set([None])
        self.ans = 0

        def dfs(node, prev):
            if not node:
                return 0
            left = dfs(node.left, node)
            right = dfs(node.right, node)
            res = 0
            if prev == None and node not in covered:
                covered.add(node)
                res += 1
            elif node.left not in covered or node.right not in covered:
                covered.update({node.left, node.right, node, prev})
                res += 1
            return left + right + res

        return dfs(root, None)

    def earliestFullBloom(self, plantTime: List[int], growTime: List[int]) -> int:
        heap = []
        N = len(plantTime)
        for i in range(N):
            heap.append((-growTime[i], plantTime[i]))
        heapq.heapify(heap)
        g, p = heapq.heappop(heap)
        cur_time = p
        g_time = -g
        while heap:
            g, p = heapq.heappop(heap)
            cur_time += p
            g_time -= p
            g_time = max(-g, g_time)
        return g_time + cur_time

    def equalPairs(self, grid: List[List[int]]) -> int:
        rtable = defaultdict(int)
        ans = 0

        for row in grid:
            rtable[tuple(row)] += 1

        for col in zip(*grid):
            c = tuple(col)
            if c in rtable:
                ans += rtable[c]
        return ans

    def lengthOfLongestSubstring(self, s: str) -> int:
        counts = defaultdict(int)
        ans = left = right = 0
        N = len(s)
        while right < N:
            w = s[right]
            counts[w] += 1
            while counts[w] > 1:
                counts[s[left]] -= 1
                left += 1
            ans = max(right - left + 1, ans)
        return ans

    def connect(self, root: "Optional[Node]") -> "Optional[Node]":
        if not root:
            return None
        queue = deque([root])
        while queue:
            size = len(queue)
            prev = None
            for _ in range(size):
                cur = queue.popleft()
                cur.next = prev
                prev = cur
                if cur.right:
                    queue.append(cur.right)
                if cur.left:
                    queue.append(cur.left)
        return root

    def lastStoneWeight(self, stones: List[int]) -> int:
        heap = []
        for s in stones:
            heap.append(-s)
        heapq.heapify(heap)

        while len(heap) > 1:
            cur = heapq.heappop(heap)
            if heap[0] == cur:
                heapq.heappop(heap)
            else:
                heapq.heapreplace(heap, cur - heap[0])

        return -heap[0] if heap else 0

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

    def lastStoneWeightII(self, stones: List[int]) -> int:
        N = len(stones)

        @cache
        def dp(i, current):
            if i == N:
                return abs(current)
            add = dp(i + 1, current + stones[i])
            sub = dp(i + 1, current - stones[i])
            return min(add, sub)

        return dp(0, 0)

    def constructDistancedSequence(self, n: int) -> List[int]:
        ans, options = [], []
        for i in range(2, n + 1):
            options.append(i)
        N = (n - 1) * 2 + 1

        arr = [0] * N

        def backtrack(arr, idx, used):
            if idx == N:
                ans.append(tuple(arr))
                return
            if arr[idx] != 0:
                backtrack(arr, idx + 1, used)
                return
            if 1 not in used:
                arr[idx] = 1
                used.add(1)
                backtrack(arr, idx + 1, used)
                arr[idx] = 0
                used.remove(1)

            for i, n in enumerate(options):
                dist = idx + n
                if dist >= N:
                    break
                if arr[dist] != 0 and n not in used:
                    arr[idx] = n
                    arr[dist] = n
                    used.add(n)
                    backtrack(arr, idx + 1, used)
                    arr[idx] = 0
                    arr[dist] = 0
                    used.remove(n)

        backtrack(arr, 0, set())
        ans.sort(reversed=True)
        return list(ans[0])


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
