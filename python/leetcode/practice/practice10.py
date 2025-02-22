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
        N = len(stones)

        @cache
        def dp(i, current):
            if i == N:
                return abs(current)
            add = dp(i + 1, current + stones[i])
            sub = dp(i + 1, current - stones[i])
            return min(add, sub)

        return dp(0, 0)

    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        best = -1
        nums.sort()
        for i, n in enumerate(nums):
            t = k - n
            if t < 0:
                break
            idx = bisect.bisect_left(nums, t, lo=i) - 1
            total = n + nums[idx]
            if total < k:
                best = max(best, total)
        return best

    def find132pattern(self, nums: List[int]) -> bool:
        """ "
        i j k
        i k j
        """
        third = float("-inf")
        stack = []
        for num in reversed(nums):
            if num < third:
                return True
            while stack and num > stack[-1]:
                third = stack.pop()
            stack.append(num)
        return False

    def maxSubstringLength(self, s: str, k: int) -> bool:
        if k > len(s):
            return False
        if k < 1:
            return True
        N = len(s)
        bad = [0] * 26
        for w in s:
            bad[ord(w) - ord("a")] += 1

        good = 0
        for n in bad:
            if n == 1:
                good += 1
        if good >= k:
            return True

        @cache
        def dfs(idx, options, cnt):
            if cnt == 0:
                return True
            if idx >= N:
                return False
            ops = list(options)
            leftover = [0] * 26
            right = idx
            res = False
            while right < N:
                cur = s[right]
                alp = ord(cur) - ord("a")
                ops[alp] -= 1
                leftover[alp] = ops[alp]
                if sum(leftover) == 0 and right - idx != N - 1:
                    return dfs(right + 1, tuple(ops), cnt - 1)
                elif leftover[alp] == 0 and right - idx != N - 1:
                    j = right
                    tmp = [0] * 26
                    oo = list(options)
                    while j >= idx:
                        al = ord(s[j]) - ord("a")
                        oo[al] -= 1
                        tmp[al] = oo[al]
                        if sum(tmp) == 0:
                            break
                        j -= 1
                    if sum(tmp) == 0:
                        return dfs(right + 1, tuple(oo), cnt - 1)
                right += 1
            return res

        return dfs(0, tuple(bad), k)

    def minSwaps(self, data: List[int]) -> int:
        total = sum(data)
        if total == 1:
            return 0
        left = 0
        right = total
        cnt = sum(data[left:right])
        ans = total - cnt

        N = len(data)
        while right < N:
            cur = data[right]
            prev = data[left]
            cnt += cur - prev
            ans = min(total - cnt, ans)
            right += 1
            left += 1
        return ans

    # 1861. Rotating the Box
    def rotateTheBox(self, boxGrid: List[List[str]]) -> List[List[str]]:
        R = len(boxGrid)
        C = len(boxGrid[0])
        grid = [["."] * R for _ in range(C)]

        for r in range(R):
            rocks = 0
            for c in range(C):
                val = boxGrid[r][c]
                if val == "#":
                    rocks += 1
                elif val == "*":
                    grid[c][r] = "*"
                    idx = c
                    while rocks > 0:
                        idx -= 1
                        rocks -= 1
                        grid[idx][r] = "#"
            idx = C
            while rocks > 0:
                idx -= 1
                rocks -= 1
                grid[idx][r] = "#"

        for row in grid:
            row.reverse()
        return grid

    # 1197. Minimum Knight Moves
    def minKnightMoves(self, x: int, y: int) -> int:
        moves = ((2, 1), (1, 2), (-2, 1), (-1, 2), (-2, -1), (-1, -2), (2, -1), (1, -2))
        heap = [(0, 0, 0)]
        distances = defaultdict(lambda: inf)
        while heap:
            dist, r, c = heapq.heappop(heap)
            if r == x and c == y:
                return dist
            for dr, dc in moves:
                nr = dr + r
                nc = dc + c
                if distances[(nr, nc)] > dist + 1:
                    distances[(nr, nc)] = dist + 1
                    heapq.heappush(heap, (dist + 1, nr, nc))
        return -1

    # 1028. Recover a Tree From Preorder Traversal
    def recoverFromPreorder(self, nodes: str) -> Optional[TreeNode]:
        stack = []
        idx = 0
        root = None
        N = len(nodes)

        while idx < N:
            cnt = 0
            while idx < N and nodes[idx] == "-":
                cnt += 1
                idx += 1
            while stack and stack[-1][1] >= cnt:
                stack.pop()
            cur = ""
            while idx < N and nodes[idx].isdigit():
                cur += nodes[idx]
                idx += 1
            nnode = TreeNode(int(cur))
            if not root:
                root = nnode
            if stack:
                if not stack[-1][0].left:
                    stack[-1][0].left = nnode
                else:
                    stack[-1][0].right = nnode
                    stack.pop()
            stack.append((nnode, cnt))
        return root

    # 487. Max Consecutive Ones II
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        stack = []
        max_len = cur = 0
        for n in nums:
            if n == 0:
                prev = 0
                if stack:
                    prev = stack.pop() + 1
                max_len = max(prev + cur, max_len)
                stack.clear()
                stack.append(cur)
                cur = 0
            else:
                cur += 1
        prev = 0
        if stack:
            prev = stack.pop() + 1
        max_len = max(prev + cur, max_len)
        return max_len


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
