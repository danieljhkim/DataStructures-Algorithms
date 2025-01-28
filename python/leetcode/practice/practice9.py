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

    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        narr = sorted(nums, reverse=True)
        group = {narr[0]: 0}
        cur_group = 0
        segments = defaultdict(list)
        segments[0].append(narr[0])

        for i in range(1, len(narr)):
            n = narr[i]
            if narr[i - 1] - n > limit:
                cur_group += 1
            group[n] = cur_group
            segments[cur_group].append(n)

        for i, n in enumerate(nums):
            c = group[n]
            nums[i] = segments[c].pop()
        return nums

    # 546. Remove Boxes
    def removeBoxes(self, boxes: List[int]) -> int:
        N = len(boxes)

        @lru_cache(None)
        def dp(left, right, count):
            if left > right:
                return 0
            idx = left
            while idx + 1 < right and boxes[idx + 1] == boxes[idx]:
                idx += 1
                count += 1
            res = dp(idx + 1, right, 0) + (count + 1) ** 2

            for i in range(idx + 1, right + 1):
                if boxes[i] == boxes[left]:
                    res = max(res, dp(idx + 1, i - 1, 0) + dp(i, right, count + 1))
            return res

        return dp(0, N - 1, 0)

    def longestSubarray(self, nums: List[int], limit: int) -> int:
        min_dq = deque()
        max_dq = deque()
        ans = 0
        left = 0
        for right, n in enumerate(nums):
            while min_dq and min_dq[-1] > n:
                min_dq.pop()
            while max_dq and max_dq[-1] < n:
                max_dq.pop()
            min_dq.append(n)
            max_dq.append(n)

            while max_dq[0] - min_dq[0] > limit:
                if max_dq[0] == nums[left]:
                    max_dq.popleft()
                if min_dq[0] == nums[left]:
                    min_dq.popleft()
                left += 1
            ans = max(right - left + 1, ans)
        return ans

    def sumRemoteness(self, grid: List[List[int]]) -> int:
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(grid)
        COL = len(grid[0])
        total = sum(val for row in grid for val in row if val != -1)
        ans = 0

        def dfs(r, c):
            point = grid[r][c]
            grid[r][c] = -1
            cnt = 1
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL:
                    val = grid[nr][nc]
                    if val > 0:
                        nval, ncnt = dfs(nr, nc)
                        cnt += ncnt
                        point += nval
            return point, cnt

        for r in range(ROW):
            for c in range(COL):
                val = grid[r][c]
                if val > 0:
                    points, cnt = dfs(r, c)
                    ans += (total - points) * cnt
        return ans

    def candy(self, ratings: List[int]) -> int:
        N = len(ratings)
        left = [1] * N
        ans = 0
        for i in range(1, N):
            if ratings[i - 1] < ratings[i]:
                left[i] = left[i - 1] + 1
        for i in range(N - 1, -1, -1):
            if ratings[i + 1] < ratings[i]:
                left[i] = max(left[i + 1] + 1, left[i])
                ans += left[i]
        return ans

    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        N = len(startGene)
        queue = deque([(0, startGene)])
        visited = set([startGene])
        choices = ["A", "C", "G", "T"]
        bank = set(bank)
        while queue:
            step, cur = queue.popleft()
            if cur == endGene:
                return step
            for nc in choices:
                for i in range(N):
                    new_gene = cur[:i] + nc + cur[i + 1 :]
                    if new_gene not in visited and new_gene in bank:
                        visited.add(new_gene)
                        queue.append((step + 1, new_gene))
        return -1

    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        if endGene not in bank:
            return -1

        choices = ["A", "C", "G", "T"]
        adj = defaultdict(list)
        visited = set([startGene])
        bank.append(startGene)
        bank = set(bank)

        for g in bank:
            for nc in choices:
                for i in range(8):
                    ng = g[:i] + nc + g[i + 1 :]
                    if ng in bank:
                        adj[g].append(ng)
        if not adj[endGene]:
            return -1

        queue = deque([(0, startGene)])
        while queue:
            step, cur = queue.popleft()
            if cur == endGene:
                return step
            for nei in adj[cur]:
                if nei not in visited:
                    visited.add(nei)
                    queue.append((step + 1, nei))
        return -1

    def minDistance(self, word1: str, word2: str) -> int:
        memo = {}
        N, N2 = len(word1), len(word2)

        def dp(i, word):
            if (i, word) in memo:
                return memo[(i, word)]
            if word == word2:
                return 0
            if i >= N2:
                cost = abs(len(word) - len(word2))
                memo[(i, word)] = cost
                return cost
            res = inf
            if len(word) == i:
                nw = word + word2[i]
                res = min(dp(i + 1, nw) + 1, res)
            elif word[i] != word2[i]:
                nw = word[:i] + word2[i] + word[i + 1 :]
                res = min(res, dp(i + 1, nw) + 1)

                nw = word[:i] + word2[i] + word[i:]
                res = min(dp(i + 1, nw) + 1, res)

                nw = word[:i] + word[i + 1 :]
                res = min(dp(i, nw) + 1, res)
            else:
                res = min(dp(i + 1, word), res)

            memo[(i, word)] = res
            return res

        return dp(0, word1)

    def minDistance(self, word1: str, word2: str) -> int:
        memo = {}

        def dp(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            if i == len(word1):
                return len(word2) - j
            if j == len(word2):
                return len(word1) - i
            if word1[i] == word2[j]:
                memo[(i, j)] = dp(i + 1, j + 1)
            else:
                insert_op = dp(i, j + 1)
                delete_op = dp(i + 1, j)
                replace_op = dp(i + 1, j + 1)
                memo[(i, j)] = 1 + min(insert_op, delete_op, replace_op)
            return memo[(i, j)]

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        memo = {}
        R = len(triangle)

        def dp(r, c):
            if R == r or c >= len(triangle[r]):
                return 0
            if (r, c) in memo:
                return memo[(r, c)]
            res = min(dp(r + 1, c), dp(r + 1, c + 1))
            memo[(r, c)] = res + triangle[r][c]
            return res + triangle[r][c]

        return dp(0, 0)

    def maximalSquare(self, matrix: List[List[str]]) -> int:
        R, C = len(matrix), len(matrix[0])

        directions = [(0, 1), (1, 0), (1, 1)]
        memo = {}

        def dp(r, c):
            if (r, c) in memo:
                return memo[(r, c)]
            if not (0 <= r < R and 0 <= c < C) or matrix[r][c] == "0":
                return 0
            val = float("inf")
            for dr, dc in directions:
                val = min(val, dp(r + dr, c + dc))
            memo[(r, c)] = 1 + val
            return memo[(r, c)]

        max_side = 0
        for r in range(R):
            for c in range(C):
                if matrix[r][c] == "1":
                    max_side = max(max_side, dp(r, c))
        return max_side * max_side

    def checkIfPrerequisite(
        self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]
    ) -> List[bool]:
        adj = defaultdict(list)
        ans = []
        for s, e in prerequisites:
            adj[e].append(s)

        memo = {}

        def dfs(course, preq, visited):
            if (course, preq) in memo:
                return memo[(course, preq)]
            res = False
            for nei in adj[course]:
                if nei in visited:
                    continue
                visited.add(nei)
                if nei == preq:
                    res = True
                    break
                else:
                    out = dfs(nei, preq, visited)
                    if out:
                        res = True
                        break
            memo[(course, preq)] = res
            return res

        for s, e in queries:
            res = dfs(e, s, set([e]))
            ans.append(res)
        return ans

    # floyd warshall
    def checkIfPrerequisite(
        self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]
    ) -> List[bool]:
        table = defaultdict(bool)

        for p, t in prerequisites:
            table[(p, t)] = True

        for btw in range(numCourses):
            for src in range(numCourses):
                for target in range(numCourses):
                    if not table[(src, target)]:
                        if table[(src, btw)] and table[(btw, target)]:
                            table[(src, target)] = True

        answer = []
        for p, t in queries:
            answer.append(table[(p, t)])

        return answer

    def findMaxFish(self, grid: List[List[int]]) -> int:
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(grid)
        COL = len(grid[0])

        def dfs(r, c):
            fishes = grid[r][c]
            grid[r][c] = 0
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL:
                    if grid[nr][nc] > 0:
                        fishes += dfs(nr, nc)
            return fishes

        ans = 0
        for r in range(ROW):
            for c in range(COL):
                if grid[r][c] > 0:
                    ans = max(dfs(r, c), ans)
        return ans

    def numTrees(self, n: int) -> int:
        memo = {}

        def dp(start, end):
            if start >= end:
                return 1
            if (start, end) in memo:
                return memo[(start, end)]
            total = 0
            for i in range(start, end + 1):
                left = dp(start, i - 1)
                right = dp(i + 1, end)
                total += left * right
            memo[(start, end)] = total
            return total

        return dp(1, n)

    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        N1 = len(s1)
        N2 = len(s2)
        N3 = len(s3)
        if N1 + N2 != N3:
            return False
        memo = {}

        def dp(i1, i2, i3):
            if i3 == N3:
                return True
            if (i1, i2, i3) in memo:
                return memo[(i1, i2, i3)]
            res = False
            if i1 < N1:
                if s1[i1] == s3[i3]:
                    res = dp(i1 + 1, i2, i3 + 1)
            if i2 < N2:
                if s2[i2] == s3[i3]:
                    res = res or dp(i1, i2 + 1, i3 + 1)
            memo[(i1, i2, i3)] = res
            return res

        return dp(0, 0, 0)

    def maxMoves(self, grid: List[List[int]]) -> int:
        directions = [(0, 1), (1, 1), (-1, 1)]
        ROW = len(grid)
        COL = len(grid[0])
        memo = {}

        def dp(r, c):
            if (r, c) in memo:
                return memo[(r, c)]
            cur = grid[r][c]
            res = 0
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL:
                    nval = grid[nr][nc]
                    if nval > cur:
                        res = max(res, dp(nr, nc))
            memo[(r, c)] = res + 1
            return res + 1

        res = 0
        for r in range(ROW):
            res = max(dp(r, 0), res)

        return res - 1

    def minSteps(self, n: int) -> int:
        if n <= 1:
            return 0
        memo = {}

        def dp(count, cur):
            if count == n:
                return 0
            if count > n:
                return inf
            if (count, cur) in memo:
                return memo[(count, cur)]
            res = dp(count + cur, cur) + 1
            if count * 2 <= n:
                res = min(dp(count * 2, count) + 2, res)
            memo[(count, cur)] = res
            return res

        return dp(1, 1) + 1


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
