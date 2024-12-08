import heapq
from typing import Optional, List
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

    def hasPath(
        self, maze: List[List[int]], start: List[int], destination: List[int]
    ) -> bool:
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(maze)
        COL = len(maze[0])
        sr = start[0]
        sc = start[1]
        visited = set()
        dq = deque([(sr, sc)])
        while dq:
            r, c = dq.popleft()
            if (r, c) == (destination[0], destination[1]):
                return True
            for dr, dc in directions:
                nr, nc = r, c
                while (
                    0 <= nr + dr < ROW
                    and 0 <= nc + dc < COL
                    and maze[nr + dr][nc + dc] == 0
                ):
                    nr += dr
                    nc += dc
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    dq.append((nr, nc))

        return False

    def minOperations(self, nums: List[int], k: int) -> int:
        smallest = min(nums)
        if smallest < k:
            return -1
        heap = [-n for n in nums]
        heapq.heapify(heap)
        count = 0
        while heap[0] < -k:
            big = heap[0]
            sames = 0
            while heap and heap[0] == big:
                heapq.heappop(heap)
                sames += 1
            if heap:
                small = heap[0]
            else:
                small = -k
            count += 1
            for _ in range(sames):
                heapq.heappush(heap, small)
        return count

    def minOperations(self, n: int, m: int) -> int:
        def is_prime(n: int) -> bool:
            if n <= 1:
                return False
            if n <= 3:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True

        if is_prime(n) or is_prime(m):
            return -1
        narr = list(str(n))
        dest = list(str(m))
        adj = [str(i) for i in range(10)]
        start = narr
        nnn = int("".join(start))
        heap = [(nnn, start)]
        visited = set()
        while heap:
            ops, cur = heapq.heappop(heap)
            if cur == dest:
                return ops
            for i, n in enumerate(cur):
                n = int(n)
                if n > 0:
                    small = adj[n - 1]
                    new_n = cur[:]
                    new_n[i] = small
                    num = int("".join(new_n))
                    if num not in visited and (small != "0" and i != 0):
                        if not is_prime(num):
                            heapq.heappush(heap, (ops + num, new_n))
                        visited.add(num)
                if n < 9:
                    big = adj[n + 1]
                    new_n = cur[:]
                    new_n[i] = big
                    num = int("".join(new_n))
                    if num not in visited:
                        if not is_prime(num):
                            heapq.heappush(heap, (ops + num, new_n))
                        visited.add(num)
        return -1

    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        N = len(costs)
        target = N / 2
        memo = {}

        def recursion(i, one, two):
            if i > N or (one == 0 and two == 0):
                return 0
            if (i, one, two) in memo:
                return memo[(i, one, two)]
            cur1 = inf
            cur2 = inf
            if one > 0:
                cur1 = recursion(i + 1, one - 1, two) + costs[i][0]
            if two > 0:
                cur2 = recursion(i + 1, one, two - 1) + costs[i][1]
            res = min(cur1, cur2)
            memo[(i, one, two)] = res
            return res

        return recursion(0, target, target)

    def missingElement(self, nums: List[int], k: int) -> int:
        N = len(nums)
        small = nums[0]
        low = 0
        high = N - 1
        while low <= high:
            mid = (low + high) // 2
            diff = nums[mid] - (mid + small)
            if diff >= k:
                high = mid - 1
            else:
                low = mid + 1
        return small + k + low - 1

    def minTotalDistance(self, grid: List[List[int]]) -> int:
        def find_middle(nums):
            N = len(nums)
            mid = N // 2
            return nums[mid]

        ROW = len(grid)
        COL = len(grid[0])
        cols = []
        rows = []
        for r in range(ROW):
            for c in range(COL):
                if grid[r][c] == 1:
                    rows.append(r)
                    cols.append(c)
        cols.sort()
        row = find_middle(rows)
        col = find_middle(cols)

        ans = 0
        for r in rows:
            ans += abs(row - r)
        for c in cols:
            ans += abs(col - c)
        return ans

    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        table = defaultdict(deque)
        ROW = len(nums)
        col = 0
        for r in range(ROW):
            col = max(col, len(nums[r]))
            for c in range(len(nums[r])):

                total = r + c
                table[total].appendleft(nums[r][c])
        ans = []
        for i in range(ROW + col - 1):
            if table[i]:
                ans.extend(list(table[i]))
        return ans

    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        @lru_cache
        def is_good(word, word2):
            diff = 0
            for i in range(len(word)):
                if word[i] != word2[i]:
                    diff += 1
            if diff == 1:
                return True

        memo = {}

        def recursion(prev):
            if prev == endWord:
                return 0
            if prev in memo:
                return memo[prev]
            count = inf
            for word in wordList:
                if is_good(prev, word):
                    count = min(count, recursion(word))
            memo[prev] = count + 1
            return count + 1

        res = recursion(beginWord)
        if res == inf:
            return 0
        return res


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
