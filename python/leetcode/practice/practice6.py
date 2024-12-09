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

    def myAtoi(self, s: str) -> int:
        s = deque(s.strip())
        if not s:
            return 0
        MAX = 2**31
        sign = "+"
        if s[0] == "-":
            sign = "-"
            s.popleft()
        elif s[0] == "+":
            s.popleft()

        while s and s[0] == "0":
            s.popleft()
        str_n = ""
        while s and s[0].isdigit():
            str_n += s.popleft()

        if not str_n:
            return 0
        num = int(str_n)
        if sign == "-":
            if num > MAX:
                return -MAX
            return -num
        if num > MAX - 1:
            return MAX - 1
        return num

    def myAtoi(self, s: str) -> int:
        MAX = (2**31) - 1
        MIN = -(2**31)
        N = len(s)
        if not s:
            return 0
        sign = 1
        i = 0
        while i < N and s[i] == " ":
            i += 1
        if i == N:
            return 0

        if s[i] == "-":
            sign = -1
            i += 1
        elif s[i] == "+":
            i += 1

        while i < N and s[i] == "0":
            i += 1
        num = 0
        while i < N and s[i].isdigit():
            num *= 10
            num += int(s[i])
            if num > MAX:
                if sign == -1:
                    return MIN
                return MAX
            i += 1
        if num > MAX:
            if sign == -1:
                return MIN
            return MAX
        return num * sign

    def findMinDifference(self, timePoints: List[str]) -> int:
        times = set()
        for t in timePoints:
            hour = int(t[:2])
            minu = int(t[2:])
            times.add(hour * 60 + minu)
        if len(times) != len(timePoints):
            return 0

        trange = 60 * 24
        buckets = [0] * trange
        for m in times:
            buckets[m] += 1

        left = 0
        right = 0
        ans = float("inf")
        diffs = len(times)
        while diffs > 0:
            while buckets[left % trange] == 0:
                left += 1
            right = left + 1
            while buckets[right % trange] == 0:
                right += 1
            if right != left:
                ans = min((right - left), ans)
                diffs -= 1
            else:
                break
            left += 1
            right += 1
        return ans

    def canSortArray(self, nums: List[int]) -> bool:
        bnums = []
        prev = bin(nums[0]).count("1")
        bnums.append([nums[0]])
        N = len(nums)
        for i, n in enumerate(nums, start=1):
            cur = bin(n).count("1")
            if cur == prev:
                bnums[-1].append(n)
            else:
                bnums.append([n])
                prev = cur
        for i in range(len(bnums) - 1):
            first = max(bnums[i])
            nxt = min(bnums[i + 1])
            if first > nxt:
                return False

        return True

    def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        self.left = []
        self.leaves = [[], []]
        self.right = []

        def left_bound(node, isbound):
            if not node.left and not node.right:
                self.leaves[0].append(node.val)
                return
            if isbound:
                self.left.append(node.val)
            if node.left:
                left_bound(node.left, isbound)
                isbound = False
            if node.right:
                left_bound(node.right, isbound)

        def right_bound(node, isbound):
            if not node.left and not node.right:
                self.leaves[1].append(node.val)
                return
            if isbound:
                self.right.append(node.val)
            if node.right:
                right_bound(node.right, isbound)
                isbound = False
            if node.left:
                right_bound(node.left, isbound)

        if root.left:
            left_bound(root.left, True)
        if root.right:
            right_bound(root.right, True)
        self.leaves[1].reverse()
        self.right.reverse()
        return [root.val] + self.left + self.leaves[0] + self.leaves[1] + self.right

    def isNumber(self, s: str) -> bool:
        seen_dot = False
        seen_exp = False
        prev = None
        seen_digit = False

        for i, n in enumerate(s):
            if n.isalpha():
                if (n == "E" or n == "e") and not seen_exp and seen_digit:
                    seen_exp = True
                else:
                    return False
            elif n == ".":
                if seen_dot or seen_exp:
                    return False
                seen_dot = True
            elif n == "+" or n == "-":
                if prev and not (prev == "E" or prev == "e"):
                    return False
            elif n.isdigit():
                seen_digit = True
            else:
                return False
            prev = n
        if (
            prev.lower() == "e"
            or (prev == "." and not seen_digit)
            or (prev == "+" or prev == "-")
        ):
            return False
        return True

    def isValidPalindrome(self, s: str, k: int) -> bool:
        memo = {}

        def dp(s):
            if s in memo:
                return memo[s]
            left = 0
            right = len(s) - 1
            count = 0
            while left < right:
                w1 = s[left]
                w2 = s[right]
                if w1 != w2:
                    count += 1
                    count += min(dp(s[left:right]), dp(s[left + 1 : right + 1]))
                    memo[s] = count
                    return count
                left += 1
                right -= 1
            memo[s] = count
            return count

        res = dp(s)
        if res > k:
            return False
        return True

    def longestPalindromeSubseq(self, s: str) -> int:

        memo = {}

        def dp(s):
            if s in memo:
                return memo[s]
            left = 0
            right = len(s) - 1
            size = 0
            while left < right:
                if s[left] == s[right]:
                    size += 2
                else:
                    size += max(dp(s[left:right]), dp(s[left + 1 : right + 1]))
                    memo[s] = size
                    return size
                left += 1
                right -= 1
            if len(s) % 2 == 1:
                size += 1
            memo[s] = size
            return size

        return dp(s)

    def isArraySpecial(self, nums: List[int], queries: List[List[int]]) -> List[bool]:

        def is_good_f(frm, to):
            if False in narr[frm:to]:
                return False
            return True

        N = len(nums)
        narr = [True] * N
        is_good = False
        for i in range(N - 1):
            cur = nums[i] % 2
            nxt = nums[i + 1] % 2
            is_good = True if cur + nxt == 1 else False
            narr[i] = is_good

        narr[-1][0] = is_good
        ans = []
        for frm, to in queries:
            res = is_good_f(frm, to)
            ans.append(res)
        return ans

    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max = -inf

        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            self.max = max(node.val + max(left, 0) + max(right, 0), self.max)
            return max(left, right, 0) + node.val

        total = max(dfs(root), self.max)
        return total

    def divide(self, dividend: int, divisor: int) -> int:
        count = 0
        sign = 1
        if dividend < 0:
            sign *= -1
            dividend = -dividend
        if divisor < 0:
            sign *= -1
            divisor = -divisor
        rem = 0
        while divisor > 1:
            dividend = dividend >> 1
            divisor /= 2
            rem += 0
        return int(dividend) * sign

    def isBipartite(self, graph: List[List[int]]) -> bool:
        """ "
        0  - o - 0
         \  |  /
            0
        """
        N = len(graph)
        indegree = [0] * N
        nodes = set([n for n in range(N)])

        def bfs(start):
            queue = deque([start])
            been = set()
            while queue:
                cur = queue.popleft()
                been.add(cur)
                for n in graph[start]:
                    if n not in been:
                        been.add(n)
                        queue.append(n)


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
