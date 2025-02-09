import heapq
import random
import math
import bisect
from typing import *
from math import inf, factorial
from heapq import heapify, heappush, heappop
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

    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        N = len(words)

        def add(idx):
            res = []
            size = 0
            while idx < N and len(words[idx]) + size <= maxWidth:
                res.append(words[idx])
                size += len(words[idx]) + 1
                idx += 1
            return res

        def align(idx, arr):
            spaces = -1
            for w in arr:
                spaces += len(arr) + 1

            extra_spaces = maxWidth - spaces
            if idx == N or len(w) == 1:
                return " ".join(arr) + " " * extra_spaces

            cnt = len(arr) - 1
            extra = extra_spaces % cnt
            space = extra_spaces // cnt
            for i in range(extra):
                arr[i] += " "
            for i in range(cnt):
                arr[i] += " " * space

            return " ".join(arr)

        res = []
        i = 0
        while i < N:
            line = add(i)
            i += len(line)
            res.append(align(i, line))
        return res

    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            else:
                if parent[x] != x:
                    parent[x] = find(x)
            return parent[x]

        def union(x, y):
            rootx = find(x)
            rooty = find(y)
            if rootx != rooty:
                parent[rootx] = parent[rooty]
                return True
            return False

        for x, y in edges:
            if not union(x, y):
                return False

        root = find(0)
        for i in range(1, n):
            if root != find(i):
                return False
        return True

    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        trie = {}
        N = len(s)

        def build(word):
            cur = trie
            for w in word:
                if w not in cur:
                    cur[w] = {}
                cur = cur[w]
            cur["*"] = word

        for w in wordDict:
            build(w)

        ans = []

        def backtrack(idx, arr, table):
            if idx == N:
                ans.append(" ".join(arr))
                return
            ntable = table
            for i in range(idx, N):
                cur = s[i]
                if cur not in ntable:
                    return
                ntable = ntable[cur]
                if "*" in ntable:
                    arr.append(ntable["*"])
                    backtrack(i + 1, arr, trie)
                    arr.pop()

        backtrack(0, [], trie)
        return ans

    def maxProfit(self, prices: List[int], fee: int) -> int:
        memo = {}
        N = len(prices)

        def dp(idx, has):
            if idx == N:
                return 0
            if (idx, has) in memo:
                return memo[(idx, has)]

            res = dp(idx + 1, has)
            if has:
                res = max(prices[idx] - fee + dp(idx + 1, False), res)
            else:
                res = max(dp(idx + 1) - prices[idx], res)
            memo[(idx, has)] = res
            return res

        return dp(0, False)

    def magnificentSets(self, n: int, edges: List[List[int]]) -> int:
        parent = {}
        adj = defaultdict(list)

        def find(x):
            if x not in parent:
                parent[x] = x
            else:
                if parent[x] != x:
                    parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rootx = find(x)
            rooty = find(y)
            if rootx != rooty:
                parent[rootx] = parent[rooty]

        for s, e in edges:
            adj[s].append(e)
            adj[e].append(s)
            union(s, e)

        def bfs(start):
            levels = {start: 0}
            queue = deque([start])

            while queue:
                cur = queue.popleft()
                cur_level = levels[cur]
                for nei in adj[cur]:
                    if nei in levels:
                        if abs(levels[nei] - cur_level) != 1:
                            return -1
                    else:
                        levels[nei] = cur_level + 1
                        queue.append(nei)

            return max(levels.values()) + 1

        total = 0
        best = {}
        for node in range(1, n + 1):
            layers = bfs(node)
            if layers == -1:
                return -1
            root = find(node)
            best[root] = max(best.get(root, 0), layers)

        total = sum(best.values())
        return total

    def longestSubarray(self, nums: List[int], limit: int) -> int:
        min_dq = deque()
        max_dq = deque()
        left = ans = 0
        for i, n in enumerate(nums):
            while min_dq and nums[min_dq[-1]] > n:
                min_dq.pop()
            while max_dq and nums[max_dq[-1]] < n:
                max_dq.pop()
            min_dq.append(i)
            max_dq.append(i)
            while min_dq and max_dq and nums[max_dq[0]] - nums[min_dq[0]] > limit:
                if min_dq[0] == left:
                    min_dq.popleft()
                if max_dq[0] == left:
                    max_dq.popleft()
                left += 1
            ans = max(ans, i - left + 1)
        return ans

    def longestSubarray(self, nums: List[int], limit: int) -> int:
        min_heap = []
        max_heap = []
        left = ans = 0

        for right, n in enumerate(nums):
            heapq.heappush(min_heap, (n, right))
            heapq.heappush(max_heap, (-n, right))
            while -max_heap[0][0] - min_heap[0][0] > limit:
                left = min(max_heap[0][1], min_heap[0][1]) + 1
                while max_heap[0][1] < left:
                    heapq.heappop(max_heap)
                while min_heap[0][1] < left:
                    heapq.heappop(min_heap)
            ans = max(ans, right - left + 1)
        return ans

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        R, C = len(matrix), len(matrix[0])
        left = top = 0
        right, bottom = C - 1, R - 1
        ans = []

        while len(ans) < R * C:
            # left right
            for i in range(left, right + 1):
                ans.append(matrix[top][i])
            top += 1

            # up down
            for i in range(top, bottom + 1):
                ans.append(matrix[i][right])
            right -= 1

            if len(ans) == R * C:
                break
            # right left
            for i in range(right, left - 1, -1):
                ans.append(matrix[bottom][i])
            bottom += 1

            if len(ans) == R * C:
                break
            # down up
            for i in range(bottom, top - 1, -1):
                ans.append(matrix[i][left])
            left += 1
        return ans

    def minCost(self, costs: List[List[int]]) -> int:
        memo = {}
        N = len(costs)
        choices = [0, 1, 2]

        def dp(i, color):
            if i == N:
                return 0
            if (i, color) in memo:
                return memo[(i, color)]
            res = inf
            for c in choices:
                if c != color:
                    res = min(res, dp(i + 1, c) + costs[i][c])
            memo[(i, color)] = res
            return res

        return min(dp(0, 0), dp(0, 1), dp(0, 2))

    def stoneGameII(self, piles: List[int]) -> int:
        N = len(piles)
        memo = {}

        def dp(i, m, is_a):
            if i == N:
                return 0
            if (i, m, is_a) in memo:
                return memo[(i, m)]
            res = 0
            points = 0
            if is_a:
                for j in range(1, 2 * m + 1):
                    if i + j == N:
                        break
                    points += piles[i + j]
                    res = max(-dp(i + j, max(m, j), False) + points, res)
            else:
                for j in range(1, 2 * m + 1):
                    if i + j == N:
                        break
                    points += piles[i + j]
                    res = min(dp(i + j, max(m, j), True) + points, res)
            memo[(i, m, is_a)] = res
            return res

        return dp(0, 1, True)

    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        trie = {}
        nset1 = set(arr1)
        nset2 = set(arr2)
        longest = 0

        def build(num):
            cur = trie
            for n in str(num):
                if n not in cur:
                    cur[n] = {}
                cur = cur[n]

        def prefix_len(num):
            size = 0
            cur = trie
            for n in str(num):
                if n not in cur:
                    return size
                cur = cur[n]
                size += 1
            return size

        for n in nset1:
            build(n)

        for n in nset2:
            longest = max(longest, prefix_len(n))

        return longest

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        heap = []
        ans = 0
        for s, e in intervals:
            while heap and heap[0][0] <= s:
                heapq.heappop(heap)
            heapq.heappush(heap, (e, s))
            ans = max(len(heap), ans)
        return ans

    def isArraySpecial(self, nums: List[int]) -> bool:
        parity = nums[0] % 2 == 1
        for n in nums[1:]:
            if n % 2 == 1 == parity:
                return False
            parity = n % 2 == 1
        return True

    def reorganizeString(self, s: str) -> str:
        arr = []
        counts = Counter(s)
        heap = [(-cnt, key) for key, cnt in counts.items()]
        heapq.heapify(heap)

        while heap:
            cnt, val = heapq.heappop(heap)
            if not arr or arr[-1] != val:
                arr.append(val)
                if cnt + 1 < 0:
                    heapq.heappush(heap, (cnt + 1, val))
            else:
                if not heap:
                    return ""
                cnt2, val2 = heapq.heappop(heap)
                arr.append(val2)
                if cnt2 + 1 < 0:
                    heapq.heappush(heap, (cnt2 + 1, val2))
                heapq.heappush(heap, (cnt, val))

        return "".join(arr)

    def distinctNumbers(self, nums: List[int], k: int) -> List[int]:
        table = defaultdict(int)
        for n in nums[:k]:
            table[n] += 1
        ans = [len(table)]

        for i in range(k, len(nums)):
            right = nums[i]
            left = nums[i - k]
            table[left] -= 1
            if table[left] == 0:
                table.pop(left)
            table[right] += 1
            ans.append(len(table))

        return ans

    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        """ "
        one word -> left align
        last row -> left align
        """

        N = len(words)

        def create_row(idx):
            row = []
            size = 0
            while idx < N and size + len(words[idx]) <= maxWidth:
                size += len(words[idx]) + 1
                row.append(words[idx])
                idx += 1
            return row, idx

        def pad_spaces(row, idx):
            letters_cnt = -1
            for r in row:
                letters_cnt += len(r) + 1

            free_space = maxWidth - letters_cnt
            if idx == N or len(row) == 1:
                return " ".join(row) + " " * free_space

            each_pad = free_space // (len(row) - 1)
            extra_pad = free_space % (len(row) - 1)
            for i in range(extra_pad):
                row[i] += " "
            for i in range(len(row) - 1):
                row[i] += " " * each_pad

            return " ".join(row)

        res = []
        idx = 0
        while idx < N:
            row, idx = create_row(idx)
            res.append(pad_spaces(row, idx))
        return res

    def longestSubarray(self, nums: List[int], limit: int) -> int:
        max_dq = deque()
        min_dq = deque()
        left, n = 0, len(nums)

        for right in nums:
            while max_dq and max_dq[-1] < n:
                max_dq.pop()
            while min_dq and min_dq[-1] > n:
                min_dq.pop()
            max_dq.append(right)
            min_dq.append(right)
            if nums[max_dq[0]] - nums[min_dq[0]] > limit:
                if max_dq[0] == left:
                    max_dq.popleft()
                if min_dq[0] == left:
                    min_dq.popleft()
                left += 1

        return n - left

    def addStrings(self, num1: str, num2: str) -> str:
        idx1, idx2 = len(num1) - 1, len(num2) - 1
        carry = 0
        res = []

        while idx1 >= 0 or idx2 >= 0:
            n1 = n2 = 0
            if idx1 >= 0:
                n1 = int(num1[idx1])
                idx1 -= 1
            if idx2 >= 0:
                n2 = int(num2[idx2])
                idx2 -= 1
            sum_num = n2 + n1 + carry
            if sum_num > 9:
                carry = 1
                sum_num = sum_num - 10
            else:
                carry = 0
            res.append(str(sum_num))
        if carry:
            res.append(str(carry))
        return "".join(reversed(res))

    def maxProfit(self, prices: List[int]) -> int:
        prof = 0
        buy = prices[0]

        for n in prices[1:]:
            if buy > n:
                buy = n
            elif buy < n:
                prof = max(prof, n - buy)
        return prof

    def findValidPair(self, s: str) -> str:
        counts = Counter(s)
        for i in range(len(s) - 1):
            n1 = s[i]
            n2 = s[i + 1]
            if n1 != n2:
                if counts[n1] == int(n1) and counts[n2] == int(n2):
                    return n1 + n2
        return ""

    def maxFreeTime(
        self, eventTime: int, k: int, startTime: List[int], endTime: List[int]
    ) -> int:
        """ "
        t = 0 -> eventTime
        [startTime[i], endTime[i]]
        """
        free_times = []
        N = len(startTime)
        prev_end = 0

        for i in range(N):
            s = startTime[i]
            free_times.append(s - prev_end)
            prev_end = endTime[i]
        if eventTime > prev_end:
            free_times.append(eventTime - prev_end)

        ans = cur = sum(free_times[: k + 1])
        left = 0
        for right in range(k + 1, len(free_times)):
            cur = cur - free_times[left] + free_times[right]
            ans = max(ans, cur)
            left += 1
        return ans

    def maxFreeTime(
        self, eventTime: int, startTime: List[int], endTime: List[int]
    ) -> int:
        heap = []
        N = len(startTime)
        prev_end = 0
        free_times = []

        for i in range(N):
            s = startTime[i]
            e = endTime[i]
            free_times.append((prev_end, s))
            heap.append((-(s - prev_end), prev_end, s))
            prev_end = e

        if eventTime >= prev_end:
            free_times.append((prev_end, eventTime))
            heap.append((-(eventTime - prev_end), prev_end, eventTime))

        heapq.heapify(heap)
        ans = 0
        for i in range(len(free_times) - 1):
            s1, e1 = free_times[i]
            s2, e2 = free_times[i + 1]
            event_size = s2 - e1
            temp = []
            while heap and event_size <= -heap[0][0]:
                size, ss, ee = heapq.heappop(heap)
                temp.append((size, ss, ee))
                if -size >= event_size and ss not in {s1, s2}:
                    free = (e1 - s1 + e2 - s2) + event_size
                    ans = max(free, ans)
                    break
            ans = max(ans, e1 - s1 + e2 - s2)
            for t in temp:
                heapq.heappush(heap, t)
        return ans

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        1 -> 2 -> none
        none <- 1 <- 2
        """
        prev = None
        cur = head
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        return prev

    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        table = defaultdict(int)
        time = events[0][1]
        table[time] = events[0][0]
        for event in events[1:]:
            dur = event[1] - time
            idx = event[0]
            if dur not in table or table[dur] > idx:
                table[dur] = idx
            time = event[1]
        top = max(table.keys())
        return table[top]

    def check(self, nums: List[int]) -> bool:
        flip = False
        prev = nums[0]
        for n in nums[1:]:
            if n < prev:
                if flip:
                    return False
                flip = True
            prev = n
        if flip and nums[-1] > nums[0]:
            return False
        return True

    def wordPatternMatch(self, pattern: str, s: str) -> bool:
        N, P = len(s), len(pattern)
        ans = False

        def backtrack(idx, pid, table, wset):
            nonlocal ans
            if idx == N and pid == P and len(table) == P:
                ans = True
                return
            if ans or idx == N or pid == P:
                return

            p = pattern[pid]
            for i in range(idx + 1, N + 1):
                word = s[idx:i]
                if word in table:
                    if p == table[word]:
                        backtrack(i, pid + 1, table, wset)
                else:
                    if p in wset:
                        continue
                    table[word] = p
                    wset.add(p)
                    backtrack(i, pid + 1, table, wset)
                    table.pop(word)
                    wset.remove(p)

        backtrack(0, 0, {}, set())
        return ans

    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(heights)
        COL = len(heights[0])
        atlantic = set()
        pacific = set()

        def dfs(r, c, visited):
            visited.add((nr, nc))
            cur = heights[r][c]
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL:
                    if heights[nr][nc] >= cur and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        dfs(nr, nc, visited)

        for r in range(ROW):
            dfs(r, COL - 1, atlantic)
            dfs(r, 0, pacific)

        for c in range(COL):
            dfs(ROW - 1, c, atlantic)
            dfs(0, c, pacific)

        return list(atlantic.intersection(pacific))

    def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        self.ans = 0

        def dfs(cur, arr):
            arr.append(str(cur.val))
            if not cur.left and not cur.right:
                num = "".join(arr)
                self.ans += int(num, 2)
                arr.pop()
                return

            if cur.left:
                dfs(cur.left, arr)
            if cur.right:
                dfs(cur.right, arr)
            arr.pop()

        dfs(root, [])
        return self.ans

    def generateSentences(self, synonyms: List[List[str]], text: str) -> List[str]:
        ans = []
        table = defaultdict(set)
        texts = text.split(" ")
        N = len(texts)

        def build_index(table):
            for s in synonyms:
                wset = set(s)
                for w in s:
                    for ww in table[w]:
                        wset.update(table[ww])
                        table[ww] = wset
                    wset.update(table[w])
                    table[w] = wset

        build_index(table)

        def backtrack(idx, arr):
            if idx == N:
                ans.append(" ".join(arr))
                return
            cur = texts[idx]
            if cur in table:
                for nw in table[cur]:
                    arr.append(nw)
                    backtrack(idx + 1, arr)
                    arr.pop()
            else:
                arr.append(cur)
                backtrack(idx + 1, arr)
                arr.pop()

        backtrack(0, [])
        ans.sort()
        return ans

    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        best, cur = 1, 1
        trend = 0
        N = len(nums)
        for i in range(1, N):
            if nums[i] > nums[i - 1]:
                if trend == 1:
                    cur += 1
                else:
                    cur = 2
                    trend = 1
            elif nums[i] < nums[i - 1]:
                if trend == -1:
                    cur += 1
                else:
                    cur = 2
                    trend = -1
            else:
                cur = 1
                trend = 0
            best = max(best, cur)
        return best

    def threeSumSmaller(self, nums: List[int], target: int) -> int:
        nums.sort()
        N = len(nums)
        ans = 0
        for i in range(N - 2):
            n1 = nums[i]
            left = i + 1
            right = N - 1
            while left < right:
                if n1 + nums[left] + nums[right] < target:
                    ans += right - left
                    left += 1
                else:
                    right -= 1
        return ans

    def maxAscendingSum(self, nums: List[int]) -> int:
        ans = total = nums[0]
        for i in range(1, len(nums)):
            if nums[i - 1] < nums[i]:
                total += nums[i]
                ans = max(total, ans)
            else:
                total = nums[i]
        return ans

    def minNumberOfFrogs(self, croakOfFrogs: str) -> int:
        table = defaultdict(list)
        mapper = {"r": "c", "o": "r", "a": "o", "k": "a"}
        cnt = 0
        outs = 0
        for i, n in enumerate(croakOfFrogs):
            if n == "c":
                table[n].append(i)
                outs += 1
                cnt = max(cnt, outs)
            else:
                if len(table[mapper[n]]) == 0:
                    return -1
                table[mapper[n]].pop()
                if n != "k":
                    table[n].append(i)
            if n == "k":
                if outs == 0:
                    return -1
                outs -= 1

        for v in table.values():
            if len(v) > 0:
                return -1
        return cnt

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        N = len(nums)
        idx = 0
        res = []
        while idx < N - 3:
            if idx > 0 and nums[idx] == nums[idx - 1]:
                idx += 1
                continue
            n0 = nums[idx]
            i = idx + 1
            while i < N - 2:
                if i > idx + 1 and nums[i] == nums[i - 1]:
                    i += 1
                    continue
                n1 = nums[i]
                left = i + 1
                right = N - 1
                while left < right:
                    total = n0 + n1 + nums[left] + nums[right]
                    if total < target:
                        left += 1
                    elif total > target:
                        right -= 1
                    else:
                        res.append([n0, n1, nums[left], nums[right]])
                        left_val = nums[left]
                        right_val = nums[right]
                        while left < right and nums[left] == left_val:
                            left += 1
                        while left < right and nums[right] == right_val:
                            right -= 1
                i += 1
            idx += 1
        return res

    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        if s2 == s1:
            return True

        N1, N2 = len(s1), len(s2)

        if N1 != N2:
            return False

        p1 = p2 = -1
        for i in range(N1):
            if s1[i] != s2[i]:
                if p1 == -1:
                    p1 = i
                elif p2 == -1:
                    p2 = i
                    if s1[p1] != s2[p2] or s1[p2] != s2[p1]:
                        return False
                else:
                    return False
        if p2 == -1:
            return False
        return True

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        """ "
        row/col -> 1-9
        3 x 3 -> 1-9
        """
        rtable = defaultdict(set)
        ctable = defaultdict(set)
        sqtable = defaultdict(set)

        for r in range(9):
            for c in range(9):
                val = board[r][c]
                if val != ".":
                    sq = (r // 3, c // 3)
                    if val in rtable[r] or val in ctable[c] or val in sqtable[sq]:
                        return False
                    rtable[r].add(val)
                    ctable[c].add(val)
                    sqtable[sq].add(val)
        return True

    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        """ "
        "# a.b.com"
        """
        table = defaultdict(int)
        res = []

        for d in cpdomains:
            splits = d.split(" ")
            cnt = int(splits[0])
            domains = splits[1].split(".")
            for i in range(len(domains)):
                key = ".".join(domains[i:])
                table[key] += cnt

        for k, v in table.items():
            res.append(f"{v} {k}")

        return res

    def alertNames(self, keyName: List[str], keyTime: List[str]) -> List[str]:
        table = defaultdict(list)
        N = len(keyName)

        for i in range(N):
            times = keyTime[i].split(":")
            hour = int(times[0]) * 60
            minute = int(times[1])
            time = hour + minute
            table[keyName[i]].append(time)

        res = []
        for k, times in table.items():
            right = 0
            N = len(times)
            for left in range(N):
                cur = times[left]
                while right < N and times[right] - cur <= 60:
                    right += 1
                if right - left >= 3:
                    res.append(k)
                    break
        res.sort()
        return res

    def checkValid(self, matrix: List[List[int]]) -> bool:
        rtable = defaultdict(set)
        ctable = defaultdict(set)
        N = len(matrix)
        for r in range(N):
            for c in range(N):
                val = matrix[r][c]
                if val in rtable[r] or val in ctable[c]:
                    return False
                rtable[r].add(val)
                ctable[c].add(val)

        for val in rtable.values():
            if len(val) != N:
                return False
        for val in ctable.values():
            if len(val) != N:
                return False
        return True

    def tupleSameProduct(self, nums: List[int]) -> int:
        products = defaultdict(int)
        N = len(nums)
        for i in range(N - 1):
            for j in range(i + 1, N):
                prod = nums[i] * nums[j]
                products[prod] += 1
        ans = 0
        for n in products.values():
            cnt = (n - 1) * n // 2  # wtf?
            ans += 8 * cnt
        return ans

    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        N = len(nums)
        stack = []
        ans = [-1] * N
        i = N - 1
        for i in range(N * 2 - 1):
            cur = nums[i % N]
            while stack and nums[stack[-1]] < cur:
                out = stack.pop()
                ans[out] = cur
            stack.append(i % N)
        return ans

    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        def dfs(node, prev):
            if not node:
                return 0
            diff = node.val - prev
            left = dfs(node.left, node.val)
            right = dfs(node.right, node.val)
            best = 0
            if diff == 1:
                #  inc
                if node.left:
                    if node.left.val - node.val == 1:
                        best = left + 1
                if node.right:
                    if node.right.val - node.val == 1:
                        best = max(right + 1, best)
            elif diff == -1:
                if node.left:
                    if node.left.val - node.val == -1:
                        best = left + 1
                if node.right:
                    if node.right.val - node.val == -1:
                        best = max(right + 1, best)

            best = max(best, left, right)
            return best

        return dfs(root, root.val - 5)

    def findOrder(self, n: int, prerequisites: List[List[int]]) -> List[int]:
        """ "
        [preq, course]
        """
        indegrees = [0] * n
        adj = defaultdict(list)
        for p, c in prerequisites:
            adj[c].append(p)
            indegrees[p] += 1

        queue = deque()
        for i, v in enumerate(indegrees):
            if v == 0:
                queue.append(i)

        topo = []
        while queue:
            cur = queue.popleft()
            topo.append(cur)
            for nei in adj[cur]:
                indegrees[nei] -= 1
                if indegrees[nei] == 0:
                    queue.append(nei)
        if len(topo) == n:
            return topo
        return []

    def increasingTriplet(self, nums: List[int]) -> bool:
        N = len(nums)
        if N < 3:
            return False
        first = inf
        second = inf
        for n in nums:
            if first >= n:
                first = n
            elif second >= n:
                second = n
            else:
                return True
        return False

    def maxVowels(self, s: str, k: int) -> int:
        vowels = "aeiou"
        cnt = 0
        for i in range(k):
            if s[i] in vowels:
                cnt += 1
        best = cnt
        left = 0
        for right in range(k, len(s)):
            if s[left] in vowels:
                cnt -= 1
            if s[right] in vowels:
                cnt += 1
            best = max(best, cnt)
            left += 1
        return best

    def maxOperations(self, nums: List[int], k: int) -> int:
        counts = Counter(nums)
        res = 0
        for key, v in counts.items():
            target = k - key
            if target in counts:
                if target != key:
                    out = min(counts[target], v)
                    res += out
                    counts[target] = 0
                else:
                    res += v // 2
        return res

    def removeStars(self, s: str) -> str:
        stack = []
        for w in s:
            if w == "*" and stack:
                stack.pop()
            else:
                stack.append(w)

        return "".join(stack)

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        if not root:
            return 0

        def dfs(cur, arr):
            val = cur.val
            res = 0
            arr.append(0)
            for i in range(len(arr)):
                arr[i] += val
                if arr[i] == targetSum:
                    res += 1
            if cur.left:
                res += dfs(cur.left, arr[:])
            if cur.right:
                res += dfs(cur.right, arr[:])
            return res

        return dfs(root, [])

    def goodNodes(self, root: TreeNode) -> int:
        if not root:
            return 0

        def dfs(node, top):
            val = node.val
            res = 0
            if top <= val:
                res += 1
            if node.left:
                res += dfs(node.left, max(val, top))
            if node.right:
                res += dfs(node.right, max(val, top))
            return res

        return dfs(root, root.val)

    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        def dfs(node, is_left, cnt):
            if not node:
                return 0
            res = 0
            if is_left:
                res = dfs(node.left, True, 1)
                res = max(dfs(node.right, False, cnt + 1), res)
            else:
                res = dfs(node.left, True, cnt + 1)
                res = max(dfs(node.right, False, 1), res)
            return max(res, cnt)

        return max(dfs(root.left, True, 1), dfs(root.right, False, 1))

    def queryResults(self, limit: int, queries: List[List[int]]) -> List[int]:
        btable = {}
        ctable = defaultdict(set)
        res = []
        for b, c in queries:
            if b in btable:
                prev = btable[b]
                del btable[b]
                ctable[prev].remove(b)
                if len(ctable[prev]) == 0:
                    del ctable[prev]
            btable[b] = c
            ctable[c].add(b)
            res.append(len(ctable))
        return res

    def permute(self, n: int) -> List[List[int]]:

        def is_good(n1, n2):
            return n2 % 2 != n1 % 2

        ans = []

        def backtrack(arr, nset):
            if len(arr) == n:
                ans.append(arr[:])
            for i in range(1, n + 1):
                if i not in nset:
                    if (arr and is_good(i, arr[-1])) or not arr:
                        arr.append(i)
                        nset.add(i)
                        backtrack(arr, nset)
                        arr.pop()
                        nset.remove(i)

        backtrack([], set())
        return ans

    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        R = len(board)
        C = len(board[0])
        rtable = defaultdict(set)
        ctable = defaultdict(set)
        stable = defaultdict(set)
        self.solved = False

        for r in range(R):
            for c in range(C):
                val = board[r][c]
                if val != ".":
                    rtable[r].add(val)
                    ctable[c].add(val)
                    stable[(r // 3, c // 3)].add(val)

        def place_val(val, r, c):
            board[r][c] = val
            rtable[r].add(val)
            ctable[c].add(val)
            stable[(r // 3, c // 3)].add(val)

        def remove_val(val, r, c):
            board[r][c] = "."
            rtable[r].remove(val)
            ctable[c].remove(val)
            stable[(r // 3, c // 3)].remove(val)

        def backtrack(rr, cc):
            if rr == R and cc == 0:
                self.solved = True
            if self.solved:
                return

            if board[rr][cc] == ".":
                for val in range(1, 10):
                    val = str(val)
                    if (
                        val not in rtable[rr]
                        and val not in ctable[cc]
                        and val not in stable[(rr // 3, cc // 3)]
                    ):
                        place_val(val, rr, cc)
                        if cc == C - 1:
                            backtrack(rr + 1, 0)
                        else:
                            backtrack(rr, cc + 1)
                        if self.solved:
                            return
                        remove_val(val, rr, cc)
            else:
                if cc == C - 1:
                    backtrack(rr + 1, 0)
                else:
                    backtrack(rr, cc + 1)
                if self.solved:
                    return

        backtrack(0, 0)

    def firstMissingPositive(self, nums: List[int]) -> int:
        N = len(nums)
        i = 0

        while i < N:
            val = nums[i]
            if 0 < val <= N and nums[i] != nums[val - 1]:
                nums[i], nums[val - 1] = nums[val - 1], nums[i]
            else:
                i += 1

        for i, n in enumerate(nums):
            if n != i + 1:
                return i + 1

        return N + 1

    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        visited = set([0])
        N = len(rooms)

        def dfs(i, visited):
            if len(visited) == N:
                return True
            keys = rooms[i]
            for k in keys:
                if k not in visited:
                    visited.add(k)
                    res = dfs(k, visited)
                    if res:
                        return True
            return False

        return dfs(0, visited)

    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        N = len(isConnected)
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            else:
                if parent[x] != x:
                    parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rootx = find(x)
            rooty = find(y)
            if rootx != rooty:
                parent[rootx] = parent[rooty]

        for i, group in enumerate(isConnected):
            root = find(i)
            for c, v in enumerate(group):
                if c == i:
                    continue
                if v == 1:
                    union(root, c)

        gset = set()
        for i in range(N):
            gset.add(find(i))
        return len(gset)

    def longestSubarray(self, nums: List[int]) -> int:
        ans = right = left = prev = 0
        zero = 0
        N = len(nums)

        while right < N:
            if nums[right] == 0:
                zero += 1
            while zero > 1:
                zero += -1 if nums[left] == 0 else 0
                left += 1
            if right - left + 1 == N:
                return N - 1
            ans = max(ans, right - left)
            right += 1
        return ans

    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """ "
        0 - 1 - 2 - 3 - 4
            s   f
                s       f
        """
        slow = head
        fast = head
        prev = None
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next

        if prev is None:
            return None
        prev.next = slow.next
        slow.next = None
        return head

    def pairSum(self, head: Optional[ListNode]) -> int:
        table = defaultdict(int)
        i = ans = 0
        slow = head
        fast = head
        while fast and fast.next:
            table[i] = slow.val
            i += 1
            slow = slow.next
            fast = fast.next.next

        del table[i]
        i -= 1
        while slow:
            table[i] += slow.val
            ans = max(table[i], ans)
            slow = slow.next
            i -= 1
        return ans

    def countBadPairs(self, nums: List[int]) -> int:
        """ "
        bad -> j - i != nj - ni

        good -> j - nj == i - ni
        """
        table = defaultdict(int)
        table[-nums[0]] += 1
        total = bads = 0
        for j in range(1, len(nums)):
            need = j - nums[j]
            if need in table:
                bads += table[need]
            table[need] += 1
            total += j
        return total - bads

    def longestSubarray(self, nums: List[int], limit: int) -> int:
        min_heap = []
        max_heap = []
        ans = 0
        left = -1
        for right, n in enumerate(nums):
            heappush(min_heap, (n, right))
            heappush(max_heap, (-n, right))
            while -max_heap[0][0] - min_heap[0][0] > limit:
                if max_heap[0][1] > min_heap[0][1]:
                    out, idx = heappop(min_heap)
                    left = max(idx, left)
                else:
                    out, idx = heappop(max_heap)
                    left = max(idx, left)
            ans = max(ans, right - left)
        return ans

    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        N = len(words)

        def create_line(idx):
            line = []
            cnt = -1
            while idx < N and cnt + len(words[idx]) + 1 <= maxWidth:
                cnt += len(words[idx]) + 1
                line.append(words[idx])
                idx += 1
            return line

        def space_it(idx, line):
            wlen = -1
            for w in line:
                wlen += len(w) + 1

            extra_len = maxWidth - wlen
            if idx == N or len(line) == 1:
                return " ".join(line) + " " * extra_len

            W = len(line) - 1
            per_w = extra_len // W
            extra_w = extra_len % W

            for i in range(extra_w):
                line[i] += " "
            for i in range(W):
                line[i] += " " * per_w

            return " ".join(line)

        res = []
        idx = 0
        while idx < N:
            line = create_line(idx)
            idx += len(line)
            sentence = space_it(idx, line)
            res.append(sentence)
        return res

    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        etable = {}
        for i, n in enumerate(elements):
            if n not in etable:
                etable[n] = i

        top = max(groups) + 1
        divisions = [-1] * top
        for n, i in etable.values():
            for d in range(n, top, n):
                if divisions[d] == -1:
                    divisions[d] = i

        res = []
        for n in groups:
            res.append(divisions[n])
        return res


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
