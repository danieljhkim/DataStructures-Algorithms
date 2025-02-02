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


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
