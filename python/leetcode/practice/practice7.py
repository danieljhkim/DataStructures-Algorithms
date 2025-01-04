import heapq
from typing import Optional, List
from itertools import accumulate
import random
from collections import Counter, deque, defaultdict, OrderedDict
import math
import bisect
from math import e, inf
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

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        """ "
        1 2 3
            [3] 4 5 6
        """
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        N1 = len(nums1)
        N2 = len(nums2)
        median = (N1 + N2 + 1) // 2
        low = 0
        high = N1
        while low <= high:
            mid = (high + low) // 2
            mid2 = median - mid
            left1 = -inf if mid < 1 else nums1[mid - 1]
            right1 = inf if mid >= N1 else nums1[mid]
            left2 = -inf if mid2 < 1 else nums2[mid2 - 1]
            right2 = inf if mid >= N2 else nums2[mid2]
            if left1 <= right2 and left2 <= right1:
                if (N1 + N2) % 2 == 1:
                    return max(left1, left2)
                res = (max(left2, left1) + min(right1, right2)) / 2
                return res
            elif left1 > right2:
                high = mid - 1
            else:
                low = mid + 1

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        N1 = len(nums1)
        N2 = len(nums2)

        def search(median, low1, high1, low2, high2):
            if high1 < low1:
                return nums2[median - low1]
            if high2 < low2:
                return nums1[median - low2]
            mid1 = (high1 + low1) // 2
            mid2 = (high2 + low2) // 2
            val1 = nums1[mid1]
            val2 = nums2[mid2]
            if mid1 + mid2 < median:
                if val1 < val2:
                    return search(median, mid1 + 1, high1, low2, high2)
                return search(median, low1, high1, mid2 + 1, high2)
            else:
                if val1 > val2:
                    return search(median, low1, mid1 - 1, low2, high2)
                return search(median, low1, high1, low2, mid2 - 1)

        median = (N1 + N2) // 2
        if (N1 + N2) % 2 == 1:
            return search(median, 0, N1 - 1, 0, N2 - 1)
        else:
            return (
                search(median, 0, N1 - 1, 0, N2 - 1)
                + search(median - 1, 0, N1 - 1, 0, N2 - 1)
            ) / 2

    def trap(self, height: List[int]) -> int:
        N = len(height)
        ans = 0
        left = 0
        right = N - 1
        max_left = 0
        max_right = 0
        while left <= right:
            cur_left = height[left]
            cur_right = height[right]
            max_left = max(cur_left, max_left)
            max_right = max(cur_right, max_right)
            if max_left < max_right:
                ans += max_left - cur_left
                left += 1
            else:
                ans += max_right - cur_right
                right -= 1

        return ans

    def trap(self, height: List[int]) -> int:
        if not height:
            return 0

        left = 0
        right = len(height) - 1
        left_max = height[left]
        right_max = height[right]
        ans = 0

        while left < right:
            if height[left] < height[right]:
                left += 1
                left_max = max(left_max, height[left])
                ans += max(0, left_max - height[left])
            else:
                right -= 1
                right_max = max(right_max, height[right])
                ans += max(0, right_max - height[right])

        return ans

    def maxAverageRatio(self, classes: List[List[int]], extraStudents: int) -> float:
        ratios = []
        fulls = []
        for p, t in classes:
            if t == p:
                fulls.append((t, p))
            else:
                ratios.append((t, p))
        heapq.heapify(ratios)
        total = 0
        ext = extraStudents
        while ext > 0:
            cur_t, cur_p = heapq.heappop(ratios)
            cur_t += 1
            cur_p += 1
            heapq.heappush(ratios, (cur_t, cur_p))
            ext -= 1

        for t, p in ratios:
            total += p / t
        if fulls:
            for t, p in fulls:
                total += p / t
        return total / len(classes)

    def maxAverageRatio(self, classes: List[List[int]], extraStudents: int) -> float:
        def potential_gain(p, t):
            return (p + 1) / (t + 1) - p / t

        gains = []
        for p, t in classes:
            pg = potential_gain(p, t)
            gains.append((-pg, p, t))

        heapq.heapify(gains)
        total = 0
        ext = extraStudents
        while ext > 0:
            g, cur_p, cur_t = heapq.heappop(gains)
            cur_t += 1
            cur_p += 1
            heapq.heappush(gains, (-potential_gain(cur_p, cur_t), cur_p, cur_t))
            ext -= 1

        for g, p, t in gains:
            total += p / t

        return total / len(classes)

    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        ans = numBottles
        rem = 0
        while numBottles + rem >= numExchange:
            numBottles += rem
            ans += numBottles // numExchange
            rem = numBottles % numExchange
            numBottles //= numExchange
        return ans

    def minSteps(self, s: str, t: str) -> int:
        scounts = Counter(s)
        tcounts = Counter(t)
        lsum = 0
        rsum = 0
        for k, v in scounts.items():
            tv = tcounts[k]
            if tcounts[k] < v:
                lsum += v - tv
            elif tcounts[k] > v:
                rsum += tv - v
        ans = 0
        ans += min(rsum, lsum)
        ans += abs(rsum - lsum)
        return ans

    def distributeCoins(self, root: Optional[TreeNode]) -> int:

        self.steps = 0

        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            diff = node.val - 1 + left + right
            self.steps += abs(diff)

            return diff

        dfs(root)
        return self.steps

    def knightDialer(self, n: int) -> int:
        """ "
        136006598
        862628006
        725256005
        """
        table = defaultdict(list)
        table[0].extend([4, 6])
        table[1].extend([8, 6])
        table[2].extend([7, 9])
        table[3].extend([8, 4])
        table[4].extend([0, 3, 9])
        table[6].extend([7, 1, 0])
        table[7].extend([2, 6])
        table[8].extend([1, 3])
        table[9].extend([2, 4])

        cache = {}

        def recurs(i, length):
            if length == n:
                return 1
            if (i, length) in cache:
                return cache[(i, length)]
            count = 0
            for neigh in table[i]:
                count += recurs(neigh, length + 1)
            cache[(i, length)] = count
            return count

        ans = 0
        for i in range(0, 10):
            ans += recurs(i, 1)
        return ans % (10**9 + 7)

    def maximumLengthOfRanges(self, nums: List[int]) -> List[int]:
        N = len(nums)
        ans = []
        lcache = {}
        rcache = {}

        def checkLeft(i, val):
            if i == -1:
                return i + 1
            if i in lcache:
                if nums[lcache[i]] > val:
                    return lcache[i]
            if val < nums[i]:
                res = i + 1
                lcache[i] = res
            else:
                res = checkLeft(i - 1, val)
            return res

        def checkRight(i, val):
            if i == N:
                return i - 1
            if i in rcache:
                if nums[rcache[i]] > val:
                    return rcache[i]
            if val < nums[i]:
                res = i - 1
                rcache[i] = res
            else:
                res = checkRight(i + 1, val)
            return res

        for i, n in enumerate(nums):
            left = checkLeft(i, n)
            right = checkRight(i, n)
            ans.append(right - left + 1)
        return ans

    def maximumLengthOfRanges(self, nums: List[int]) -> List[int]:
        stack = []
        N = len(nums)
        right = [0] * N
        left = [0] * N

        for i, n in enumerate(nums):
            while stack and nums[stack[-1]] < n:
                stack.pop()
            if stack:
                left[i] = stack[-1] + 1
            stack.append(i)

        stack.clear()
        for i in range(N - 1, -1, -1):
            n = nums[i]
            while stack and nums[stack[-1]] < n:
                stack.pop()
            if stack:
                right[i] = stack[-1] - 1
            else:
                right[i] = N - 1
            stack.append(i)
        ans = [right[i] - left[i] + 1 for i in range(N)]
        return ans

    def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        """ "
        sub4u4 ("sub stit u tion")
        "internationalization", abbr = "i12iz4n"
        """
        N = len(word)
        BN = len(abbr)
        wi = 0
        ai = 0
        while wi < N and ai < BN:
            num = ""
            if abbr[ai] == "0":
                return False
            while ai < BN and abbr[ai].isdigit():
                num += abbr[ai]
                ai += 1
            if num:
                wi += int(num) + 1
                continue
            a = abbr[ai]
            w = word[wi]
            if a != w:
                return False
            wi += 1
            ai += 1
        if wi == N and ai == BN:
            return True
        return False

    def lowestCommonAncestor(self, p: "Node", q: "Node") -> "Node":
        hset = set()
        curp = p
        curq = q

        while curp and curq:
            if curp in hset:
                return curp
            if curq in hset:
                return curq
            hset.add(curp)
            hset.add(curq)
            curp = curp.parent
            curq = curq.parent
            if curp == curq:
                return curp

        top = curp if curp else curq

        while top:
            if top in hset:
                return top
            top = top.parent

    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        def dfs(node):
            if not node:
                return 0
            cur = 0
            if node.val >= low:
                cur += dfs(node.left)
            cur += node.val
            if node.val <= high:
                cur += dfs(node.right)
            return cur

        return dfs(root)

    def longestOnes(self, nums: List[int], k: int) -> int:
        left = 0
        zeros = 0
        N = len(nums)
        ans = 0
        for right, n in enumerate(nums):
            if n == 0:
                zeros += 1
            while left < right and (zeros > k):
                if nums[left] == 0:
                    zeros -= 1
                left += 1
            ans = max(ans, right - left + 1)

        return ans

    def moveZeroes(self, nums: List[int]) -> None:
        N = len(nums)
        pos = 0
        for i in range(N):
            n = nums[i]
            if n == 0:
                pos = max(i, pos)
                while pos < N and nums[pos] == 0:
                    pos += 1
                if pos >= N:
                    break
                nums[i], nums[pos] = nums[pos], nums[i]
                pos += 1

    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:

        while k > 0:
            small = nums[0]
            chosen = 0
            for i, n in enumerate(nums):
                if n < small:
                    chosen = i
                    small = n
            nums[chosen] *= multiplier
            k -= 1
        return nums

    def finalPrices(self, prices: List[int]) -> List[int]:
        stack = []
        for i, n in enumerate(prices):
            while stack and prices[stack[-1]] >= n:
                idx = stack.pop()
                prices[idx] -= prices[i]
            stack.append(i)
        return prices

    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        table = defaultdict(list)
        N = len(beginWord)
        for word in wordList:
            for i in range(N):
                w = word[:i] + "*" + word[i + 1 :]
                table[w].append(word)

        queue = deque([(beginWord, 1)])
        visited = set([beginWord])

        while queue:
            cur, dist = queue.popleft()
            if cur == endWord:
                return dist
            candidates = []
            for i in range(N):
                candidates.append(cur[:i] + "*" + cur[i + 1 :])
            for w in candidates:
                for dest in table[w]:
                    if dest not in visited:
                        queue.append((dest, dist + 1))
                        visited.add(dest)
        return 0

    def findLadders(
        self, beginWord: str, endWord: str, wordList: List[str]
    ) -> List[List[str]]:
        table = defaultdict(list)
        N = len(beginWord)
        for word in wordList:
            for i in range(N):
                w = word[:i] + "*" + word[i + 1 :]
                table[w].append(word)

        queue = deque([(beginWord, [beginWord])])
        visited = set([beginWord])
        ans = []
        shortest = inf
        while queue:
            cur, dist = queue.popleft()
            if cur == endWord:
                if len(dist) <= shortest:
                    ans.append(dist)
                    shortest = min(shortest, len(dist))
                else:
                    break
            candidates = []
            visited.add(cur)
            for i in range(N):
                candidates.append(cur[:i] + "*" + cur[i + 1 :])
            for w in candidates:
                for dest in table[w]:
                    if dest not in visited:
                        ndist = dist[:]
                        ndist.append(dest)
                        queue.append((dest, ndist))
        return ans

    def calculate(self, s: str) -> int:
        """ "
        3+3+2*2*2+1
        2*2+1
        -2+2-1


        prev * and when + -> add to total
        prev * and when * -> calc and sotre in subtotal
        prev +, store in subtotal
        """
        sub_total = 0
        total = 0
        prev_sign = "+"
        cur = 0
        for n in s + "+":
            if n == " ":
                continue
            if n.isdigit():
                cur *= 10
                cur += int(n)
                continue
            if prev_sign == "+":
                total += sub_total
                sub_total = cur
            elif prev_sign == "-":
                total += sub_total
                sub_total = -cur
            elif prev_sign == "*":
                sub_total *= cur
            elif prev_sign == "/":
                sub_total = int(sub_total / cur)
            cur = 0
            prev_sign = n
        return total + sub_total

    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        """ "
        gas = [1,2,3,4,5], cost = [3,4,5,1,2]

        [-2, -2, -2, 3, 3]
        """
        totals = []
        N = len(gas)

        for i in range(N):
            totals.append(gas[i] - cost[i])
        if sum(totals) < 0:
            return -1

        idx = 0
        total = 0
        top = -10000
        right = N - 1
        while 0 < right:
            total += totals[right]
            if total >= top:
                idx = right
                top = total
            right -= 1
        return idx

    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        ROW = len(matrix)
        COL = len(matrix[0])
        rremove = set()
        cremove = set()

        for r in range(ROW):
            for c in range(COL):
                if matrix[r][c] == 0:
                    rremove.add(r)
                    cremove.add(c)

        for r in rremove:
            for i in range(COL):
                matrix[r][i] = 0
        for c in cremove:
            for i in range(ROW):
                matrix[i][c] = 0

    def maxKDivisibleComponents(
        self, n: int, edges: List[List[int]], values: List[int], k: int
    ) -> int:
        self.count = 0
        adj = defaultdict(list)
        for n1, n2 in edges:
            adj[n1].append(n2)
            adj[n2].append(n1)

        def dfs(cur, parent):
            total = values[cur]
            for neigh in adj[cur]:
                if neigh != parent:
                    total += dfs(neigh, cur)
            if total % k == 0:
                self.count += 1
            return total

        dfs(0, -1)
        return self.count

    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        ROW = len(grid)
        COL = len(grid[0])
        cache = {}

        def recurs(r, c, total):
            if r == ROW - 1 and c == COL - 1 and total == k:
                return 1
            if (r, c, total) in cache:
                return cache[(r, c, total)]
            count = 0
            if r + 1 < ROW:
                ttotal = total ^ grid[r + 1][c]
                count += recurs(r + 1, c, ttotal)
            if c + 1 < COL:
                ttotal = total ^ grid[r][c + 1]
                count += recurs(r, c + 1, ttotal)
            cache[(r, c, total)] = count % (10**9 + 7)
            return cache[(r, c, total)]

        return recurs(0, 0, grid[0][0]) % (10**9 + 7)

    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        x = []
        y = []
        for sx, sy, ex, ey in rectangles:
            x.append((sx, ex))
            y.append((sy, ey))

        def isGood(intervals):
            intervals.sort()
            heap = [intervals[0][1]]
            count = 0
            for i in range(1, len(intervals)):
                start, end = intervals[i]
                while heap and heap[0] <= start:
                    heapq.heappop(heap)
                if not heap:
                    count += 1
                heapq.heappush(heap, end)
            return count >= 2

        return isGood(x) or isGood(y)

    def connect(self, root: "Node") -> "Node":
        if not root:
            return root
        queue = deque([root])
        while queue:
            size = len(queue)
            arr = []
            for _ in range(size):
                cur = queue.popleft()
                arr.append(cur)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            for i in range(1, size):
                prev = arr[i - 1]
                nxt = arr[i]
                prev.next = nxt
        return root

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        indegree = [0] * numCourses
        adj = defaultdict(list)
        for end, start in prerequisites:
            indegree[end] += 1
            adj[start].append(end)

        topo_sorted = []
        zeros = deque()
        for i, c in enumerate(indegree):
            if c == 0:
                zeros.append(i)

        while zeros:
            cur = zeros.popleft()
            topo_sorted.append(cur)
            for nei in adj[cur]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    zeros.append(nei)

        if len(topo_sorted) == numCourses:
            return topo_sorted
        return []

    def snakesAndLadders(self, board: List[List[int]]) -> int:
        N = len(board)
        END = N * N

        def current_pos(n):
            row = (n - 1) // N
            col = (n - 1) % N
            if row % 2 == 1:
                col = (N - 1) - col
            row = N - 1 - row
            return row, col

        heap = [(0, 0, False)]
        visited = set()
        while heap:
            moves, cur, used = heapq.heappop(heap)
            cur = cur
            if cur >= END - 1:
                return moves
            if (cur, used) in visited:
                continue
            visited.add((cur, used))
            for i in range(1, 7):
                ndist = cur + i
                nr, nc = current_pos(ndist)
                if nr < N and nc < N:
                    val = board[nr][nc]
                    if used or val == -1:
                        new_pos = ndist
                        new_used = False
                    else:
                        new_pos = val
                        new_used = True
                    heapq.heappush(heap, (moves + 1, new_pos, new_used))
        return -1

    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        cur_max = 0
        cur_min = 0
        max_sum = nums[0]
        min_sum = nums[0]
        N = len(nums)
        for n in nums:
            cur_max = max(cur_max + n, n)
            max_sum = max(cur_max, max_sum)
            cur_min = min(cur_min + n, n)
            min_sum = min(cur_min, min_sum)
        total = sum(nums)
        if min_sum == total:
            return max_sum
        return max(max_sum, total - min_sum)

    def kSmallestPairs(
        self, nums1: List[int], nums2: List[int], k: int
    ) -> List[List[int]]:
        ans = []
        N1 = len(nums1)
        N2 = len(nums2)
        heap = [(nums1[0] + nums2[0], (0, 0))]
        visited = set([(0, 0)])
        while len(ans) < k and heap:
            total, (i, j) = heapq.heappop(heap)
            ans.append([nums1[i], nums2[j]])
            if i + 1 < N1 and (i + 1, j) not in visited:
                visited.add((i + 1, j))
                heapq.heappush(heap, (nums1[i] + nums2[j + 1], (i + 1, j)))
            if j + 1 < N2 and (i, j + 1) not in visited:
                visited.add((i, j + 1))
                heapq.heappush(heap, (nums1[i] + nums2[j + 1], (i, j + 1)))
        return ans

    def leftmostBuildingQueries(
        self, heights: List[int], queries: List[List[int]]
    ) -> List[int]:
        cache = {}

        def find(i):
            if i in cache:
                return cache[i]
            while i > 0 and heights[i - 1] < heights[i]:
                i -= 1
            cache[i] = i
            return i

        N = len(queries)
        ans = [-1] * N
        i = 0
        for a, b in queries:
            big = max(a, b)
            small = min(a, b)
            big_left = find(big)
            small_left = find(small)
            if big_left < small:
                ans[i] = max(small_left, big_left)
            i += 1
        return ans

    def minimumOperations(self, root: Optional[TreeNode]) -> int:
        def swaps(nums):
            target = sorted(nums)
            ans = 0
            table = {val: idx for idx, val in enumerate(nums)}
            for i in range(len(nums)):
                if target[i] != nums[i]:
                    idx = table[target[i]]
                    table[nums[i]] = idx
                    nums[idx] = nums[i]
                    ans += 1
            return ans

        if not root:
            return root
        queue = deque([root])
        ans = 0
        while queue:
            size = len(queue)
            nums = []
            for i in range(size):
                cur = queue.popleft()
                nums.append(cur.val)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            ans += swaps(nums)
        return ans

    def minimumDiameterAfterMerge(
        self, edges1: List[List[int]], edges2: List[List[int]]
    ) -> int:

        def bfs(adj, start):
            longest = 0
            ans = start
            queue = deque([(start, 0)])
            visited = set([start])
            while queue:
                node, dist = queue.popleft()
                for nei in adj[node]:
                    if nei not in visited:
                        queue.append((nei, dist + 1))
                        if dist + 1 > longest:
                            longest = dist + 1
                            ans = nei
                        visited.add(nei)
            return ans, longest

        adj1 = defaultdict(list)
        adj2 = defaultdict(list)

        for u, v in edges1:
            adj1[v].append(u)
            adj1[u].append(v)
        for u, v in edges2:
            adj2[v].append(u)
            adj2[u].append(v)

        end1, _ = bfs(adj1, 0)
        _, diameter1 = bfs(adj1, end1)

        end2, _ = bfs(adj2, 0)
        _, diameter2 = bfs(adj2, end2)

        half = math.ceil(diameter1 / 2) + math.ceil(diameter2 / 2) + 1
        return max(half, diameter1, diameter2)

    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        ans = []
        queue = deque([root])
        while queue:
            size = len(queue)
            large = float("-inf")
            for i in range(size):
                cur = queue.popleft()
                if cur.val > large:
                    large = cur.val
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            ans.append(large)
        return ans

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        ans = []
        queue = deque([root])
        while queue:
            size = len(queue)
            level = []
            for _ in range(size):
                cur = queue.popleft()
                level.append(cur.val)
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            ans.append(level)
        return ans

    def singleNonDuplicate(self, nums: List[int]) -> int:
        """ "
        1 1 2 2 3 4 4
        """
        N = len(nums)
        low = 0
        high = N - 1
        while low < high:
            mid = (high + low) // 2
            if mid % 2 == 1:
                mid -= 1
            if nums[mid] != nums[mid + 1]:
                high = mid
            else:
                low = mid + 2
        return nums[low]

    def highFive(self, items: List[List[int]]) -> List[List[int]]:
        table = defaultdict(list)
        for id, val in items:
            heap = table[id]
            if len(heap) < 5:
                heapq.heappush(heap, val)
            else:
                if heap[0] < val:
                    heapq.heappop(heap)
                    heapq.heappush(heap, val)

        ans = []
        small = min(table)
        big = max(table)
        for i in range(small, big + 1):
            if i in table:
                heap = table[i]
                avg = int(sum(heap) / 5)
                ans.append([i, avg])
        return ans

    def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        self.left = []
        self.right = []

        def right_dfs(node, is_bounds):
            if is_bounds or (not node.left and not node.right):
                self.right.append(node.val)
            if node.right:
                right_dfs(node.right, is_bounds)
            if node.left:
                right_dfs(node.left, is_bounds and not node.right)

        def left_dfs(node, is_bounds):
            if is_bounds or (not node.left and not node.right):
                self.left.append(node.val)
            if node.left:
                left_dfs(node.left, is_bounds)

            if node.right:
                left_dfs(node.right, is_bounds and not node.left)

        if root.right:
            right_dfs(root.right, True)
        if root.left:
            left_dfs(root.left, True)
        self.right.reverse()
        return [root.val] + self.left + self.right

    def removeVowels(self, s: str) -> str:
        vset = set(["a", "e", "i", "o", "u"])
        ans = []
        for w in s:
            if w not in vset:
                ans.append(w)
        return "".join(ans)

    def canWinNim(self, n: int) -> bool:
        return n % 4 != 0

    def hammingDistance(self, x: int, y: int) -> int:
        xb = bin(x)[2:]
        yb = bin(y)[2:]
        NX = len(xb)
        NY = len(yb)
        size = max(NX, NY)
        xb = xb.zfill(size)
        yb = yb.zfill(size)
        ans = 0
        idx = 0
        while idx < size:
            if xb[idx] != yb[idx]:
                ans += 1
            idx += 1
        return ans

    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        self.p = False
        self.q = False

        def dfs(node):
            if not node:
                return node
            left = dfs(node.left)
            right = dfs(node.right)
            if left and right:
                return node
            if node == p:
                self.p = True
            if node == q:
                self.q = True
            if not left or not right:
                if node == p or node == q:
                    return node
            return left or right

        ans = dfs(root)
        if self.q and self.p:
            return ans
        return None

    def maxProduct(self, nums: List[int]) -> int:
        prefix = [nums[0]]
        for n in nums[1:]:
            if prefix[-1] != 0:
                new = prefix[-1] * n
            else:
                new = n
            prefix.append(new)
        top = max(prefix)
        prefix = [nums[-1]]
        for i in range(len(nums) - 2, -1, -1):
            n = nums[i]
            if prefix[-1] != 0:
                new = prefix[-1] * n
            else:
                new = n
            prefix.append(new)
        top = max(max(prefix), top)
        pos = 1
        for n in nums:
            if n > 0:
                pos *= n
                top = max(top, pos)
            else:
                pos = 1

        return top

    def longestSubstring(self, s: str, k: int) -> int:
        if len(s) < k:
            return 0
        counts = Counter(s)

        for key, val in counts.items():
            if val < k:
                subs = s.split(key)
                top = 0
                for st in subs:
                    top = max(self.longestSubstring(st, k), top)
                return top
        return len(s)

    def floodFill(
        self, image: List[List[int]], sr: int, sc: int, color: int
    ) -> List[List[int]]:
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(image)
        COL = len(image[0])
        visited = set()

        def dfs(r, c, val):
            if image[r][c] != val:
                return
            image[r][c] = color
            visited.add((r, c))
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        dfs(nr, nc, val)

        val = image[sr][sc]
        dfs(sr, sc, val)
        return image

    def sufficientSubset(
        self, root: Optional[TreeNode], limit: int
    ) -> Optional[TreeNode]:
        def dfs(node, total):
            if not node:
                return float("-inf")
            total += node.val
            if not node.left and not node.right:
                return total
            left = dfs(node.left, total)
            right = dfs(node.right, total)
            if left < limit:
                node.left = None
            if right < limit:
                node.right = None
            return max(left, right)

        dfs(root, 0)
        if not root:
            return root
        if root.val < limit and not root.left and not root.right:
            return None
        return root

    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:

        def dfs(node):
            if not node:
                return node
            if node.val == val:
                return node
            left = dfs(node.left)
            right = dfs(node.right)
            return left or right

        return dfs(root)

    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:

        indegree = [0] * (n + 1)
        adj = defaultdict(list)
        for prev, nxt in relations:
            adj[prev].append(nxt)
            indegree[nxt] += 1
        queue = deque()
        for i, n in enumerate(indegree):
            if n == 0:
                queue.append((i, 1))
        ans = 1
        while queue:
            cur, count = queue.popleft()
            ans = max(count, ans)
            for nei in adj[cur]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    queue.append((nei, count + 1))
        if sum(indegree) == 0:
            return ans
        return -1

    def maximumProduct(self, nums: List[int]) -> int:
        nums.sort()
        l1 = nums[0]
        l2 = nums[1]
        r1 = nums[-1]
        r2 = nums[-2]
        r3 = nums[-3]

        ans = max(r1 * r2 * r3, l1 * l2 * r1)
        return ans

    def topKFrequent(self, words: List[str], kk: int) -> List[str]:

        table = defaultdict(int)
        for w in words:
            table[w] += 1

        freq_pairs = [(freq, word) for word, freq in table.items()]
        freq_pairs.sort(key=lambda x: (-x[0], x[1]))

        ans = []
        i = 0
        while i < len(freq_pairs):
            current_freq = freq_pairs[i][0]
            group = [freq_pairs[i][1]]
            j = i + 1
            while j < len(freq_pairs) and freq_pairs[j][0] == current_freq:
                group.append(freq_pairs[j][1])
                j += 1

            group.sort()

            ans.extend(group)
            i = j

        return ans[:kk]

    def isPowerOfTwo(self, n: int) -> bool:
        if n == 1:
            return True
        if n <= 0:
            return False
        if n % 2 == 1:
            return False
        return self.isPowerOfTwo(n / 2)

    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        N = len(nums)
        nset = set([i for i in range(1, N + 1)])
        nset.difference_update(nums)
        return list(nset)

    def subtractProductAndSum(self, n: int) -> int:
        prod = 1
        total = 0
        while n > 0:
            rem = n % 10
            n //= 10
            prod *= rem
            total += rem
        return prod - total

    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        memo = {}
        N = len(values)

        def dp(i, j):
            if j in memo:
                return memo[j]
            if j >= N or i >= N:
                return float("-inf")
            max_score = float("-inf")
            if i < j:
                max_score = values[i] + values[j] + i - j
            score_a = dp(i, j + 1)
            score_b = dp(i + 1, i + 2)
            memo[j] = max(max_score, score_a, score_b)
            return memo[j]

        return dp(0, 1)

    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        top = -inf
        best_1 = values[0]
        for i in range(1, len(values)):
            top = max(top, best_1 + values[i] - i)
            best_1 = max(best_1, values[i] + i)
        return top

    def earliestAcq(self, logs: List[List[int]], n: int) -> int:
        """ "
        [time, x, y]
        """
        parent = {}
        logs.sort(key=lambda x: x[0])

        def find(x):
            if x not in parent:
                parent[x] = x
            if x != parent[x]:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rootx = find(x)
            rooty = find(y)
            if rootx != rooty:
                parent[rootx] = parent[rooty]

        def is_connected(x, y):
            return find(x) == find(y)

        for time, x, y in logs:
            if not is_connected(x, y):
                union(x, y)
                n -= 1
                if n == 1:
                    return time
        return -1

    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(heights)
        COL = len(heights[0])
        distances = defaultdict(lambda: float("inf"))
        heap = [(0, 0, 0)]
        distances[(0, 0)] = 0
        while heap:
            dist, r, c = heapq.heappop(heap)
            if (r, c) == (ROW - 1, COL - 1):
                return dist
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL:
                    ndist = max(abs(heights[nr][nc] - heights[r][c]), dist)
                    if distances[(nr, nc)] > ndist:
                        distances[(nr, nc)] = ndist
                        heapq.heappush(heap, (ndist, nr, nc))

    def isValid(self, s: str) -> bool:
        table = {"}": "{", ")": "(", "]": "["}
        stack = []
        for w in s:
            if not w in table:
                stack.append(w)
            else:
                if not stack:
                    return False
                prev = stack[-1]
                if table[w] != prev:
                    return False
                stack.pop()
        if stack:
            return False
        return True

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        table = {}
        groups = defaultdict(dict)

        for w in strs:
            table[w] = Counter(w)

        for i, n in enumerate(strs):
            found = False
            size = len(n)
            for key, v in groups[size].items():
                if table[key] == table[n]:
                    v.append(n)
                    found = True
                    break
            if not found:
                groups[size][n] = [n]

        ans = []
        small = min(groups)
        big = max(groups)
        for i in range(small, big + 1):
            if i in groups:
                for v in groups[i].values():
                    ans.append(v)
        return ans

    def pivotIndex(self, nums: List[int]) -> int:
        prefix = [nums[0]]

        for n in nums[1:]:
            total = prefix[-1] + n
            prefix.append(total)

        last = prefix[-1]
        for i, n in enumerate(prefix):
            if n - nums[i] == last - n:
                return i
        return -1

    def majorityElement(self, nums: List[int]) -> int:
        c1 = None
        c2 = None
        v1 = 0
        v2 = 0
        for n in nums:
            if n == c1:
                v1 += 1
            elif n == c2:
                v2 += 1
            elif v1 == 0:
                c1 = n
                v1 = 0
            elif v2 == 0:
                c2 = n
                v2 = 0
            else:
                v1 -= 1
                v2 -= 1
        if nums.count(c1) > len(nums) // 2:
            return c1
        return c2

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        N = len(nums)
        low = 0
        high = N

    def searchRange(self, nums: List[int], target: int) -> List[int]:

        left = bisect.bisect_left(nums, target)
        right = bisect.bisect_right(nums, target)
        if left < len(nums) and nums[left] == target:
            return [left, right - 1]
        return [-1, -1]

    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        self.ans = -inf

        def dfs(node, small, big):
            if not node:
                return inf, -inf
            small = min(node.val, small)
            big = max(node.val, big)

            lsmall, lbig = dfs(node.left, small, big)
            if lsmall != inf:
                self.ans = max(abs(big - lsmall), self.ans)
            if lbig != -inf:
                self.ans = max(abs(small - lbig), self.ans)

            rsmall, rbig = dfs(node.right, small, big)
            if rsmall != inf:
                self.ans = max(abs(big - rsmall), self.ans)
            if rbig != -inf:
                self.ans = max(abs(small - rbig), self.ans)

            small = min(small, lsmall, rsmall)
            big = max(big, lbig, rbig)

            return small, big

        dfs(root, root.val, root.val)
        return self.ans

    def smallestSubsequence(self, s: str) -> str:
        table = defaultdict(list)
        N = len(s)
        for i, n in enumerate(s):
            table[n].append(i)
        target = len(table)

        self.found = None

        def backtrack(arr, used, pos):
            if len(arr) == target:
                self.found = "".join(arr)
            if pos > N or self.found:
                return
            for i in range(26):
                if i not in used:
                    ch = chr(i + ord("a"))
                    if ch in table:
                        arr.append(ch)
                        used.add(i)
                        for j in table[ch]:
                            if j >= pos:
                                backtrack(arr, used, j)
                        used.remove(i)
                        arr.pop()

        backtrack([], set(), 0)
        return self.found

    def smallestSubsequence(self, s: str) -> str:
        stack = []
        used = set()
        counts = Counter(s)
        for w in s:
            counts[w] -= 1
            if w in used:
                continue
            used.add(w)
            while stack and stack[-1] > w and counts[stack[-1]] > 0:
                out = stack.pop()
                used.remove(out)
            stack.append(w)
        return "".join(stack)

    def prevPermOpt1(self, arr: List[int]) -> List[int]:
        left = 0
        N = len(arr)
        first = arr[0]
        for i in range(N - 1):
            cur = arr[i]
            top_i = i + 1
            top = arr[top_i]
            for j in range(i + 1, N):
                if arr[j] > top:
                    top_i = j
                    top = arr[j]
            if top > cur:
                arr[top_i], arr[i] = arr[i], arr[top_i]
                return arr
        return arr

    def prevPermOpt1(self, arr: List[int]) -> List[int]:
        n = len(arr)
        # Step 1: find the first index i from the right where arr[i] > arr[i+1]
        i = n - 2
        while i >= 0 and arr[i] <= arr[i + 1]:
            i -= 1
        if i < 0:
            return arr  # Already in smallest permutation form

        # Step 2: from the right, find j where arr[j] < arr[i]
        j = n - 1
        while j > i and arr[j] >= arr[i]:
            j -= 1

        # If there are duplicates equal to arr[j], move j left to the first occurrence
        while j > 0 and arr[j - 1] == arr[j]:
            j -= 1

        # Step 3: swap arr[i] and arr[j]
        arr[i], arr[j] = arr[j], arr[i]

        return arr

    def anagramMappings(self, nums1: List[int], nums2: List[int]) -> List[int]:
        table = defaultdict(list)
        for i, n in enumerate(nums2):
            table[n].append(i)
        ans = []
        for n in nums1:
            cur = table[n].pop()
            ans.append(cur)
        return ans

    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        queue = deque([root])
        while queue:
            size = len(queue)
            subt = 0
            for _ in range(size):
                cur = queue.popleft()
                subt += cur.val
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            if not queue:
                return subt

    def checkRecord(self, s: str) -> bool:
        acnt = 0
        cons = 0
        for w in s:
            if w == "A":
                acnt += 1
                if acnt == 2:
                    return False
                cons = 0
            elif w == "L":
                cons += 1
                if cons >= 3:
                    return False
            else:
                cons = 0
        return True

    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        """ "
        src, dst, time
        k = start
        """
        adj = defaultdict(list)
        for s, d, t in times:
            adj[s].append((d, t))
        heap = [(0, k)]
        visited = set()
        while heap:
            time, cur = heapq.heappop(heap)
            visited.add(cur)
            if len(visited) == n:
                return time
            for nei, w in adj[cur]:
                if nei not in visited:
                    ntime = time + w
                    heapq.heappush(heap, (ntime, nei))
        return -1

    def validMountainArray(self, arr: List[int]) -> bool:
        if len(arr) < 3:
            return False
        prev = arr[0]
        changed = False
        incr = False
        for n in arr[1:]:
            if n > prev:
                incr = True
                if changed:
                    return False
            elif n < prev:
                if not changed:
                    changed = True
            else:
                return False
            prev = n
        if not incr:
            return False
        return changed

    def minAreaRect(self, points: List[List[int]]) -> int:
        """ "
        x1, y2  |  x2, y2
        x1, y1  |  x2, y1
        """
        xadj = defaultdict(list)
        for x, y in points:
            xadj[x].append(y)
        ans = inf
        table = {}

        for x in sorted(xadj):
            ys = xadj[x]
            ys.sort()
            for i in range(len(ys)):
                y1 = ys[i]
                for j in range(i + 1, len(ys)):
                    y2 = ys[j]
                    if (y1, y2) in table:
                        width = x - table[(y1, y2)]
                        height = y2 - y1
                        area = width * height
                        ans = min(area, ans)
                    table[(y1, y2)] = x
        if ans != inf:
            return ans
        return 0

    def firstUniqChar(self, s: str) -> int:
        counts = Counter(s)
        for i, n in enumerate(s):
            if counts[n] == 1:
                return i
        return -1

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """ "
        1 -> 2 -> 3
        1 <- 2
        """
        if not head:
            return head
        cur = head
        prev = None
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        return prev

    def search(self, nums: List[int], target: int) -> int:
        N = len(nums)
        pivot = 0
        for i in range(N - 1):
            if nums[i] > nums[i + 1]:
                pivot = i
                break

        low = 0
        high = N - 1
        while low <= high:
            mid = (low + high) // 2
            idx = (mid + pivot) % N
            if nums[idx] > target:
                high = mid - 1
            elif nums[idx] < target:
                low = mid + 1
            else:
                return True
        return False

    def isCousins(self, root: Optional[TreeNode], x: int, y: int) -> bool:
        queue = deque([(root, 0)])
        found = {}
        while queue:
            size = len(queue)
            for _ in range(size):
                cur, depth = queue.popleft()
                if cur.left:
                    queue.append((cur.left, depth + 1))
                    if cur.left.val == x or cur.left.val == y:
                        found[cur.left.val] = (cur, depth + 1)
                if cur.right:
                    queue.append((cur.right, depth + 1))
                    if cur.right.val == x or cur.right.val == y:
                        found[cur.right.val] = (cur, depth + 1)
            if len(found) == 2:
                break
        if len(found) < 2:
            return False
        if found[x][1] == found[y][1] and found[x][0] != found[y][0]:
            return True
        return False

    def minPathSum(self, grid: List[List[int]]) -> int:
        heap = [(grid[0][0], 0, 0)]
        directions = [(1, 0), (0, 1)]
        distances = defaultdict(lambda: inf)
        ROW = len(grid)
        COL = len(grid[0])
        while heap:
            total, r, c = heapq.heappop(heap)
            if r == ROW - 1 and c == COL - 1:
                return total
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL:
                    nt = total + grid[nr][nc]
                    if distances[(nr, nc)] > nt:
                        heapq.heappush(heap, (nt, nr, nc))
                        distances[(nr, nc)] = nt

        def minPathSum(self, grid: List[List[int]]) -> int:
            directions = [(1, 0), (0, 1)]
            distances = defaultdict(lambda: inf)
            ROW = len(grid)
            COL = len(grid[0])
            queue = deque([(grid[0][0], 0, 0)])
            ans = inf
            while queue:
                total, r, c = queue.popleft()
                if r == ROW - 1 and c == COL - 1:
                    ans = min(ans, total)
                for dr, dc in directions:
                    nr = dr + r
                    nc = dc + c
                    if 0 <= nr < ROW and 0 <= nc < COL:
                        nt = total + grid[nr][nc]
                        if distances[(nr, nc)] > nt:
                            queue.append((nt, nr, nc))
                            distances[(nr, nc)] = nt
            return ans


class MyCalendarTwo:
    def __init__(self):
        self.booked = []
        self.overlaps = []

    def book(self, start: int, end: int) -> bool:

        for s2, e2 in self.overlaps:
            if not (end <= s2 or start >= e2):
                return False

        for s1, e1 in self.booked:
            if not (end <= s1 or start >= e1):
                self.overlaps.append((max(s1, start), min(e1, end)))

        self.booked.append((start, end))
        return True


class Solution:

    def __init__(self, head: Optional[ListNode]):
        self.head = head
        self.arr = []
        cur = head
        while cur:
            self.arr.append(cur)
            cur = cur.next

    def getRandom(self) -> int:
        size = len(self.arr)
        randidx = random.randint(0, size)
        return self.arr[randidx]


class MinStack:

    def __init__(self):
        self.stack = []
        self.heap = []
        self.table = {}
        self.idx = 0

    def push(self, val: int) -> None:
        self.idx += 1
        heapq.heappush(self.heap, (val, self.idx))
        self.stack.append((val, self.idx))
        self.table[self.idx] = val

    def pop(self) -> None:
        val, idx = self.stack.pop()
        self.table.pop(idx, None)

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        while self.heap and self.heap[0][1] not in self.table:
            heapq.heappop(self.heap)
        return self.heap[0][0]


class RobotRoomCleaner:

    def __init__(self):
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.visited = set()

    def turnBack(self, robot):
        robot.turnRight()
        robot.turnRight()
        robot.move()
        robot.turnRight()
        robot.turnRight()

    def cleanRoom(self, robot):
        """
        :type robot: Robot
        :rtype: None
        """

        def backtrack(robot, r, c, d):
            robot.clean()
            for i in range(4):
                nd = (i + d) % 4
                nr = self.directions[nd] + r
                nc = self.directions[nd] + c
                if (nr, nc) not in self.visited and robot.move():
                    self.visited.add((nr, nc))
                    backtrack(robot, nr, nc, nd)
                    self.turnBack(robot)

        backtrack(robot, 0, 0, 0)


class MedianFinder:

    def __init__(self):
        self.low = []
        self.high = []

    def addNum(self, num: int) -> None:
        heapq.heappush(self.low, -num)
        heapq.heappop(self.high, -heapq.heappop(self.low))
        if len(self.low) < len(self.high):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def findMedian(self) -> float:
        if len(self.low) > len(self.high):
            return -self.low[0]
        return (-self.low[0] + self.high[0]) / 2


class MedianFinder:

    def __init__(self):
        self.data = []

    def left_bound(self, n):
        N = len(self.data)
        low = 0
        high = N - 1
        while low <= high:
            mid = (low + high) // 2
            num = self.data[mid]
            if num < n:
                low = mid + 1
            else:
                high = mid - 1
        return low

    def addNum(self, num: int) -> None:
        idx = self.left_bound(num)
        self.data.insert(idx)

    def findMedian(self) -> float:
        size = len(self.data)
        if size % 2 == 1:
            return self.data[size // 2]
        return self.data[size // 2] + self.data[size // 2 - 1]


class Bank:

    def __init__(self, balance: List[int]):
        self.balance = balance
        self.size = len(balance)

    def transfer(self, account1: int, account2: int, money: int) -> bool:
        if not (0 < account1 <= self.size) or not (0 < account1 <= self.size):
            return False
        if self.balance[account1 - 1] < money:
            return False
        self.balance[account1 - 1] -= money
        self.balance[account2 - 1] += money
        return True

    def deposit(self, account: int, money: int) -> bool:
        if not (0 < account <= self.size):
            return False
        self.balance[account - 1] += money
        return True

    def withdraw(self, account: int, money: int) -> bool:
        if not (0 < account <= self.size) or self.balance[account - 1] < money:
            return False
        self.balance[account - 1] -= money
        return True


def test_solution():
    s = Solution()
    m = [[0, 0, 0, 5], [4, 3, 1, 4], [0, 1, 1, 4], [1, 2, 1, 3], [0, 0, 1, 1]]
    s.setZeroes(m)


if __name__ == "__main__":
    test_solution()
