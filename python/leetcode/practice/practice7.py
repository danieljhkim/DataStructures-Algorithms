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
            aa = find(a)
            bb = find(b)
            ans[i] = max(aa, bb)
            i += 1
        return ans


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


def test_solution():
    s = Solution()
    m = [[0, 0, 0, 5], [4, 3, 1, 4], [0, 1, 1, 4], [1, 2, 1, 3], [0, 0, 1, 1]]
    s.setZeroes(m)


if __name__ == "__main__":
    test_solution()
