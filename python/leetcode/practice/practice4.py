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

    def shortestDistanceAfterQueries(
        self, n: int, queries: List[List[int]]
    ) -> List[int]:

        ans = []
        graph = defaultdict(list)
        for i in range(n - 1):
            graph[i].append(i + 1)

        for src, dst in queries:
            queue = deque([(0, 0)])
            graph[src].append(dst)
            visited = set()
            while queue:
                src, steps = queue.popleft()
                if src == n - 1:
                    ans.append(steps)
                    break
                for nei in graph[src]:
                    if nei not in visited:
                        queue.append((nei, steps + 1))
                    visited.add(nei)
        return ans

    def findKthPositive(self, arr: List[int], k: int) -> int:
        """_summary_
        1 2   4 6 7
        0 1 2 3 4 5 6
        """
        low = 0
        high = len(arr) - 1
        while low <= high:
            mid = (high + low) // 2
            cur = arr[mid]
            diff = cur - mid - 1
            if diff < k:
                low = mid + 1
            else:
                high = mid - 1
        ans = -1
        if 0 < low < len(arr):
            diff = low - arr[low] + k
            return arr[low] + diff
        if high < 0:
            return k
        if low >= len(arr):
            diff = arr[-1] - len(arr)
            diff = k - diff
            return arr[-1] + diff
        return ans

    def numWays(self, steps: int, arrLen: int) -> int:
        choices = [0, 1, -1]
        memo = {}

        def recurs(pos, step):
            if (pos, step) in memo:
                return memo[(pos, step)]
            if step < 0 or pos >= arrLen or pos - step > 0 or pos < 0:
                memo[(pos, step)] = 0
                return 0
            if step == 0 and pos == 0:
                memo[(pos, step)] = 1
                return 1
            count = 0
            for i in range(len(choices)):
                count += recurs(pos + choices[i], step - 1)
            memo[(pos, step)] = count
            return count

        return recurs(0, steps) % (10**9 + 7)

    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        """ "
        - M: unrevealed mine
        - E: unrevelaed empty square
        - B: revealed blank with no adj mines
        - X: revealed mine
        """
        directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
        ]
        ROW = len(board)
        COL = len(board[0])

        def count_m(r, c):
            count = 0
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL:
                    if board[nr][nc] in ["M", "X"]:
                        count += 1
            return count

        def dfs(r, c):
            if not (0 <= r < ROW and 0 <= c < COL):
                return
            cur = board[r][c]
            if cur == "E":
                count = count_m(r, c)
                if count == 0:
                    board[r][c] = "B"
                    for dr, dc in directions:
                        nr = dr + r
                        nc = dc + c
                        dfs(nr, nc)
                else:
                    board[r][c] = str(count)

        r, c = click
        cur = board[r][c]
        if cur == "E":
            dfs(r, c)
        elif cur == "M":
            board[r][c] = "X"
        return board

    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        N = len(nums)
        for i in range(N - 1, -1, -1):
            n = nums[i]
            if n == 0:
                j = i + 1
                while j < N and nums[j] != 0:
                    nums[j], nums[j - 1] = nums[j - 1], nums[j]
                    j += 1

    def subarraySum(self, nums: List[int], k: int) -> int:

        table = defaultdict(int)
        table[0] += 1
        ans = 0
        total = 0
        for n in nums:
            total += n
            if total - k in table:
                ans += table[total - k]
            table[total] += 1
        return ans

    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = deque([(root, 0)])
        table = defaultdict(list)
        while queue:
            cur, level = queue.popleft()
            table[level].append(cur.val)
            if cur.left:
                queue.append((cur.left, level - 1))
            if cur.right:
                queue.append((cur.right, level + 1))
        ans = []
        min_level = min(table.keys())
        mx_level = max(table.keys())
        for i in range(min_level, mx_level + 1):
            ans.append(table[i])
        return ans

    def sortedSquares(self, nums: List[int]) -> List[int]:
        negs = []
        pos = []
        for i, n in enumerate(nums):
            if n < 0:
                negs.append(n**2)
            else:
                pos.append(n**2)
        negs.reverse()
        l = 0
        r = 0
        j = 0
        while l < len(negs) and r < len(pos):
            if negs[l] > pos[r]:
                nums[j] = pos[r]
                r += 1
            else:
                nums[j] = negs[l]
                l += 1
            j += 1
        while l < len(negs):
            nums[j] = negs[l]
            l += 1
            j += 1
        while r < len(pos):
            nums[j] = pos[r]
            r += 1
            j += 1
        return nums

    def customSortString(self, order: str, s: str) -> str:
        table = defaultdict(int)
        for i, w in enumerate(order):
            table[w] = i
        ans = []
        buckets = [0] * len(order)
        for w in s:
            if w in table:
                idx = table[w]
                buckets[idx] += 1
            else:
                ans.append(w)
        for i, v in enumerate(buckets):
            if v > 0:
                ans.append(s[i] * v)
        return "".join(ans)

    def hammingWeight(self, n: int) -> int:
        count = 0
        while n > 0:
            rem = n % 2
            count += rem
            n = n // 2
        return count

    def minimumTime(self, grid: List[List[int]]) -> int:
        if grid[0][1] > 1 and grid[1][0] > 1:
            return -1
        ROW = len(grid)
        COL = len(grid[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        heap = [(0, 0, 0)]
        visited = set()
        while heap:
            time, r, c = heapq.heappop(heap)
            if r == ROW - 1 and c == COL - 1:
                return time
            if (r, c) in visited:
                continue
            visited.add((r, c))
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL and (nr, nc) not in visited:
                    wait_time = 0
                    if (grid[nr][nc] - time) % 2 == 0:
                        wait_time = 1
                    new_time = max(grid[nr][nc] + wait_time, time + 1)
                    heapq.heappush(heap, (new_time, nr, nc))

        return -1

    def findSpecialInteger(self, arr: List[int]) -> int:
        candidates = [None, None, None]
        votes = [0, 0, 0]
        for n in arr:
            if n == candidates[0]:
                votes[0] += 1
            elif n == candidates[1]:
                votes[1] += 1
            elif n == candidates[2]:
                votes[2] += 1
            elif votes[0] == 0:
                candidates[0] = n
            elif votes[1] == 0:
                candidates[1] = n
            elif votes[2] == 0:
                candidates[2] = n
            else:
                votes[0] -= 1
                votes[1] -= 1
                votes[2] -= 1
        limit = len(arr) // 4
        for c in candidates:
            if arr.count(c) > limit:
                return c
        return arr[0]

    def getAllElements(
        self, root1: Optional[TreeNode], root2: Optional[TreeNode]
    ) -> List[int]:

        def dfs(node):
            if not node:
                return []
            larr = dfs(node.left)
            cur = [node.val]
            rarr = dfs(node.right)
            return larr + cur + rarr

        arr1 = dfs(root1)
        arr2 = dfs(root2)
        arr = []

        i = 0
        j = 0
        while i < len(arr1) and j < len(arr2):
            if arr1[i] < arr2[j]:
                arr.append(arr1[i])
                i += 1
            else:
                arr.append(arr2[j])
                j += 1
        if i < len(arr1):
            arr.extend(arr1[i:])
        if j < len(arr2):
            arr.extend(arr2[j:])
        return arr

    def largestIsland(self, grid: List[List[int]]) -> int:
        """_summary_
        0 1 0 1
        1 0 1 0
        0 1 0 0
        """
        scores = defaultdict(int)
        edges = defaultdict(list)
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(grid)
        COL = len(grid[0])

        def dfs(r, c, root):
            cur = grid[r][c]
            if cur == 0:
                edges[root].append((r, c))
            if cur != 1:
                return 0
            size = 1
            grid[r][c] = root
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < ROW and 0 <= nc < COL:
                    size += dfs(nr, nc, root)
            return size

        for r in range(ROW):
            for c in range(COL):
                if grid[r][c] == 1:
                    size = dfs(r, c, (r, c))
                    scores[(r, c)] = size

        ans = 1
        top = 0

        for root, size in scores.items():
            temptop = size
            if root in edges:
                temptop += 1
                borders = edges[root]
                for r, c in borders:
                    stuff = size
                    seen = set()
                    for dr, dc in directions:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < ROW and 0 <= nc < COL and grid[nr][nc] != 0:
                            rooty = grid[nr][nc]
                            if rooty == root:
                                continue
                            if (root, rooty) not in seen:
                                seen.add((root, rooty))
                                seen.add((rooty, root))
                                stuff += scores[rooty]
                    ans = max(stuff + 1, ans)
            top = max(temptop, top)
        return max(ans, top)

    def sumSubarrayMins(self, arr: List[int]) -> int:
        ans = [n for n in arr]
        N = len(arr)
        for size in range(2, len(arr)):
            queue = deque([(n, i) for i, n in enumerate(arr[: size - 1])])
            left = 0
            for right in range(size - 1, N):
                cur = arr[right]
                while queue and queue[-1][0] >= cur:
                    queue.pop()

                diff = right - left

    def minStickers(self, stickers: List[str], target: str) -> int:
        def find_best(leftover):
            winners = defaultdict(int)
            new_target = defaultdict(int)
            for w, c in leftover.items():
                candidates = diction[w]
                for can in candidates:
                    winners[can] += min(c, table[can][w])
            if not winners:
                return -1
            best, z = max(winners.items(), key=lambda x: x[1])
            for w, c in leftover.items():
                left = max(c - table[best][w], 0)
                if left > 0:
                    new_target[w] = left
            return new_target

        def rid_bottle_necks(leftover, bottle_necks):
            winners = defaultdict(int)
            new_target = defaultdict(int)
            b = set([diction[w] for w in bottle_necks])
            for w, c in leftover.items():
                candidates = b.intersection(diction[w])
                for can in candidates:
                    winners[can] += min(c, table[can][w])
            if not winners:
                return -1
            best, z = max(winners.items(), key=lambda x: x[1])
            for w, c in leftover.items():
                left = max(c - table[best][w], 0)
                if left > 0:
                    new_target[w] = left
            return new_target

        wset = set(target)
        table = {}
        diction = defaultdict(set)
        ctarget = Counter(target)
        for s in stickers:
            counter = defaultdict(int)
            for w in s:
                if w in ctarget:
                    diction[w].add(s)
                    counter[w] += 1
            table[s] = counter
        ans = 0
        bottle_necks = []
        for k, v in diction.items():
            if len(v) <= 2:
                bottle_necks.add(k)
        while len(ctarget) > 0:
            ctarget = find_best(ctarget, bottle_necks)
            if ctarget == -1:
                return -1
            ans += 1
        return ans

    def isMajorityElement(self, nums: List[int], target: int) -> bool:
        if nums.count(target) > (len(nums) // 2):
            return True
        return False

    def minAddToMakeValid(self, s: str) -> int:
        left = 0
        right = 0
        for p in s:
            if p == "(":
                left += 1
            else:
                if left < 1:
                    right += 1
                else:
                    left -= 1
        return left + right

    def numSimilarGroups(self, strs: List[str]) -> int:
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

        def is_similar(s1, s2):
            diff = 0
            for i in range(len(s1)):
                if s2[i] != s1[i]:
                    diff += 1
                    if diff > 2:
                        return False
            return True

        N = len(strs)
        for i in range(N - 1):
            s1 = strs[i]
            rootx = find(s1)
            for j in range(i + 1, N):
                s2 = strs[j]
                rooty = find(s2)
                if rootx == rooty:
                    continue
                if is_similar(s1, s2):
                    union(s1, s2)

        groups = set()
        for s in strs:
            root = find(s)
            groups.add(root)
        return len(groups)

    def findLengthOfLCIS(self, nums: List[int]) -> int:
        ans = 1
        count = 1
        for i in range(len(nums) - 1):
            left = nums[i]
            right = nums[i + 1]
            if left < right:
                count += 1
                ans = max(count, ans)
            else:
                count = 1
        return ans

    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        """_summary_
        1 2 3 4 5 6
        [1,2,3,8,9,10]
        1 1 5 1 1
        """

        def count_arth(num):
            ans = 0
            diff = num - 3
            while diff >= 0:
                ans += 1
                ans += diff
                diff -= 1
            return ans

        if len(nums) < 3:
            return 0
        N = len(nums)
        diffs = []
        for i in range(N - 1):
            diffs.append(nums[i] - nums[i + 1])

        count = 1
        cur = diffs[0]
        ans = 0
        diffs.append(5000)
        for n in diffs[1:]:
            if n == cur:
                count += 1
            else:
                if count + 1 >= 3:
                    ans += count_arth(count + 1)
                count = 1
                cur = n
        return ans


class Robot:
    def move(self):
        """
        Returns true if the cell in front is open and robot moves into the cell.
        Returns false if the cell in front is blocked and robot stays in the current cell.
        :rtype bool
        """

    def turnLeft(self):
        """
        Robot will stay in the same cell after calling turnLeft/turnRight.
        Each turn will be 90 degrees.
        :rtype void
        """

    def turnRight(self):
        """
        Robot will stay in the same cell after calling turnLeft/turnRight.
        Each turn will be 90 degrees.
        :rtype void
        """

    def clean(self):
        """
        Clean the current cell.
        :rtype void
        """


class Solution:
    def __init__(self):
        self.grid = {0: {0: 1}}
        self.curx = 0
        self.cury = 0
        self.ylen = {0}
        self.xlen = {0}
        self.cur_dir = 0
        self.cleaned = set()
        self.obs = set()
        self.map = {
            0: (1, 0),
            90: (0, 1),
            180: (-1, 0),
            -90: (0, -1),
        }

    def cleanRoom(self, robot):
        """
        :type robot: Robot
        :rtype: None
        """
        self.robot = robot
        while (len(self.cleaned) + len(self.obs)) <= (len(self.xlen) * len(self.ylen)):
            robot.clean()
            for degree, coord in self.map.items():
                nx = self.curx + coord[0]
                ny = self.cury + coord[1]
                new_coord = (nx, ny)
                outcome = True
                while outcome:
                    if new_coord not in self.cleaned and new_coord not in self.obs:
                        robot.clean()
                        outcome = self.move(degree)
                    else:
                        outcome = False

    def move(self, degree):
        # till 0
        pos = self.cur_dir
        while pos > degree:
            self.robot.turnRight()
            pos -= 90
        while pos < degree:
            self.robot.turnLeft()
            pos += 90
        self.cur_dir = pos
        outcome = self.robot.move()
        if outcome:
            nx, ny = self.map[self.cur_dir]
            self.curx += nx
            self.cury += ny
            self.xlen.add(self.curx)
            self.ylen.add(self.cury)
            if (self.curx, self.cury) not in self.cleaned:
                self.robot.clean()
                self.cleaned.add((self.curx, self.cury))
        else:
            self.obs.add((self.curx + 1, self.cury))
        return outcome


def test_solution():
    s = Solution()
    print(-4 % 2)


if __name__ == "__main__":

    test_solution()
