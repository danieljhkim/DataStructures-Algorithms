from curses.ascii import isdigit
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

    def maxTwoEvents(self, events: List[List[int]]) -> int:
        events.sort(key=lambda x: (x[0], x[1]))
        heap = []
        max_val = -1
        temp_val = 0
        for s, e, val in events:
            while heap and heap[0][0] < s:
                out = heapq.heappop(heap)
                temp_val = max(temp_val, out[1])
            max_val = max(temp_val + val, max_val)
            heapq.heappush(heap, (e, val))
        return max_val

    def maximumLength(self, s: str) -> int:
        """ "
        a a a - 3:1, 2:2, 1:3

        a a a a - 4:1, 3:2, 2:3, 1:4

        a a a a a
        """
        best = -1

        def log(letter, length):
            nonlocal best
            size = length
            for i in range(0, length):
                table[letter][size] += i + 1
                if table[letter][size] >= 3 and size > best:
                    best = size
                size -= 1

        left = 0
        right = 0
        N = len(s)
        table = defaultdict(lambda: defaultdict(int))
        while right < N:
            cur = s[left]
            while right < N and cur == s[right]:
                right += 1
            length = right - left
            log(cur, length)
            left = right

        return best

    def numFriendRequests(self, ages: List[int]) -> int:
        ages.sort()
        N = len(ages)

        def bsearch(idx):
            low = 0
            high = idx
            age = ages[idx]
            target = age // 2 + 7
            while low <= high:
                mid = (low + high) // 2
                if ages[mid] <= target:
                    low = mid + 1
                else:
                    high = mid - 1

            if ages[low] <= age and low != idx:
                return low
            return -1

        ans = 0
        i = N - 1
        while i > 0:
            dup = 1
            cur = ages[i]
            while i > 0 and ages[i - 1] == cur:
                i -= 1
                dup += 1
            res = bsearch(i)
            if res >= 0:
                ans += (i - res) * dup
            if dup > 1:
                ans += dup * (dup - 1) // 2
            i -= 1
        return ans

    def getLonelyNodes(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        self.ans = []

        def dfs(node, only):
            if only:
                self.ans.append(node.val)
            isalone = True
            if node.left and node.right:
                isalone = False
            if node.left:
                dfs(node.left, isalone)
            if node.right:
                dfs(node.right, isalone)

        dfs(root, False)
        return self.ans

    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()
        ans = 0
        left = 0
        for right in range(len(nums)):
            while nums[right] - nums[left] > 2 * k:
                left += 1
            ans = max(ans, right - left + 1)
        return ans

    def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        """ "
        sub4u4 ("sub stit u tion")
        """
        idx = 0
        widx = 0
        N = len(abbr)
        WN = len(word)
        while idx < N and widx < WN:
            w = abbr[idx]
            if w.isdigit():
                if w == "0":
                    return False
                num = w
                while idx < N - 1 and abbr[idx + 1].isdigit():
                    idx += 1
                    num += abbr[idx]
                widx += int(num) - 1
                idx += 1
            else:
                if widx >= WN or w != word[widx]:
                    return False
                widx += 1
                idx += 1

        if widx != WN or idx != N:
            return False
        return True

    def uniquePaths(self, m: int, n: int) -> int:
        cache = {}

        def dp(r, c):
            if r == m - 1 and c == n - 1:
                return 1
            if (r, c) in cache:
                return cache[(r, c)]
            count = 0
            nr = r + 1
            nc = c + 1
            if 0 <= nr < m:
                count += dp(nr, c)
            if 0 <= nc < n:
                count += dp(r, nc)
            cache[(r, c)] = count
            return count

        return dp(0, 0)

    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        def findMid(node):
            if node and not node.next:
                return node, None
            slow = node
            fast = node
            prev = slow
            while fast and fast.next:
                fast = fast.next.next
                prev = slow
                slow = slow.next
            right = None
            mid = slow
            if slow:
                right = slow.next
                slow.next = None
                prev.next = None
            return mid, right

        def recursion(node):
            if node and not node.next:
                return TreeNode(node.val)
            if not node:
                return None
            mid, right = findMid(node)
            tnode = TreeNode(mid.val)
            if node:
                tnode.left = recursion(node)
            if right:
                tnode.right = recursion(right)
            return tnode

        return recursion(head)

    def numIslands(self, grid: List[List[str]]) -> int:
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(grid)
        COL = len(grid[0])

        def dfs(r, c):
            cur = grid[r][c]
            if cur == "0":
                return
            if cur == "1":
                grid[r][c] = "0"
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL:
                    dfs(nr, nc)

        ans = 0
        for r in range(ROW):
            for c in range(COL):
                if grid[r][c] == "1":
                    dfs(r, c)
                    ans += 1

        return ans

    def removeInvalidParentheses(self, s: str) -> List[str]:

        def check(arr):
            left = 0
            right = 0
            for i, w in enumerate(arr):
                if w == ")":
                    if left > 0:
                        left -= 1
                    else:
                        right += 1
                elif w == "(":
                    left += 1
            return left, right

        left, right = check(list(s))
        N = len(s)
        ans = set()
        remove = left + right
        NN = N - remove
        if remove == 0:
            return [s]

        def back(l_rm, idx, arr):
            if len(arr) == NN and l_rm <= 0:
                l, r = check(arr)
                if l + r == 0:
                    ans.add("".join(arr))
                return

            left = l_rm
            for i in range(idx, N):
                w = s[i]
                if w == ")":
                    if left > 0:
                        arr.append(w)
                        back(left - 1, i + 1, arr[:])
                        arr.pop()
                    continue
                elif w == "(":
                    arr.append(w)
                    left += 1
                else:
                    arr.append(w)

        back(left, 0, [])
        ans = list(ans)
        if len(ans) == 0:
            return [""]
        return ans

    def convert(self, s: str, numRows: int) -> str:
        table = {}
        for i in range(numRows):
            table[i] = []

        N = len(s)
        i = 0
        while i < N:
            idx = 1
            table[idx].append(s[i])
            while i < N - 1 and idx < numRows - 1:
                idx += 1
                i += 1
                table[idx].append(s[i])
            i += 1
            idx -= 1
            while i < N and idx > 0:
                table[idx].append(s[i])
                idx -= 1
                i += 1
        ans = []
        for i in range(numRows):
            ans.extend(table[i])
        return "".join(ans)

    def trap(self, height: List[int]) -> int:

        def check_monotonic(height):
            up = True
            down = True
            for i in range(1, len(height)):
                if height[i - 1] > height[i]:
                    up = False
                elif height[i - 1] < height[i]:
                    down = False
            return up or down

        if check_monotonic(height):
            return 0

        N = len(height)
        lcache = {}
        rcache = {}

        def left(idx, h):
            if (idx, h) in lcache:
                return lcache[(idx, h)]
            if idx < 0:
                return h
            h = max(height[idx], h)
            res = left(idx - 1, h)
            lcache[(idx, h)] = res
            return res

        def right(idx, h):
            if (idx, h) in rcache:
                return rcache[(idx, h)]
            if idx >= N:
                return h
            h = max(height[idx], h)
            res = right(idx + 1, h)
            rcache[(idx, h)] = res
            return res

        ans = 0
        for i in range(1, N):
            l = left(i - 1, height[i])
            r = right(i + 1, height[i])
            ans += min(l, r) - height[i]
        return ans

    def pickGifts(self, gifts: List[int], k: int) -> int:
        heap = []
        total = sum(gifts)
        for g in gifts:
            heapq.heappush(heap, -g)
        ans = 0
        for i in range(k):
            out = heapq.heappop(heap)
            stay = int((-out) ** 0.5)
            ans += -out - stay
            heapq.heappush(heap, -stay)
        return total - ans

    def findCelebrity(self, n: int) -> int:
        def knows(a, b):
            pass

        memo = {}

        def knows_wrapper(a, b):
            if (a, b) in memo:
                return memo[(a, b)]
            res = knows(a, b)
            memo[(a, b)] = res
            return res

        def dfs(a, b):
            if a >= n:
                return True
            if a == b or (knows_wrapper(a, b) and not knows_wrapper(b, a)):
                return dfs(a + 1, b)
            else:
                return False

        candidate = 0
        for i in range(1, n):
            if knows_wrapper(candidate, i) and not knows_wrapper(i, candidate):
                candidate = i

        if dfs(0, candidate):
            return candidate
        return -1

    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        buy = prices[0]
        for p in prices[1:]:
            if p > buy:
                profit = max(p - buy, profit)
            else:
                buy = p
        return profit

    def maxProfit(self, k, prices: List[int]) -> int:
        memo = {}

        def recurs(idx, count):
            if count <= 0 or idx >= len(prices) - 1:
                return 0
            if (idx, count) in memo:
                return memo[(idx, count)]
            profit = 0
            buy = prices[idx]
            for i in range(idx + 1, len(prices)):
                p = prices[i]
                if p > buy:
                    prof = p - buy
                    future = recurs(i, count - 1)
                    profit = max(prof + future, profit)
                    if profit < self.max and count == 0:
                        break
                else:
                    buy = p
            self.max = max(self.max, profit)
            memo[(idx, count)] = profit
            return profit

        return recurs(0, k)

    def maxProfit(self, k: int, prices: List[int]) -> int:
        memo = {}

        def recurs(idx, holding, count):
            if idx == len(prices) or count == 0:
                return 0
            key = (idx, holding, count)
            if key in memo:
                return memo[key]

            # Option 1: Do nothing
            profit = recurs(idx + 1, holding, count)

            if holding:
                # Option 2: Sell stock
                sell = prices[idx] + recurs(idx + 1, 0, count - 1)
                profit = max(profit, sell)
            else:
                # Option 3: Buy stock
                buy = -prices[idx] + recurs(idx + 1, 1, count)
                profit = max(profit, buy)

            memo[key] = profit
            return profit

        return recurs(0, 0, k)

    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        words = []
        digits = []
        for l in logs:
            if l[-1].isdigit():
                digits.append(l)
            else:
                ll = l.split(" ")
                words.append((" ".join(ll[1:]), ll[0]))

        words.sort(key=lambda x: (x[0], x[1]))
        ans = []
        for c, k in words:
            ans.append(k + " " + c)
        ans.extend(digits)
        return ans

    def generateParenthesis(self, n: int) -> List[str]:
        ans = set()

        def recurs(left, right, word):
            if left == n and right == n:
                ans.add("".join(word))
                return
            if left < n:
                word.append("(")
                recurs(left + 1, right, word)
                word.pop()
            if right < left:
                word.append(")")
                recurs(left, right + 1, word)
                word.pop()

        recurs(0, 0, [])
        return list(ans)

    def findMaxAverage(self, nums: List[int], k: int) -> float:
        left = 0
        right = 0
        N = len(nums)
        total = 0
        ans = float("-inf")
        while right < N:
            while right < N and right - left < k:
                total += nums[right]
                right += 1
            ans = max(total, ans)
            total -= nums[left]
            left += 1
        return ans / k

    def countAndSay(self, n: int) -> str:

        def recurs(i, s):
            if i >= n:
                return s
            res = []
            left = 0
            right = 0
            N = len(s)
            while right < N:
                while right < N and s[right] == s[left]:
                    right += 1
                count = right - left
                res.append(str(count) + s[left])
                left = right
            return recurs(i + 1, "".join(res))

        return recurs(0, "1")

    def str2tree(self, s: str) -> Optional[TreeNode]:
        """ "
        4(2(3)(1))(6(5))
        """
        if not s:
            return None
        stack = []
        N = len(s)
        i = 0
        while i < N:
            if s[i] == "(" or s[i].isdigit() or s[i] == "-":
                n = 0
                if s[i] == "(":
                    i += 1
                sign = 1
                if s[i] == "-":
                    sign = -1
                    i += 1
                while i < N and s[i].isdigit():
                    n = n * 10 + int(s[i])
                    i += 1
                tn = TreeNode(n * sign)
                stack.append(tn)
            elif s[i] == ")":
                node = stack.pop()
                if stack[-1].left:
                    stack[-1].right = node
                else:
                    stack[-1].left = node
                i += 1
        return stack[0]

    def findScore(self, nums: List[int]) -> int:
        N = len(nums)
        marked = set()
        snums = [(nums[i], i) for i in range(N)]
        snums.sort(key=lambda x: (x[0], x[1]))
        i = 0
        score = 0
        while i < N:
            val, idx = snums[i]
            if idx in marked:
                i += 1
                continue
            score += val
            marked.add(idx)
            left = idx - 1
            right = idx + 1
            if left >= 0:
                marked.add(left)
            if right < N:
                marked.add(right)
            i += 1
        return score

    def numFriendRequests(self, ages: List[int]) -> int:
        """ "
        - y <= .5 * x + 7
        - y > x
        """

        def search(x):
            high = x
            low = 0
            target = ages[x] / 2 + 7
            while low <= high:
                mid = (low + high) // 2
                age = ages[mid]
                if age <= target:
                    low = mid + 1
                else:
                    high = mid - 1
            return low

        ages.sort()
        N = len(ages)
        i = N - 1
        ans = 0
        while i >= 0:
            cur = ages[i]
            right = i
            while i > 0 and ages[i - 1] == cur:
                i -= 1
            count = right - i + 1
            if i == 0:
                ans += count * (count - 1)
                return ans
            idx = search(i)
            if idx < i:
                if count > 1:
                    ans += count * (i - idx) + count * (count - 1)
                else:
                    ans += i - idx
            i -= 1
        return ans

    def numFriendRequests(self, ages: List[int]) -> int:
        counts = Counter(ages)
        ans = 0
        for agex, countx in counts.items():
            for agey, county in counts.items():
                if agey > agex:
                    continue
                if agey <= agex * 0.5 + 7:
                    continue
                ans += countx * county
                if agex == agey:
                    ans -= countx
        return ans

    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        self.total = 0

        def dfs(node, arr):
            arr.append(str(node.val))
            if not node.left and not node.right:
                num = int("".join(arr))
                self.total += num
            if node.left:
                dfs(node.left, arr)
            if node.right:
                dfs(node.right, arr)
            arr.pop()

        dfs(root, [])
        return self.total

    def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
        sum1 = Counter(target)
        sum2 = Counter(arr)
        if sum1 != sum2:
            return False
        return True

    def minRemoveToMakeValid(self, s: str) -> str:
        left = 0
        right = 0
        stack = []
        for w in s:
            if w == "(":
                stack.append(w)
                left += 1
            elif w == ")":
                if left > 0:
                    left -= 1
                    stack.append(w)
                else:
                    continue
            else:
                stack.append(w)
        ans = []
        while stack:
            w = stack.pop()
            if w == "(":
                if right > 0:
                    right -= 1
                    ans.append(w)
            elif w == ")":
                right += 1
                ans.append(w)
            else:
                ans.append(w)

        ans.reverse()
        return "".join(ans)

    def numOfWays(self, n: int) -> int:
        """ "
        1 2 3
        """
        grid = [[0] * 3 for _ in range(n)]
        grid[0][0] = 0
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        self.ans = 0

        def dfs(grid, r, c):
            if r == n:
                self.ans += 1
                return
            if c >= 3:
                dfs(grid, r + 1, 0)
                return
            cur = grid[r][c]
            if cur > 0:
                return
            for color in range(1, 4):
                for dr, dc in directions:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < n and 0 <= nc < 3:
                        ncolor = grid[nr][nc]
                        if ncolor == color:
                            continue
                grid[r][c] = color
                dfs(grid, r, c + 1)
            grid[r][c] = 0

        dfs(grid, 0, 0)
        return self.ans * 3


class WordDictionary:

    def __init__(self):
        self.table = {}

    def addWord(self, word: str) -> None:
        table = self.table
        for i, w in enumerate(word):
            if w not in table:
                table[w] = {}
            table = table[w]
        table["*"] = {}

    def search(self, word: str) -> bool:
        def recursion(word, table):
            for i, w in enumerate(word):
                if w == ".":
                    for k, v in table.items():
                        res = recursion(word[i + 1 :], v)
                        if res:
                            return True
                    return False
                if w not in table:
                    return False
                table = table[w]
            if "*" in table:
                return True
            return False

        return recursion(word, self.table)


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
