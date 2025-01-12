import heapq
from tkinter import S
from token import NL
from typing import Optional, List
from itertools import accumulate
import random
from collections import Counter, deque, defaultdict, OrderedDict
import math
import bisect
from math import inf
from functools import lru_cache
from sortedcontainers import SortedSet


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

    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        mod = 10**9 + 7
        memo = {}

        def dp(end):
            if end < 0:
                return 0
            if end == 0:
                return 1
            if end in memo:
                return memo[end]
            count = 0
            if end >= zero:
                count += dp(end - zero)
            if end >= one:
                count += dp(end - one)
            memo[end] = count % mod
            return count

        ans = 0
        for i in range(low, high + 1):
            ans += dp(i)
        return ans % mod

    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        self.ans = 0
        N = len(days)
        memo = {}

        def dp(idx, to):
            if (idx, to) in memo:
                return memo[(idx, to)]
            if idx == N:
                return 0
            cur = days[idx]
            res = inf
            if to > cur:
                res = dp(idx + 1, to)
            else:
                res = min(dp(idx + 1, cur + 1) + costs[0], res)
                res = min(dp(idx + 1, cur + 7) + costs[1], res)
                res = min(dp(idx + 1, cur + 30) + costs[2], res)
            memo[(idx, to)] = res
            return res

        return dp(0, 0)

    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        N = len(days)
        memo = {}

        def dp(i):
            if i >= N:
                return 0
            if i in memo:
                return memo[i]

            res = inf
            j = bisect.bisect_left(days, days[i] + 1, i, N)
            res = min(res, costs[0] + dp(j))

            j = bisect.bisect_left(days, days[i] + 7, i, N)
            res = min(res, costs[1] + dp(j))

            j = bisect.bisect_left(days, days[i] + 30, i, N)
            res = min(res, costs[2] + dp(j))

            memo[i] = res
            return res

        return dp(0)

    def countStudents(self, students: List[int], s: List[int]) -> int:
        idx = 0
        students = deque(students)
        N = len(students)
        s = deque(s)
        while students and s and idx < N:
            if students[0] == s[0]:
                students.popleft()
                s.popleft()
                idx = 0
            else:
                cur = students.popleft()
                students.append(cur)
                idx += 1
        return len(students)

    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return TreeNode(val)
        self.found = False

        def dfs(node):
            if self.found or not node:
                return
            if val < node.val:
                if node.left:
                    dfs(node.left)
                else:
                    node.left = TreeNode(val)
                    self.found = True
            if val > node.val:
                if node.right:
                    dfs(node.right)
                else:
                    node.right = TreeNode(val)
                    self.found = True

        dfs(root)
        return root

    def maxScore(self, s: str) -> int:
        ones = s.count("1")
        zeros = 0
        ans = 0
        for i in range(len(s) - 1):
            if s[i] == "1":
                ones -= 1
            else:
                zeros += 1
            ans = max(ans, zeros + ones)
        return ans

    def stringShift(self, s: str, shift: List[List[int]]) -> str:
        dq = deque(s)
        for d, v in shift:
            if d == 0:
                for i in range(v):
                    cur = dq.popleft()
                    dq.append(cur)
            else:
                for i in range(v):
                    cur = dq.pop()
                    dq.appendleft(cur)

        return "".join(dq)

    def isAnagram(self, s: str, t: str) -> bool:
        sc = Counter(s)
        tc = Counter(t)
        return sc == tc

    def titleToNumber(self, columnTitle: str) -> int:
        ans = 0
        for i, w in enumerate(columnTitle):
            ch = ord(w) - ord("A") + 1
            ans = ans * 26 + ch
        return ans

    def exist(self, board: List[List[str]], word: str) -> bool:
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(board)
        COL = len(board[0])

        N = len(word)

        def dfs(r, c, idx, visited):
            if idx == N - 1:
                return True
            if board[r][c] != word[idx]:
                return False
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL:
                    if (nr, nc) not in visited and board[nr][nc] == word[idx + 1]:
                        visited.add((nr, nc))
                        res = dfs(nr, nc, idx + 1)
                        if res:
                            return True
                        else:
                            visited.remove((nr, nc))
            return False

        for r in range(ROW):
            for c in range(COL):
                if board[r][c] == word[0]:
                    visited = set([(r, c)])
                    res = dfs(r, c, 0, visited)
                    if res:
                        return True
        return False

    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(board)
        COL = len(board[0])
        table = {}
        ans = set()
        for w in words:
            diction = table
            for i, l in enumerate(w):
                if l not in diction:
                    diction[l] = {}
                diction = diction[l]
                if i == len(w) - 1:
                    diction["*"] = True

        def dfs(r, c, table, visited, word):
            sword = "".join(word)
            if "*" in table and sword not in ans:
                ans.add(sword)
            if len(table) == 0:
                return
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL:
                    if (nr, nc) not in visited and board[nr][nc] in table:
                        visited.add((nr, nc))
                        diction = table[board[nr][nc]]
                        word.append(board[nr][nc])
                        dfs(nr, nc, diction, visited, word)
                        visited.remove((nr, nc))
                        word.pop()

        for r in range(ROW):
            for c in range(COL):
                if board[r][c] in table:
                    visited = set([(r, c)])
                    dfs(r, c, table[board[r][c]], visited, [board[r][c]])
        return list(ans)

    def sumFourDivisors(self, nums: List[int]) -> int:
        memo = {}

        def divs(num):
            if num in memo:
                return memo[num]
            count = 0
            n = math.isqrt(num)
            total = 0
            i = 1
            while i <= n:
                if num % i == 0:
                    if i * i == num:
                        count += 1
                        total += i
                    else:
                        total += i
                        count += 2
                        total += num // i
                    if count > 4:
                        total = 0
                        break
                i += 1
            if count == 4:
                memo[num] = total
                return total
            memo[num] = 0
            return 0

        ans = 0
        for n in nums:
            ans += divs(n)
        return ans

    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        N = len(nums)
        ans = 1
        cur = 1
        for i in range(1, N):
            if nums[i] != nums[i - 1]:
                cur += 1
            else:
                cur = 1
            ans += cur
        return ans

    def minimumOperationsToWriteY(self, grid: List[List[int]]) -> int:
        yset = set()
        N = len(grid)
        mid = N // 2
        ycounts = defaultdict(int)
        counts = defaultdict(int)
        for i in range(mid, N):
            yset.add((i, mid))
        c = 0
        e = N - 1
        for r in range(0, mid):
            yset.add((r, c))
            yset.add((r, e))
            c += 1
            e -= 1
        for r in range(N):
            for c in range(N):
                color = grid[r][c]
                if (r, c) not in yset:
                    counts[color] += 1
                else:
                    ycounts[color] += 1
        ytotal = len(yset)
        xtotal = N * N - ytotal
        ans = inf
        xarr = sorted([(v, c) for c, v in counts.items()], reverse=True)
        for i in range(3):
            ycolor = ycounts[i]
            ychange = ytotal - ycolor
            xcolor = 0
            for v, c in xarr:
                if c != i:
                    xcolor = v
                    break

            xchange = xtotal - xcolor
            ans = min(ans, xchange + ychange)
        return ans

    def longestSubarray(self, nums: List[int], limit: int) -> int:
        left = 0
        right = 0
        N = len(nums)
        small = nums[0]
        big = small
        ans = 0
        while right < N:
            cur = nums[right]
            diff = abs(cur - big)
            diff2 = abs(cur - small)
            if diff > limit or diff2 > limit:
                ans = max(right - left, ans)
                target = 0
                if diff > diff2:
                    target = big
                else:
                    target = small
                while left < right and nums[left] != target:
                    left += 1

            else:
                ans = max(right - left + 1, ans)
                if cur > big:
                    big = cur
                if small > cur:
                    small = cur
            right += 1
        return ans

    def vowelStrings(self, words: List[str], queries: List[List[int]]) -> List[int]:
        prefix = [0]
        vowels = {"a", "e", "i", "o", "u"}
        for w in words:
            c = 0
            if w[0] in vowels and w[-1] in vowels:
                c = 1
            prefix.append(prefix[-1] + c)
        ans = []
        for s, e in queries:
            c = prefix[e + 1] - prefix[s + 1]
            ans.append(c)
        return ans

    def permute(self, nums: List[int]) -> List[List[int]]:
        N = len(nums)
        ans = []

        def backtrack(arr, pos, nset):
            if len(arr) == N:
                ans.append(arr[:])
                return
            for i in range(pos, N):
                if i not in nset:
                    nset.add(i)
                    arr.append(nums[i])
                    backtrack(arr, pos + 1, nset)
                    arr.pop()
                    nset.remove(i)

        backtrack([], 0, set())
        return ans

    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        letters = []
        digits = []
        for l in logs:
            if not l[-1].isdigit():
                lls = l.split(" ")
                name = lls[0]
                items = " ".join(lls[1:])
                letters.append((items, name))
            else:
                digits.append(l)
        letters.sort(key=lambda x: (x[0], x[1]))
        ans = []
        for l in letters:
            entry = l[1] + " " + l[0]
            ans.append(entry)
        ans.extend(digits)
        return ans

    def sumEvenGrandparent(self, root: Optional[TreeNode]) -> int:

        def dfs(node, count):
            if not node:
                return 0
            if count == 2:
                return node.val
            res = 0
            if count == 1:
                res += dfs(node.left, count + 1)
                res += dfs(node.right, count + 1)
            if node.val % 2 == 0:
                res += dfs(node.left, 1)
                res += dfs(node.right, 1)
            else:
                res += dfs(node.left, 0)
                res += dfs(node.right, 0)
            return res

        return dfs(root, 0)

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True

        def dfs(node, prev, isLeft):
            if not node:
                return True
            if node.left and node.left.val >= node.val:
                return False
            if node.right and node.right.val <= node.val:
                return False
            if isLeft and node.left and node.left.val >= prev:
                return False
            if not isLeft and node.right and node.right.val <= prev:
                return False
            return dfs(node.left, node.val, True) and dfs(node.right, node.val, False)

        return dfs(root.left, root.val, True) and dfs(root.right, root.val, False)

    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        num1 = []
        num2 = []
        cur1 = l1
        cur2 = l2
        while cur1:
            num1.append(str(cur1.val))
            cur1 = cur1.next
        while cur2:
            num2.append(str(cur2.val))
            cur2 = cur2.next
        res = int("".join(num1)) + int("".join(num2))
        res = reversed(str(res))
        head = ListNode(-1)
        cur = head
        for n in res:
            cur2 = ListNode(int(n))
            cur.next = cur2
            cur = cur.next
        return head.next

    def findPairs(self, nums: List[int], k: int) -> int:
        """_summary_
        n1 - n2 = k
        n2 - n1 = k

        n1 - k = n2
        n1 + k = n2
        """
        table = Counter(nums)
        ans = 0
        for n in table:
            if k > 0 and n + k in table:
                ans += 1
            elif k == 0 and table[n] > 1:
                ans += 1
        return ans

    def findPairs(self, nums: List[int], k: int) -> int:
        table = defaultdict(int)
        ans = 0
        for i, n in enumerate(nums):
            if k > 0:
                if n + k in table and table[n + k] > 0:
                    ans += 1
                    table[n + k] = -inf
                if n - k in table and table[n - k] > 0:
                    ans += 1
                    table[n - k] = -inf
            else:
                if n in table and table[n] > 0:
                    ans += 1
                    table[n] = -inf
            table[n] += 1
        return ans

    def candyCrush(self, board: List[List[int]]) -> List[List[int]]:
        ROW = len(board)
        COL = len(board[0])

        def crush_down(r, c, num, visited):
            if not (0 <= r < ROW and 0 <= c < COL) or board[r][c] != num:
                return []
            tarr = set([(r, c)])
            visited.add((r, c))
            up = crush_down(r + 1, c, num, visited)
            tarr.update(up)
            return tarr

        def crush_right(r, c, num, visited):
            if not (0 <= r < ROW and 0 <= c < COL) or board[r][c] != num:
                return []
            tarr = set([(r, c)])
            visited.add((r, c))
            up = crush_right(r, c + 1, num, visited)
            tarr.update(up)
            return tarr

        def down(c):
            zero = ROW - 1
            right = ROW - 1
            while right >= 0:
                if board[right][c] != 0:
                    board[right][c], board[zero][c] = board[zero][c], board[right][c]
                    zero -= 1
                right -= 1

        while True:
            crushed = set()
            vertset = set()
            hortset = set()
            cols = set()
            for r in range(ROW):
                for c in range(COL):
                    if board[r][c] != 0:
                        if (r, c) not in vertset:
                            res = crush_down(r, c, board[r][c], vertset)
                            if len(res) > 2:
                                crushed.update(res)
                        if (r, c) not in hortset:
                            res = crush_right(r, c, board[r][c], hortset)
                            if len(res) > 2:
                                crushed.update(res)
            if len(crushed) == 0:
                return board
            cols = set()
            for r, c in crushed:
                cols.add(c)
                board[r][c] = 0
            for c in cols:
                down(c)

    def countPalindromicSubsequence(self, s: str) -> int:
        table = set(s)
        ans = 0
        for w in table:
            i, j = s.index(w), s.rindex(w)
            tmp = set()
            for k in range(i + 1, j):
                tmp.add(s[k])
            ans += len(tmp)
        return ans

    def longestSubarray(self, nums: List[int], limit: int) -> int:
        dec_dq = deque()
        inc_dq = deque()
        max_len = 0
        left = 0
        N = len(nums)
        for right in range(N):
            cur = nums[right]
            while dec_dq and dec_dq[-1] < cur:
                dec_dq.pop()
            while inc_dq and inc_dq[-1] > cur:
                inc_dq.pop()
            dec_dq.append(cur)
            inc_dq.append(cur)
            while dec_dq and inc_dq and dec_dq[0] - inc_dq[0] > limit:
                if inc_dq[0] == nums[left]:
                    inc_dq.popleft()
                if dec_dq[0] == nums[left]:
                    dec_dq.popleft()
                left += 1
            max_len = max(max_len, right - left + 1)
        return max_len

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        R = len(matrix)
        C = len(matrix[0])
        right = C - 1
        left = 0
        up = 0
        bottom = R - 1
        res = []
        while len(res) < R * C:
            # left right
            for i in range(left, right + 1):
                res.append(matrix[up][i])

            # up down
            for i in range(up + 1, bottom + 1):
                res.append(matrix[i][right])

            if up != bottom:
                # right left
                for i in range(right - 1, left + 1, -1):
                    res.append(matrix[bottom][i])

            if left != right:
                # down up
                for i in range(bottom + 1, up - 1, -1):
                    res.append(matrix[i][left])

            up += 1
            bottom -= 1
            right -= 1
            left += 1
        return res

    def minAvailableDuration(
        self, slots1: List[List[int]], slots2: List[List[int]], duration: int
    ) -> List[int]:
        slots1.sort()
        slots2.sort()
        N1 = len(slots1)
        N2 = len(slots2)
        idx1 = 0
        idx2 = 0
        while idx1 < N1 and idx2 < N2:
            while idx1 < N1 and slots1[idx1][1] - slots1[idx1][0] < duration:
                idx1 += 1
            while idx2 < N2 and slots2[idx2][1] - slots2[idx2][0] < duration:
                idx2 += 1
            if idx1 >= N1 or idx2 >= N2:
                break
            if slots1[idx1][1] <= slots2[idx2][0]:
                idx1 += 1
            elif slots2[idx2][1] <= slots1[idx1][0]:
                idx2 += 1
            elif (
                min(slots1[idx1][1], slots2[idx2][1])
                - max(slots1[idx1][0], slots2[idx2][0])
                >= duration
            ):
                start = max(slots1[idx1][0], slots2[idx2][0])
                return [start, start + duration]
            else:
                if slots2[idx2][1] < slots1[idx1][1]:
                    idx2 += 1
                elif slots2[idx2][1] > slots1[idx1][1]:
                    idx1 += 1
                else:
                    idx2 += 1
                    idx1 += 1
        return []

    def shiftingLetters(self, s: str, shifts: List[List[int]]) -> str:
        N = len(s)
        lines = [0] * (N + 1)
        for st, e, d in shifts:
            if d > 0:
                lines[st] += 1
                lines[e + 1] -= 1
            else:
                lines[st] -= 1
                lines[e + 1] += 1
        total = 0
        for i, n in enumerate(lines):
            total += n
            lines[i] = total

        arr = []
        for i, w in enumerate(s):
            cur = chr((ord(w) - ord("a") + lines[i]) % 26 + ord("a"))
            arr.append(cur)
        return "".join(arr)

    def reverseOnlyLetters(self, s: str) -> str:
        N = len(s)
        left = 0
        right = N - 1
        s = list(s)
        while left < right:
            while left < right and not s[left].isalpha():
                left += 1
            while left < right and not s[right].isalpha():
                right -= 1
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
        return "".join(s)

    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        table = {}
        cur = head
        idx = 0
        while cur:
            if cur in table:
                return cur
            table[cur] = idx
            idx += 1
            cur = cur.next
        return -1

    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        idx = 1
        cur = head
        odds = ListNode(-1)
        dummy = odds
        evens = ListNode(-1)
        dummy_even = evens
        while cur:
            if idx % 2 == 1:
                odds.next = cur
                odds = odds.next
            else:
                evens.next = cur
                evens = evens.next
            idx += 1
            cur = cur.next
        evens.next = None
        odds.next = dummy_even.next
        return dummy.next

    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:

        def recursion(arr):
            if len(arr) == 1:
                return TreeNode(arr[0])
            if len(arr) == 0:
                return None
            mid = len(arr) // 2
            node = TreeNode(arr[mid])
            node.left = recursion(arr[:mid])
            node.right = recursion(arr[mid + 1 :])
            return node

        return recursion(nums)

    def strStr(self, haystack: str, needle: str) -> int:
        return haystack.find(needle)

    def mostVisitedPattern(
        self, username: List[str], timestamp: List[int], website: List[str]
    ) -> List[str]:
        table = defaultdict(list)
        N = len(username)
        for i in range(N):
            entry = (timestamp[i], website[i])
            table[username[i]].append(entry)

        scores = defaultdict(int)
        for k, v in table.items():
            v.sort()
            visited = set()
            if len(v) >= 3:
                for i in range(len(v) - 2):
                    for j in range(i + 1, len(v) - 1):
                        for k in range(j + 1, len(v)):
                            item = (v[i][1], v[j][1], v[k][1])
                            if item not in visited:
                                scores[item] += 1
                                visited.add(item)
        max_count = max(scores.values())
        candidates = [seq for seq, cnt in scores.items() if cnt == max_count]
        return list(min(candidates))

    def suggestedProducts(
        self, products: List[str], searchWord: str
    ) -> List[List[str]]:
        idx = 0
        N = len(searchWord)
        table = defaultdict(list)
        queue = deque(products)
        while queue and idx < N:
            size = len(queue)
            for _ in range(size):
                cur = queue.popleft()
                if len(cur) > idx and cur[idx] == searchWord[idx]:
                    table[idx].append(cur)
                    queue.append(cur)
            idx += 1
        ans = []
        for idx in range(N):
            if idx not in table:
                ans.append([])
                continue
            cur = table[idx]
            cur.sort()
            size = min(len(cur), 3)
            ans.append(cur[:size])
        return ans

    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        ans = []

        def backtrack(arr, idx, total):
            if len(arr) == k and total == n:
                ans.append(arr[:])
                return
            if total > n or idx > 9 or len(arr) > k:
                return
            for i in range(idx + 1, 10):
                arr.append(i)
                backtrack(arr, i, total + i)
                arr.pop()

        backtrack([], 0, 0)
        return ans

    def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
        def dfs(node, res):
            if not node:
                return node
            if not node.left and not node.right:
                res.append(node.val)
                return None
            node.left = dfs(node.left, res)
            node.right = dfs(node.right, res)
            return node

        ans = []
        while root:
            res = []
            root = dfs(root, res)
            ans.append(res)

        return ans

    def minOperations(self, boxes: str) -> List[int]:
        N = len(boxes)
        ans = [0] * N
        left_balls = 0
        left_moves = 0
        right_balls = right_moves = 0
        for i, n in enumerate(boxes):
            ans[i] = left_moves
            left_balls += int(n)
            left_moves += left_balls
        for i in range(N - 1, -1, -1):
            ans[i] = right_moves
            right_balls += int(n)
            right_moves += right_balls
        return ans

    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        def dfs(node):
            if not node or node == p or node == q:
                return node
            left = dfs(node.left)
            right = dfs(node.right)
            if left and right:
                return node
            return left or right

        return dfs(root)

    def compareVersion(self, version1: str, version2: str) -> int:
        v1 = version1.split(".")
        v2 = version2.split(".")
        N = max(len(v1), len(v2))
        for i in range(N):
            val1 = 0
            val2 = 0
            if i < len(v1):
                val1 = int(v1[i])
            if i < len(v2):
                val2 = int(v2[i])
            if val1 < val2:
                return -1
            elif val1 > val2:
                return 1
        return 0

    def multiply(self, num1: str, num2: str) -> str:
        """ "
        123
         19
        """
        N1 = len(num1)
        N2 = len(num2)
        dq = deque()
        carry = 0
        idx1 = N1 - 1
        summ = 0
        counter = 1
        while idx1 >= 0:
            n1 = int(num1[idx1])
            stack = []
            carry = 0
            for j in range(N2 - 1, -1, -1):
                n2 = int(num2[j])
                total = n1 * n2 + carry
                if total > 9:
                    carry = total // 10
                    total = total % 10
                else:
                    carry = 0
                stack.append(str(total))
            if carry:
                stack.append(str(carry))
            stack.reverse()
            out = int("".join(stack))
            out *= counter
            summ += out
            counter *= 10
            idx1 -= 1
        return summ

    def mySqrt(self, x: int) -> int:
        if x < 2:
            return 0
        n = 1
        prev = n
        while n * n <= x:
            prev = n
            n = n * n + 1
        return prev

    def connect(self, root: "Node") -> "Node":
        if not root:
            return root
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

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        dq = deque()
        for i in range(k):
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()
            dq.append(i)

        ans = [nums[dq[0]]]
        N = len(nums)
        for i in range(k, N):
            while dq and i - dq[0] + 1 > k:
                dq.popleft()
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()
            dq.append(i)
            ans.append(nums[dq[0]])
        return ans

    def slowestKey(self, releaseTimes: List[int], keysPressed: str) -> str:
        key = keysPressed[0]
        time = releaseTimes[0]
        N = len(releaseTimes)

        for i in range(1, N):
            prev = releaseTimes[i - 1]
            cur = releaseTimes[i]
            diff = cur - prev
            if diff > time:
                key = [keysPressed[i]]
                time = diff
            elif diff == time:
                key.append(keysPressed[i])
        key.sort()
        return key[-1]

    def numKLenSubstrNoRepeats(self, s: str, k: int) -> int:
        left = 0
        N = len(s)
        if N < k:
            return 0
        seen = set()
        left = 0
        count = 0

        for right, ch in enumerate(s):
            while ch in seen:
                seen.remove(s[left])
                left += 1
            seen.add(ch)

            while right - left + 1 > k:
                seen.remove(s[left])
                left += 1

            if right - left + 1 == k:
                count += 1
                seen.remove(s[left])
                left += 1

        return count

    def minDifficulty(self, jobs: List[int], d: int) -> int:
        if len(jobs) < d:
            return -1
        memo = {}
        N = len(jobs)

        def dpp(idx, left):
            if idx == N and left == 0:
                return 0
            if idx >= N or left <= 0:
                return float("inf")
            if (idx, left) in memo:
                return memo[(idx, left)]
            top = -1
            res = float("inf")
            i = idx
            while i < N:
                top = max(top, jobs[i])
                res = min(dpp(i + 1, left - 1) + top, res)
                i += 1
            memo[(idx, left)] = res
            return res

        ans = dpp(0, d)
        return ans if ans != float("inf") else -1

    def longestSubarray(self, nums: List[int], limit: int) -> int:
        min_dq = deque()
        max_dq = deque()
        ans = -inf
        left = 0
        for i, n in enumerate(nums):
            while min_dq and nums[min_dq[-1]] > n:
                min_dq.pop()
            min_dq.append(i)
            while max_dq and nums[max_dq[-1]] < n:
                max_dq.pop()
            max_dq.append(i)
            while max_dq and min_dq and nums[max_dq[0]] - nums[min_dq[0]] > limit:
                if max_dq[0] == left:
                    max_dq.popleft()
                if min_dq[0] == left:
                    min_dq.popleft()
                left += 1
            ans = max(ans, i - left + 1)
        return ans

    def longestSubarray(self, nums: List[int], limit: int) -> int:
        min_heap = []
        max_heap = []
        left = 0
        ans = -inf
        for i, n in enumerate(nums):
            heapq.heappush(min_heap, (n, i))
            heapq.heappush(max_heap, (-n, i))
            while min_heap and max_heap and -max_heap[0][0] - min_heap[0][0] > limit:
                if max_heap[0][1] == left:
                    heapq.heappop(max_heap)
                if min_heap[0][1] == left:
                    heapq.heappop(min_heap)
                left += 1
            ans = max(ans, i - left + 1)
        return ans

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        ROW = len(matrix)
        COL = len(matrix[0])
        left = 0
        right = COL - 1
        top = 0
        bottom = ROW - 1
        res = []
        while len(res) < ROW * COL:
            for i in range(left, right + 1):
                res.append(matrix[top][i])
            for i in range(top + 1, bottom + 1):
                res.append(matrix[i][right])
            if len(res) == ROW * COL:
                break
            for i in range(right - 1, left - 1, -1):
                res.append(matrix[bottom][i])
            if len(res) == ROW * COL:
                break
            for i in range(bottom - 1, top, -1):
                res.append(matrix[i][left])
            left += 1
            right -= 1
            top += 1
            bottom -= 1
        return res

    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        trie = {}
        ans = 0

        def build(num):
            table = trie
            for w in num:
                if w not in table:
                    table[w] = {}
                table = table[w]

        def size(num):
            table = trie
            cnt = 0
            for w in num:
                if w not in table:
                    return cnt
                table = table[w]
                cnt += 1
            return cnt

        for n in arr1:
            build(str(n))
        for n in arr2:
            ans = max(ans, size(str(n)))
        return ans

    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: (x[0], -x[1]))
        stack = []
        cnt = 0
        for s, e in intervals:
            while stack and stack[-1] > s:
                stack.pop()
                cnt += 1
            stack.append(e)
        return cnt

    def stringMatching(self, words: List[str]) -> List[str]:
        table = {}
        ans = set()
        for w in words:
            table[w] = set(w)
        for w in words:
            for word in table:
                if len(word) >= len(w) and w != word:
                    if table[w].issubset(table[word]):
                        if word.index(w) >= 0:
                            ans.add(w)

        return list(ans)

    def compress(self, chars: List[str]) -> int:
        stack = []
        idx = 0
        N = len(chars)
        while idx < N:
            cur = chars[idx]
            i = idx
            while i < N and chars[i] == cur:
                i += 1
            cnt = i - idx
            stack.append(cur)
            if cnt > 1:
                cnt = str(cnt)
                for c in cnt:
                    stack.append(c)
            idx = i
        chars.clear()
        chars.extend(stack)
        return len(chars)

    def killProcess(self, pid: List[int], ppid: List[int], kill: int) -> List[int]:
        adj = defaultdict(list)
        N = len(pid)
        for i in range(N):
            p = pid[i]
            parent = ppid[i]
            adj[parent].append(p)

        self.ans = []
        visited = set([kill])

        def dfs(cur):
            self.ans.append(cur)
            for c in adj[cur]:
                if c not in visited:
                    visited.add(c)
                    dfs(c)

        dfs(kill)
        return self.ans

    def reverseWords(self, s: str) -> str:
        arr = s.split(" ")
        for i, w in enumerate(arr):
            rev = reversed(list(w))
            arr[i] = "".join(rev)
        return " ".join(arr)

    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        dummy = ListNode(-1, head)
        cur = dummy
        while cur and cur.next and cur.next.next:
            first = cur.next
            second = cur.next.next
            first.next = second.next
            second.next = first
            cur.next = second
            cur = first
        return dummy.next

    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        ans = set()
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(board)
        COL = len(board[0])
        adj = defaultdict(list)

        def dfs(idx, word, used, r, c):
            if idx == len(word):
                return True
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL:
                    if board[nr][nc] == word[idx] and (nr, nc) not in used:
                        used.add((nr, nc))
                        res = dfs(idx + 1, word, used, nr, nc)
                        if res:
                            return True
                        used.remove((nr, nc))
            return False

        for r in range(ROW):
            for c in range(COL):
                w = board[r][c]
                adj[w].append((r, c))

        for word in words:
            if word not in ans:
                for r, c in adj[word[0]]:
                    if dfs(1, word, {(r, c)}, r, c):
                        ans.add(word)
                        break
        return list(ans)

    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        ans = set()
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(board)
        COL = len(board[0])
        table = {}

        def dfs(node, used, r, c):
            if "*" in node:
                ans.add(node["*"])
            if not node or len(ans) == len(words):
                return
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL:
                    letter = board[nr][nc]
                    if letter in node and (nr, nc) not in used:
                        used.add((nr, nc))
                        dfs(node[letter], used, nr, nc)
                        used.remove((nr, nc))

        def build_trie(word):
            cur = table
            for w in word:
                if w not in cur:
                    cur[w] = {}
                cur = cur[w]
            cur["*"] = word

        for w in words:
            build_trie(w)

        for r in range(ROW):
            for c in range(COL):
                w = board[r][c]
                if w in table:
                    dfs(table[w], {(r, c)}, r, c)

        return list(ans)

    def peopleIndexes(self, favoriteCompanies: List[List[str]]) -> List[int]:
        setarr = [set(com) for com in favoriteCompanies]
        ans = []
        for i, n in enumerate(setarr):
            good = True
            for j, v in enumerate(setarr):
                if j != i:
                    if v.issuperset(n):
                        good = False
                        break
            if good:
                ans.append(i)
        return ans

    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        table = defaultdict(list)
        for i, w in enumerate(words):
            char = w[0]
            table[char].append((w, i))
        ans = 0
        for i, w in enumerate(words):
            options = table[w[0]]
            found = False
            for o, j in options:
                if i == j:
                    found = True
                    continue
                if not found:
                    continue
                if o.startswith(w) and o.endswith(w):
                    ans += 1

        return ans

    def coinChange(self, coins: List[int], amount: int) -> int:
        memo = {}

        def dp(left):
            if left in memo:
                return memo[left]
            if left < 0:
                return inf
            if left == 0:
                return 0
            cnt = inf
            for c in coins:
                cnt = min(dp(left - c), cnt)
            memo[left] = cnt + 1
            return memo[left]

        res = dp(amount)
        return res if res != inf else -1

    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float("inf")] * (amount + 1)
        dp[0] = 0

        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != float("inf") else -1

    def prefixCount(self, words: List[str], pref: str) -> int:
        cnt = 0
        for w in words:
            if w.startswith(pref):
                cnt += 1
        return cnt

    def equalDigitFrequency(self, s: str) -> int:
        ans = set()
        N = len(s)
        for i in range(N):
            counts = defaultdict(int)
            total = 0
            max_cnt = 0
            for j in range(i, N):
                num = int(s[j])
                counts[num] += 1
                max_cnt = max(max_cnt, counts[num])
                total += 1
                if total // len(counts) == max_cnt and total % len(counts) == 0:
                    ans.add(s[i : j + 1])

        return len(ans)

    def runningSum(self, nums: List[int]) -> List[int]:
        arr = [nums[0]]
        for n in nums[1:]:
            arr.append(arr[-1] + n)
        return arr

    def maxFreq(self, s: str, maxLetters: int, minSize: int, maxSize: int) -> int:
        N = len(s)
        left = right = 0
        table = defaultdict(int)
        freq = [0] * 26
        dis = 0
        while right < N and left < N:
            idx = ord(s[right]) - ord("a")
            freq[idx] += 1
            if freq[idx] == 1:
                dis += 1
            while right - left + 1 > minSize:
                idx = ord(s[left]) - ord("a")
                freq[idx] -= 1
                if freq[idx] == 0:
                    dis -= 1
                left += 1
            if right - left + 1 >= minSize and dis <= maxLetters:
                table[s[left : right + 1]] += 1
            right += 1

        if len(table) == 0:
            return 0
        return max(table.values())

    def canConstruct(self, s: str, k: int) -> bool:
        if k > len(s):
            return False
        if k == len(s):
            return True
        counts = Counter(s)
        odds = 0
        for v in counts.items():
            if v % 2 == 1:
                odds += 1
        return odds <= k

    def isArmstrong(self, n: int) -> bool:
        K = len(str(n))
        rem = 0
        total = 0
        prev = n
        while n > 0:
            rem = n % 10
            total += rem**K
            if total > prev:
                return False
            n //= 10
        return total == prev

    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        """ "
        root, left1, left2, left3, right1,
        """
        N = len(preorder)
        self.idx = 0

        def build(prev):
            if self.idx == N:
                return None
            if preorder[self.idx] > prev:
                return None
            cur = TreeNode(preorder[self.idx])
            self.idx += 1
            cur.left = build(cur.val)
            cur.right = build(prev)

            return cur

        if len(preorder) == 0:
            return None
        return build(inf)

    def maxDepth(self, root: "Node") -> int:
        if not root:
            return 0
        depth = 0
        for c in root.children:
            depth = max(depth, self.maxDepth(c))
        return depth + 1

    def bstToGst(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(node, total):
            if not node:
                return 0
            right = dfs(node.right, total)
            prev = node.val
            node.val += right + total
            left = dfs(node.left, node.val)
            return prev + left + right

        dfs(root, 0)
        return root

    def missingNumber(self, nums: List[int]) -> int:
        total = sum(nums)
        N = len(nums)
        total2 = sum([i for i in range(0, N)])
        return total2 - total

    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        y = coordinates[0][1] - coordinates[1][1]
        x = coordinates[0][0] - coordinates[1][0]
        if x == 0:
            for xx, y in coordinates:
                if xx != coordinates[0][0]:
                    return False
            return True
        slope = y / x
        for i in range(len(coordinates) - 1):
            yc = coordinates[i][1] - coordinates[i + 1][1]
            xc = coordinates[i][0] - coordinates[i + 1][0]
            if xc == 0:
                return False
            slope2 = yc / xc
            if slope != slope2:
                return False
        return True

    def numTeams(self, rating: List[int]) -> int:
        """ "
        [1,2,3,4,5]
        3 + 2 + 1
        """

        N = len(rating)
        ans = 0
        for i in range(N):
            small_left = big_left = 0
            small_right = big_right = 0
            for j in range(i):
                if rating[j] < rating[i]:
                    small_left += 1
                else:
                    big_left += 1
            for j in range(i + 1, N):
                if rating[j] < rating[i]:
                    small_right += 1
                else:
                    big_right += 1
            ans += small_left * big_right
            ans += big_left * small_right
        return ans

    def maxDepth(self, root: "Node") -> int:
        if not root:
            return 0
        stack = [(root, 1)]
        ans = 0
        while stack:
            cur, depth = stack.pop()
            ans = max(depth, ans)
            for c in cur.children:
                stack.append((c, depth + 1))
        return ans

    def minAvailableDuration(
        self, slots1: List[List[int]], slots2: List[List[int]], duration: int
    ) -> List[int]:
        slots1.sort()
        slots2.sort()
        N1, N2 = len(slots1), len(slots2)
        i1 = i2 = 0
        while i1 < N1 and i2 < N2:
            s1, e1 = slots1[i1]
            s2, e2 = slots2[i2]
            e = min(e1, e2)
            s = max(s2, s1)
            if e - s >= duration:
                return [s, s + duration]
            if e1 > e2:
                i2 += 1
            else:
                i1 += 1
        return []

    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        banned = set(banned)
        table = defaultdict(int)
        N = len(paragraph)
        left = right = 0
        while right < N:
            cur = paragraph[right]
            if not cur.isalnum():
                right += 1
                left = right
            while right < N and paragraph[right].isalnum():
                right += 1
            word = paragraph[left:right]
            table[word.lower()] += 1
            right += 1
            left = right

        cnt = 0
        word = None
        for k, v in table.items():
            if v > cnt and k not in banned:
                word = k
                cnt = v
        return word

    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        def dfs(cur):
            if not cur:
                return -inf, -inf
            left, ltotal = dfs(cur.left)
            right, rtotal = dfs(cur.right)
            left2 = left + cur.val
            right2 = right + cur.val

            return max(left2, right2, cur.val), max(
                left + right + cur.val, ltotal, rtotal, cur.val, left, right
            )

        one, two = dfs(root)
        return max(one, two)

    def maxVacationDays(self, flights: List[List[int]], days: List[List[int]]) -> int:
        """ "
        flights[i][j] - city i to city j
        days[i][j] - max vacay days in city i in week j
        """
        W = len(days[0])
        N = len(flights)
        memo = {}

        def dp(city, week):
            if week == W - 1:
                return days[city][week]
            if (city, week) in memo:
                return memo[city, week]
            res = 0
            for c in range(N):
                if flights[city][c] == 1 or c == city:
                    vacay = dp(c, week + 1)
                    res = max(vacay, res)
            res += days[city][week]
            memo[(city, week)] = res
            return res

        ans = 0
        for j in range(N):
            if j == 0 or flights[0][j]:
                ans = max(ans, dp(j, 0))
        return ans


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
    dd = OrderedDict()
