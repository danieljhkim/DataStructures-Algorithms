from curses.ascii import isdigit
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


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()