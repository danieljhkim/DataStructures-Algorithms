import heapq
import random
import math
import bisect
from typing import *
from math import inf, factorial, gcd, lcm
from functools import lru_cache, cache
from heapq import heapify, heappush, heappop
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

    def punishmentNumber(self, n: int) -> int:
        memo = {}

        def dp(snum, cur, idx):
            if idx == len(snum):
                if cur == 0:
                    return True
                return False
            if cur < 0:
                return False
            if (snum, cur, idx) in memo:
                return memo[(snum, cur, idx)]
            res = False
            for i in range(idx, len(snum)):
                n = int(snum[idx : i + 1])
                res = dp(snum, cur - n, i + 1) or res
                if res:
                    break
            memo[(snum, cur, idx)] = res
            return res

        ans = 0
        for i in range(1, n):
            num = str(i * i)
            res = dp(num, i, 0)
            if res:
                ans += i * i
        return ans

    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        covered = set([None])
        self.ans = 0

        def dfs(node, prev):
            if not node:
                return 0
            left = dfs(node.left, node)
            right = dfs(node.right, node)
            res = 0
            if prev == None and node not in covered:
                covered.add(node)
                res += 1
            elif node.left not in covered or node.right not in covered:
                covered.update({node.left, node.right, node, prev})
                res += 1
            return left + right + res

        return dfs(root, None)

    def earliestFullBloom(self, plantTime: List[int], growTime: List[int]) -> int:
        heap = []
        N = len(plantTime)
        for i in range(N):
            heap.append((-growTime[i], plantTime[i]))
        heapq.heapify(heap)
        g, p = heapq.heappop(heap)
        cur_time = p
        g_time = -g
        while heap:
            g, p = heapq.heappop(heap)
            cur_time += p
            g_time -= p
            g_time = max(-g, g_time)
        return g_time + cur_time

    def equalPairs(self, grid: List[List[int]]) -> int:
        rtable = defaultdict(int)
        ans = 0

        for row in grid:
            rtable[tuple(row)] += 1

        for col in zip(*grid):
            c = tuple(col)
            if c in rtable:
                ans += rtable[c]
        return ans

    def lengthOfLongestSubstring(self, s: str) -> int:
        counts = defaultdict(int)
        ans = left = right = 0
        N = len(s)
        while right < N:
            w = s[right]
            counts[w] += 1
            while counts[w] > 1:
                counts[s[left]] -= 1
                left += 1
            ans = max(right - left + 1, ans)
        return ans

    def connect(self, root: "Optional[Node]") -> "Optional[Node]":
        if not root:
            return None
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

    def lastStoneWeight(self, stones: List[int]) -> int:
        heap = []
        for s in stones:
            heap.append(-s)
        heapq.heapify(heap)

        while len(heap) > 1:
            cur = heapq.heappop(heap)
            if heap[0] == cur:
                heapq.heappop(heap)
            else:
                heapq.heapreplace(heap, cur - heap[0])

        return -heap[0] if heap else 0

    def lastStoneWeightII(self, stones: List[int]) -> int:
        N = len(stones)

        @cache
        def dp(i, current):
            if i == N:
                return abs(current)
            add = dp(i + 1, current + stones[i])
            sub = dp(i + 1, current - stones[i])
            return min(add, sub)

        return dp(0, 0)

    def twoSumLessThanK(self, nums: List[int], k: int) -> int:
        best = -1
        nums.sort()
        for i, n in enumerate(nums):
            t = k - n
            if t < 0:
                break
            idx = bisect.bisect_left(nums, t, lo=i) - 1
            total = n + nums[idx]
            if total < k:
                best = max(best, total)
        return best

    def find132pattern(self, nums: List[int]) -> bool:
        """ "
        i j k
        i k j
        """
        third = float("-inf")
        stack = []
        for num in reversed(nums):
            if num < third:
                return True
            while stack and num > stack[-1]:
                third = stack.pop()
            stack.append(num)
        return False

    def maxSubstringLength(self, s: str, k: int) -> bool:
        if k > len(s):
            return False
        if k < 1:
            return True
        N = len(s)
        bad = [0] * 26
        for w in s:
            bad[ord(w) - ord("a")] += 1

        good = 0
        for n in bad:
            if n == 1:
                good += 1
        if good >= k:
            return True

        @cache
        def dfs(idx, options, cnt):
            if cnt == 0:
                return True
            if idx >= N:
                return False
            ops = list(options)
            leftover = [0] * 26
            right = idx
            res = False
            while right < N:
                cur = s[right]
                alp = ord(cur) - ord("a")
                ops[alp] -= 1
                leftover[alp] = ops[alp]
                if sum(leftover) == 0 and right - idx != N - 1:
                    return dfs(right + 1, tuple(ops), cnt - 1)
                elif leftover[alp] == 0 and right - idx != N - 1:
                    j = right
                    tmp = [0] * 26
                    oo = list(options)
                    while j >= idx:
                        al = ord(s[j]) - ord("a")
                        oo[al] -= 1
                        tmp[al] = oo[al]
                        if sum(tmp) == 0:
                            break
                        j -= 1
                    if sum(tmp) == 0:
                        return dfs(right + 1, tuple(oo), cnt - 1)
                right += 1
            return res

        return dfs(0, tuple(bad), k)

    def minSwaps(self, data: List[int]) -> int:
        total = sum(data)
        if total == 1:
            return 0
        left = 0
        right = total
        cnt = sum(data[left:right])
        ans = total - cnt

        N = len(data)
        while right < N:
            cur = data[right]
            prev = data[left]
            cnt += cur - prev
            ans = min(total - cnt, ans)
            right += 1
            left += 1
        return ans

    # 1861. Rotating the Box
    def rotateTheBox(self, boxGrid: List[List[str]]) -> List[List[str]]:
        R = len(boxGrid)
        C = len(boxGrid[0])
        grid = [["."] * R for _ in range(C)]

        for r in range(R):
            rocks = 0
            for c in range(C):
                val = boxGrid[r][c]
                if val == "#":
                    rocks += 1
                elif val == "*":
                    grid[c][r] = "*"
                    idx = c
                    while rocks > 0:
                        idx -= 1
                        rocks -= 1
                        grid[idx][r] = "#"
            idx = C
            while rocks > 0:
                idx -= 1
                rocks -= 1
                grid[idx][r] = "#"

        for row in grid:
            row.reverse()
        return grid

    # 1197. Minimum Knight Moves
    def minKnightMoves(self, x: int, y: int) -> int:
        moves = ((2, 1), (1, 2), (-2, 1), (-1, 2), (-2, -1), (-1, -2), (2, -1), (1, -2))
        heap = [(0, 0, 0)]
        distances = defaultdict(lambda: inf)
        while heap:
            dist, r, c = heapq.heappop(heap)
            if r == x and c == y:
                return dist
            for dr, dc in moves:
                nr = dr + r
                nc = dc + c
                if distances[(nr, nc)] > dist + 1:
                    distances[(nr, nc)] = dist + 1
                    heapq.heappush(heap, (dist + 1, nr, nc))
        return -1

    # 1028. Recover a Tree From Preorder Traversal
    def recoverFromPreorder(self, nodes: str) -> Optional[TreeNode]:
        stack = []
        idx = 0
        root = None
        N = len(nodes)

        while idx < N:
            cnt = 0
            while idx < N and nodes[idx] == "-":
                cnt += 1
                idx += 1
            while stack and stack[-1][1] >= cnt:
                stack.pop()
            cur = ""
            while idx < N and nodes[idx].isdigit():
                cur += nodes[idx]
                idx += 1
            nnode = TreeNode(int(cur))
            if not root:
                root = nnode
            if stack:
                if not stack[-1][0].left:
                    stack[-1][0].left = nnode
                else:
                    stack[-1][0].right = nnode
                    stack.pop()
            stack.append((nnode, cnt))
        return root

    # 487. Max Consecutive Ones II
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        stack = []
        max_len = cur = 0
        for n in nums:
            if n == 0:
                prev = 0
                if stack:
                    prev = stack.pop() + 1
                max_len = max(prev + cur, max_len)
                stack.clear()
                stack.append(cur)
                cur = 0
            else:
                cur += 1
        prev = 0
        if stack:
            prev = stack.pop() + 1
        max_len = max(prev + cur, max_len)
        return max_len

    # 889. Construct Binary Tree from Preorder and Postorder Traversal
    def constructFromPrePost(
        self, preorder: List[int], postorder: List[int]
    ) -> Optional[TreeNode]:
        N = len(preorder)

        def build(prel, prer, postr):
            if prel > prer:
                return None
            if prel == prer:
                return TreeNode(preorder[prel])
            left_node = preorder[prel + 1]
            n_nodes = 1
            while n_nodes + postr < N and postorder[postr + n_nodes - 1] != left_node:
                n_nodes += 1
            root = TreeNode(preorder[prel])
            root.left = build(prel + 1, prel + n_nodes, postr)
            root.right = build(prel + n_nodes + 1, prer, postr + n_nodes)
            return root

        return build(0, N - 1, 0)

    # 2023. Number of Pairs of Strings With Concatenation Equal to Target
    def numOfPairs(self, nums: List[str], target: str) -> int:
        table = defaultdict(int)
        cnt = 0
        nset = set()
        N = len(target)

        for i, n in enumerate(nums):
            if len(n) >= N:
                continue
            nset.add(len(n))
            table[n] += 1

        for diff in nset:
            left = target[:diff]
            right = target[diff:]
            if left in table and right in table:
                if left != right:
                    cnt += table[left] * table[right]
                else:
                    cnt += table[left] * (table[left] - 1)
        return cnt

    # 2467. Most Profitable Path in a Tree
    def mostProfitablePath(
        self, edges: List[List[int]], bob: int, amount: List[int]
    ) -> int:
        adj = defaultdict(list)
        for s, e in edges:
            adj[s].append(e)
            adj[e].append(s)

        parent = {0: None}
        queue = deque([0])
        while queue:
            cur = queue.popleft()
            for nei in adj[cur]:
                if nei not in parent:
                    parent[nei] = cur
                    queue.append(nei)

        def dfs(a, b, avisited):
            if a == b:
                if amount[a] != 0:
                    ap = amount[a] // 2
                else:
                    ap = 0
            else:
                ap = amount[a]
            prev_a, prev_b = amount[a], amount[b]
            amount[a] = 0
            amount[b] = 0
            best = -inf
            is_leaf = True
            for adest in adj[a]:
                if adest not in avisited:
                    avisited.add(adest)
                    is_leaf = False
                    if b != 0:
                        best = max(best, dfs(adest, parent[b], avisited))
                    else:
                        best = max(best, dfs(adest, b, avisited))
                    avisited.remove(adest)
            amount[a] = prev_a
            amount[b] = prev_b
            if is_leaf:
                return ap
            return best + ap

        return dfs(0, bob, set([0]))

    # 1524. Number of Sub-arrays With Odd Sum
    def numOfSubarrays(self, arr: List[int]) -> int:
        MOD = 10**9 + 7
        evens = 1
        odds = 0
        ans = total = 0
        for i, n in enumerate(arr):
            total += n
            if total % 2 == 0:
                ans += odds
                evens += 1
            else:
                ans += evens
                odds += 1
            ans %= MOD
        return ans

    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.ans = None

        def dfs(node, cnt):
            if self.ans:
                return 0
            if not node:
                return cnt
            cnt = dfs(node.left, cnt)
            cnt += 1
            if cnt == k:
                self.ans = node
            cnt = dfs(node.right, cnt)

        dfs(root, 0)
        return self.ans

    def maxAbsoluteSum(self, nums: List[int]) -> int:
        min_sum = 0
        max_sum = 0
        total = 0
        for n in nums:
            total += n
            min_sum = min(min_sum, total)
            max_sum = max(max_sum, total)
        return max_sum - min_sum

    # 2460. Apply Operations to an Array
    def applyOperations(self, nums: List[int]) -> List[int]:
        N = len(nums)
        for i in range(N - 1):
            n1 = nums[i]
            n2 = nums[i + 1]
            if n1 == n2:
                nums[i] = n1 * 2
                nums[i + 1] = 0

        idx = 0
        for i in range(N):
            if nums[i] != 0:
                nums[idx] = nums[i]
                idx += 1

        for i in range(idx, N):
            nums[i] = 0
        return nums

    # 3467. Transform Array by Parity
    def transformArray(self, nums: List[int]) -> List[int]:
        evens = 0
        for n in nums:
            if n % 2 == 0:
                evens += 1

        for i in range(len(nums)):
            if evens > 0:
                nums[i] = 0
                evens -= 1
            else:
                nums[i] = 1
        return nums

    # 3468. Find the Number of Copy Arrays
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        N = len(original)
        s, e = bounds[0][0], bounds[0][1]
        cnt = e - s + 1

        prev = s
        for i in range(1, N):
            diff = original[i] - original[i - 1]
            prev = diff + prev
            if prev < bounds[i][0]:
                cnt -= bounds[i] - prev
                prev = bounds[i]
            if not (bounds[i][0] <= prev <= bounds[i][1]):
                return 0
            cnt = min(cnt, bounds[i][1] - prev + 1)
        if cnt < 0:
            return 0
        return cnt

    # 3469. Find Minimum Cost to Remove Array Elements
    def minCost(self, nums: List[int]) -> int:
        N = len(nums)

        @cache
        def dp(idx, rem):
            if idx == N - 1:
                return max(nums[idx], rem)
            if idx == N - 2:
                return min(min(nums[idx:]), rem) + max(max(nums[idx:]), rem)
            if idx == N:
                return rem
            n1 = nums[idx]
            n2 = nums[idx + 1]
            tmp = sorted([n1, n2, rem])
            res = dp(idx + 2, tmp[0]) + tmp[-1]
            res = min(dp(idx + 2, tmp[-1]) + tmp[1], res)
            return res

        return dp(1, nums[0])

    # 3470. Permutations IV
    def permute(self, n: int, k: int) -> List[int]:
        pos = math.factorial(n)
        if k > pos:
            return []
        ans = []
        self.cnt = 0
        nums = [i for i in range(1, n + 1)]

        def backtrack(arr, nums):
            if self.cnt == k:
                return
            if len(arr) == n:
                self.cnt += 1
                if self.cnt == k:
                    ans.append(tuple(arr[:]))
                return
            size = len(nums)
            for i in range(size):
                val = nums[i]
                if val == -1:
                    continue
                if not arr or (val % 2 == 0) != (arr[-1] % 2 == 0):
                    nums[i] = -1
                    arr.append(val)
                    backtrack(arr, nums)
                    arr.pop()
                    nums[i] = val

        backtrack([], nums)
        if self.cnt == k:
            return ans[0]
        return []

    # 867. Transpose Matrix
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        R = len(matrix)
        C = len(matrix[0])
        tmat = [[0] * R for _ in range(C)]

        for r in range(C):
            for c in range(R):
                tmat[r][c] = matrix[c][r]
        for row in tmat:
            row.reverse()
        return tmat

    # 286. Walls and Gates
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        """
        Do not return anything, modify rooms in-place instead.
        """
        R = len(rooms)
        C = len(rooms[0])
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        dq = deque()

        for r in range(R):
            for c in range(C):
                if rooms[r][c] == 0 and (r, c):
                    dq.append((0, r, c))
        while dq:
            dist, rr, cc = dq.popleft()
            for dr, dc in directions:
                nr = dr + rr
                nc = dc + cc
                if 0 <= nr < R and 0 <= nc < C:
                    val = rooms[nr][nc]
                    if val != -1:
                        if val > dist + 1:
                            rooms[nr][nc] = dist + 1
                            dq.append((dist + 1, nr, nc))

    # 803. Bricks Falling When Hit
    def hitBricks(
        self, grid: List[List[int]], hits: List[List[int]]
    ) -> List[int]:  # TLE
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        R = len(grid)
        C = len(grid[0])
        ans = []

        def dfs(r, c, visited):
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 1:
                    pos = (nr, nc)
                    if pos not in visited:
                        visited.add(pos)
                        dfs(nr, nc, visited)

        visited = set()
        for r in range(R):
            for c in range(C):
                pos = (r, c)
                if pos not in visited:
                    visited.add(pos)
                    dfs(r, c, visited)
        del visited

        def check(r, c, paths):
            if r == 0:
                return True
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 1:
                    pos = (nr, nc)
                    if pos not in paths:
                        paths.add(pos)
                        res = check(nr, nc, paths)
                        if res:
                            return True
            return False

        for r, c in hits:
            cnt = 0
            if grid[r][c] == 1:
                grid[r][c] = 0
                for dr, dc in directions:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 1:
                        if nr == 0:
                            continue
                        paths = set([(nr, nc)])
                        res = check(nr, nc, paths)
                        if not res:
                            cnt += len(paths)
                            for rr, cc in paths:
                                grid[rr][cc] = 0
            ans.append(cnt)
        return ans

    def plusOne(self, digits: List[int]) -> List[int]:
        N = len(digits)
        last = digits[-1] + 1
        if last >= 10:
            digits[-1] = 0
            carry = 1
            for i in range(N - 2, -1, -1):
                n = digits[i] + carry
                if n >= 10:
                    carry = 1
                    digits[i] = 0
                else:
                    digits[i] = n
                    carry = 0
                    break
            if carry == 1:
                digits.insert(0, 1)
        else:
            digits[-1] = last
        return digits

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        ans = [intervals[0]]
        for s, e in intervals[1:]:
            end = ans[-1][1]
            if s <= end:
                ans[-1][1] = max(end, e)
            else:
                ans.append([s, e])
        return ans

    # 833. Find And Replace in String
    def findReplaceString(
        self, s: str, indices: List[int], sources: List[str], targets: List[str]
    ) -> str:
        N = len(s)
        K = len(indices)
        arr = []
        idx = 0
        ops = []

        for k in range(K):
            i = indices[k]
            w = sources[k]
            ops.append((i, w, targets[k]))

        ops.sort(reverse=True)
        while idx < N:
            found = False
            while ops and ops[-1][0] == idx:
                _, w, nw = ops.pop()
                n = len(w)
                if s[idx : idx + n] == w:
                    arr.append(nw)
                    idx += n
                    found = True
                    break
            if found:
                continue
            arr.append(s[idx])
            idx += 1
        return "".join(arr)

    # 1254. Number of Closed Islands
    def closedIsland(self, grid: List[List[int]]) -> int:
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        R = len(grid)
        C = len(grid[0])
        ans = 0

        def dfs(r, c):
            grid[r][c] = 1
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < R and 0 <= nc < C:
                    if grid[nr][nc] == 0:
                        dfs(nr, nc)

        for c in range(C):
            if grid[0][c] == 0:
                dfs(0, c)
            if grid[-1][c] == 0:
                dfs(R - 1, c)

        for r in range(R):
            if grid[r][0] == 0:
                dfs(r, 0)
            if grid[r][C - 1] == 0:
                dfs(r, C - 1)

        for r in range(1, R - 1):
            for c in range(1, C - 1):
                if grid[r][c] == 0:
                    ans += 1
                    dfs(r, c)

        return ans

    # 1151. Minimum Swaps to Group All 1's Together
    def minSwaps(self, data: List[int]) -> int:
        total = sum(data)
        left = 0
        right = total
        cur_total = sum(data[:right])
        best = total - cur_total

        while right < len(data):
            cur_total -= data[left]
            cur_total += data[right]
            best = min(total - cur_total, best)
            left += 1
            right += 1

        return best

    def evaluateTree(self, root: Optional[TreeNode]) -> bool:
        # 2 = OR, 3 = AND
        if not root:
            return False
        val = root.val
        if val == 0:
            return False
        if val == 1:
            return True
        left = self.evaluateTree(root.left)
        right = self.evaluateTree(root.right)
        if val == 2:
            return left or right
        return left and right

    # 2161. Partition Array According to Given Pivot
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        left = cnt = 0
        N = len(nums)

        for i in range(N):
            n = nums[i]
            if n < pivot:
                nums[left] = n
                left += 1
            elif n > pivot:
                nums.append(n)
            else:
                cnt += 1

        for i in range(left, left + cnt):
            nums[i] = pivot

        return nums[: left + cnt] + nums[N:]

    # 1209. Remove All Adjacent Duplicates in String II
    def removeDuplicates(self, s: str, k: int) -> str:
        cnt = 1
        prev = s[0]
        stack = [prev]
        for w in s[1:]:
            stack.append(w)
            if w == prev:
                cnt += 1
                if cnt == k:
                    for _ in range(k):
                        stack.pop()
                    cnt = 0
                    if stack:
                        cnt = 1
                        prev = stack[-1]
                        idx = len(stack) - 1
                        while idx > 0 and stack[idx] == stack[idx - 1]:
                            # instead, one can store freq in the stack
                            idx -= 1
                            cnt += 1
            else:
                prev = w
                cnt = 1
        return "".join(stack)

    # 1780. Check if Number is a Sum of Powers of Three
    def checkPowersOfThree(self, n: int) -> bool:
        while n > 0:
            if n % 3 == 2:
                return False
            n //= 3
        return True

    # 214. Shortest Palindrome
    def shortestPalindrome(self, s: str) -> str:
        rev = s[::-1]
        if rev == s:
            return s
        N = len(s)
        if N <= 1:
            return s
        max_val = N
        is_even = True
        if N % 2 == 1:
            max_val -= 1
            is_even = False

        @cache
        def dp(start, end):
            if start == -1:
                return N - (end)
            if s[start] != s[end]:
                return max_val
            if start == 0:
                return N - (end) - 1
            res = min(
                dp(start - 1, end + 1), dp(start - 1, end - 1), dp(start - 1, end)
            )
            return res

        mid = N // 2
        if is_even:
            res = dp(mid - 1, mid - 1)
        else:
            res = dp(mid, mid)
        dp.cache_clear()

        if res == 0:
            return s
        return rev[: N - res] + s

    # 2579. Count Total Number of Colored Cells
    def coloredCells(self, n: int) -> int:
        # 0 4 8 12
        # 1 2 3 4
        cur = 0
        total = 1
        for _ in range(n - 1):
            cur += 4
            total += cur
        return total

    def uniqueOccurrences(self, arr: List[int]) -> bool:
        cnt = Counter(arr)
        nset = set()
        for k, v in cnt.items():
            if v in nset:
                return False
            nset.add(v)
        return True

    # 1769. Minimum Number of Operations to Move All Balls to Each Box
    def minOperations(self, boxes: str) -> List[int]:
        N = len(boxes)
        ans = []
        ball = total = 0

        for n in boxes:
            total += ball
            ans.append(total)
            if n == "1":
                ball += 1

        ball = total = 0
        idx = N - 1
        for n in boxes[::-1]:
            total += ball
            ans[idx] += total
            if n == "1":
                ball += 1

        return ans

    # 2965. Find Missing and Repeated Values
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        N = len(grid)
        n2 = N * N
        total = n2 * (n2 + 1) // 2
        seen = set()
        missing = 0
        for r in range(N):
            for c in range(N):
                val = grid[r][c]
                if missing == 0:
                    if val in seen:
                        missing = val
                        continue
                    else:
                        seen.add(val)
                total -= val
        return [missing, total]

    # 289. Game of Life
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        directions = (
            (0, 1),
            (1, 0),
            (-1, 0),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
        R = len(board)
        C = len(board[0])
        changes = []

        def check(r, c, cur):
            total = 0
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < R and 0 <= nc < C:
                    total += board[nr][nc]
                    if total > 3:
                        return 0
            if cur == 0:
                if total == 3:
                    return 1
                return 0
            if total < 2:
                return 0
            return 1

        for r in range(R):
            for c in range(C):
                val = board[r][c]
                res = check(r, c, val)
                if res != val:
                    changes.append((r, c, res))

        for r, c, val in changes:
            board[r][c] = val

    # 1506. Find Root of N-Ary Tree
    def findRoot(self, tree: List["Node"]) -> "Node":
        visited = set()

        def dfs(node):
            for ch in node.children:
                if ch.val not in visited:
                    visited.add(ch.val)
                    dfs(ch)

        for node in tree:
            if node not in visited:
                dfs(node)

        for node in tree:
            if node.val not in visited:
                return node

    # 224. Basic Calculator
    def calculate(self, s: str) -> int:

        def find(i):
            idx = i
            total = 0
            tsign = 1
            while idx < N and s[idx] != ")":

                while idx < N and s[idx] == " ":
                    idx += 1
                if idx == N:
                    return total

                if s[idx] == "-":
                    tsign = -1
                    idx += 1
                    continue
                elif s[idx] == "+":
                    idx += 1
                    tsign = 1
                    continue

                if s[idx] == "(":
                    ttotal, nidx = find(idx + 1)
                    idx = nidx
                    total += ttotal * tsign
                else:
                    num = []
                    while idx < N and s[idx].isdigit():
                        num.append(s[idx])
                        idx += 1
                    if num:
                        n = int("".join(num)) * tsign
                        total += n
            return total, idx + 1

        N = len(s)
        total = idx = 0
        sign = 1
        while idx < N:

            while idx < N and s[idx] == " ":
                idx += 1
            if idx == N:
                return total

            if s[idx] == "-":
                sign = -1
                idx += 1
                continue
            elif s[idx] == "+":
                idx += 1
                sign = 1
                continue

            if s[idx] == "(":
                ntotal, nidx = find(idx + 1)
                total += ntotal * sign
                idx = nidx
            else:
                num = []
                while idx < N and s[idx].isdigit():
                    num.append(s[idx])
                    idx += 1
                if num:
                    n = int("".join(num)) * sign
                    total += n
        return total

    # 2523. Closest Prime Numbers in Range
    def closestPrimes(self, left: int, right: int) -> List[int]:

        def get_all_primes(n):
            primes = [True] * (n + 1)
            primes[0] = primes[1] = False
            p = 2
            while p * p <= n:
                if primes[p]:
                    for i in range(p * p, n + 1, p):
                        primes[i] = False
                p += 1
            return [i for i, is_prime in enumerate(primes) if is_prime]

        primes = get_all_primes(right)
        start = bisect.bisect_left(left)
        small, ans = inf, (-1, -1)
        for idx in range(start, len(primes) - 1):
            l = primes[idx]
            r = primes[idx + 1]
            if (r - l) < small:
                ans = (l, r)
                small = r - l
        return list(ans)

    # 852. Peak Index in a Mountain Array
    def peakIndexInMountainArray(self, arr: List[int]) -> int:
        N = len(arr)
        low, high = 0, N - 1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] < arr[mid + 1]:
                low = mid + 1
            else:
                high = mid - 1
        return low

    # 1095. Find in Mountain Array
    def findInMountainArray(self, target: int, mountainArr: "MountainArray") -> int:
        N = mountainArr.length()
        low, high = 0, N - 1
        while low <= high:
            mid = (low + high) // 2
            if mountainArr.get(mid) < mountainArr.get(mid + 1):
                low = mid + 1
            else:
                high = mid - 1
        peak = low

        big = mountainArr.get(peak)
        if big < target:
            return -1
        elif big == target:
            return peak

        def bsearch(low, high, sign):
            while low <= high:
                mid = (low + high) // 2
                val = mountainArr.get(mid) * sign
                if val < target * sign:
                    low = mid + 1
                elif val > target * sign:
                    high = mid - 1
                else:
                    return mid
            return -1

        left = bsearch(0, peak, 1)
        if left != -1:
            return left
        return bsearch(peak, N - 1, -1)

    # 1110. Delete Nodes And Return Forest
    def delNodes(
        self, root: Optional[TreeNode], to_delete: List[int]
    ) -> List[TreeNode]:
        ans = []
        nset = set(to_delete)

        def dfs(node, is_del, parent):
            if not node:
                return
            if node.val in nset:
                nset.remove(node.val)
                dfs(node.left, True, node)
                dfs(node.right, True, node)
                if parent is None:
                    return
                elif parent.left is node:
                    parent.left = None
                elif parent.right is node:
                    parent.right = None
            else:
                if is_del:
                    ans.append(node)
                dfs(node.left, False, node)
                dfs(node.right, False, node)

        dfs(root, True, None)
        return ans

    # 354. Russian Doll Envelopes
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort()
        N = len(envelopes)

        @cache
        def dp(idx):
            if idx == N:
                return 0
            cur_w = envelopes[idx][0]
            small_h = envelopes[idx][1]
            nidx = i = idx
            while i + 1 < N and envelopes[i + 1][0] == cur_w:
                i += 1
                if envelopes[i][1] < small_h:
                    small_h = envelopes[i][1]
                    nidx = i
            res = 0
            nidx = i + 1
            for j in range(nidx, N):
                nh = envelopes[j][1]
                if nh > small_h:
                    res = max(dp(j), res)
            return res + 1

        return dp(0)

    # 2379. Minimum Recolors to Get K Consecutive Black Blocks
    def minimumRecolors(self, blocks: str, k: int) -> int:
        counts = Counter(blocks[:k])
        left = 0
        res = counts["W"]
        if res == 0:
            return 0
        for right in range(k, len(blocks)):
            counts[blocks[left]] -= 1
            counts[blocks[right]] += 1
            res = min(counts["W"], res)
            left += 1
        return res

    # 1100. Find K-Length Substrings With No Repeated Characters
    def numKLenSubstrNoRepeats(self, s: str, k: int) -> int:
        ans = right = left = 0
        wset = set()
        N = len(s)
        while right < N:
            cur = s[right]
            while cur in wset:
                wset.remove(s[left])
                left += 1
            wset.add(cur)
            if right - left + 1 == k:
                ans += 1
                wset.remove(s[left])
                left += 1
            right += 1
        return ans

    def calculateTime(self, keyboard: str, word: str) -> int:
        time = 0
        idx = 0
        for i, w in enumerate(word):
            nxt = keyboard.index(w)
            time += abs(idx - nxt)
            idx = nxt
        return time

    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        queue = deque([root])
        max_sum = root.val
        level = max_level = 1

        while queue:
            size = len(queue)
            cur_sum = 0
            for _ in range(size):
                cur = queue.popleft()
                cur_sum += cur.val
                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)
            if cur_sum > max_sum:
                max_level = level
                max_sum = cur_sum
            level += 1
        return max_level

    def oddEvenJumps(self, arr: List[int]) -> int:
        """ "
        odd -> idx with smallest to the right
        even -> idx with biggest to the left
        """
        N = len(arr)
        next_odd = [-1] * N
        next_even = [-1] * N

        indices = sorted(range(N), key=lambda x: (arr[x], x))
        stack = []
        for i in indices:
            while stack and stack[-1] < i:
                next_odd[stack.pop()] = i
            stack.append(i)

        indices = sorted(range(N), key=lambda x: (-arr[x], x))
        stack = []
        for i in indices:
            while stack and stack[-1] < i:
                next_even[stack.pop()] = i
            stack.append(i)

        @cache
        def find_odd(idx):
            return next_odd[idx]

        @cache
        def find_even(idx):
            return next_even[idx]

        @cache
        def recursion(idx, is_even):
            if idx == N - 1:
                return 1
            if is_even:
                nidx = find_even(idx)
                if nidx == -inf or nidx == inf or nidx == -1:
                    return 0
                return recursion(nidx, False)
            else:
                nidx = find_odd(idx)
                if nidx == -inf or nidx == inf or nidx == -1:
                    return 0
                return recursion(nidx, True)

        ans = 0
        for i in range(N):
            ans += recursion(i, False)
        recursion.cache_clear()
        find_even.cache_clear()
        find_odd.cache_clear()
        return ans

    # 3208. Alternating Groups II
    def numberOfAlternatingGroups(self, colors: List[int], k: int) -> int:
        N = len(colors)
        arr = colors + colors[:k]
        left = cnt = 0
        prev = arr[0]
        for right in range(1, N + k):
            if prev != arr[right]:
                diff = right - left + 1
                if diff == k:
                    cnt += 1
                    left += 1
            else:
                left = right
            if left == N:
                break
            prev = arr[right]
        return cnt


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
