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


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
