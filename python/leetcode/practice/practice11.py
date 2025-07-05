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

    # 2021. Brightest Position on Street
    def brightestPosition(self, lights: List[List[int]]) -> int:
        table = SortedDict()
        for pos, r in lights:
            table[pos + r + 1] = table.get(pos + r + 1, 0) - 1
            table[pos - r] = table.get(pos - r, 0) + 1
        best = idx = total = 0
        for k, v in table.items():
            total += v
            if total > best:
                idx = k
                best = total
        return idx

    # 3169. Count Days Without Meetings
    def countDays(self, days: int, meetings: List[List[int]]) -> int:  # TLE
        table = defaultdict(int)
        for s, e in meetings:
            table[s] += 1
            table[e + 1] -= 1
        ans = total = 0
        for i in range(1, days + 1):
            if i in table:
                total += table[i]
            if total == 0:
                ans += 1
        return ans

    # 3169. Count Days Without Meetings
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        table = defaultdict(int)
        for s, e in meetings:
            table[s] += 1
            table[e + 1] -= 1
        ans = total = 0
        prev = 1
        timeline = sorted(table.items(), key=lambda x: x[0])
        for k, v in timeline:
            total += v
            if total == 0:
                prev = k
            else:
                if prev > 0:
                    ans += k - prev
                    prev = -1
        return ans + days - timeline[-1][0] + 1

    def minOperations(self, grid: List[List[int]], x: int) -> int:
        count = len(grid) * len(grid[0])
        if count == 1:
            return 0

        arr = []
        for row in grid:
            arr.extend(row)

        arr.sort()
        mid = arr[count // 2]
        ans = 0
        for n in arr:
            if n % x != mid % x:
                return -1
            ans += abs(mid - n) // x
        return ans

    # 2780. Minimum Index of a Valid Split
    def minimumIndex(self, nums: List[int]) -> int:
        N = len(nums)
        dom, vote = None, 0
        for n in nums:
            if dom == n:
                vote += 1
            elif vote == 0:
                dom = n
                vote = 1
            else:
                vote -= 1

        cnt = nums.count(dom)
        vote1 = 0
        for i, n in enumerate(nums):
            if n == dom:
                vote1 += 1
                cnt -= 1
            else:
                vote1 -= 1
            if vote1 > 0 and cnt > ((N - i - 1) // 2):
                return i
        return -1

    # 2503. Maximum Number of Points From Grid Queries
    def maxPoints(self, grid: List[List[int]], queries: List[int]) -> List[int]:
        R, C = len(grid), len(grid[0])
        directions = ((0, 1), (1, 0), (-1, 0), (0, -1))
        nqueries = [(n, i) for i, n in enumerate(queries)]
        nqueries.sort()
        choices = set([(0, 0)])

        def dfs(val, r, c, nset):
            if grid[r][c] >= val:
                return 0
            points = 0
            for dr, dc in directions:
                nr, nc = dr + r, dc + c
                dest = (nr, nc)
                if (
                    0 <= nr < R
                    and 0 <= nc < C
                    and grid[nr][nc] > 0
                    and dest not in nset
                ):
                    if grid[nr][nc] < val:
                        grid[nr][nc] = 0
                        points += dfs(val, nr, nc, nset) + 1
                    else:
                        nset.add(dest)
            return points

        ans = [0] * len(queries)
        points = 0
        prev = -1
        for val, idx in nqueries:
            if val == prev:
                ans[idx] = points
                continue
            removes = []
            nset = set()
            for r, c in choices:
                if grid[r][c] > 0:
                    if grid[r][c] < val:
                        grid[r][c] = 0
                        points += dfs(val, r, c, nset) + 1
                        removes.append((r, c))
                else:
                    removes.append((r, c))
            choices.update(nset)
            choices.difference_update(removes)
            prev = val
            ans[idx] = points

        return ans

    # 2818. Apply Operations to Maximize Score
    def maximumScore(self, nums: List[int], k: int) -> int:
        # WRONG - dont get it
        MOD = 10**9 + 7
        N = len(nums)
        top = max(nums)
        primes = [True] * (top + 1)
        primes[1], primes[0] = False, False
        p = 2
        while p * p <= top:
            if primes[p]:
                for i in range(p * p, top + 1, p):
                    primes[i] = False
            p += 1

        nprimes = [i for i, n in enumerate(primes) if n]

        @cache
        def prime_count(num):
            if num <= 1:
                return 0
            res = 0
            for n in nprimes:
                if num % n == 0:
                    while num % n == 0:
                        num //= n
                        res += 1
                    break
            return prime_count(num) + res

        points = []
        heap = []
        for i, n in enumerate(nums):
            heap.append((-n, i))
            points.append((prime_count(n), i))
        used = set()
        heapq.heapify(heap)
        points.sort()
        ans = 1
        while heap and k > 0:
            n, idx = heapq.heappop(heap)
            if idx in used:
                continue
            left = bisect.bisect_left(points, idx, key=lambda x: x[1])
            right = bisect.bisect_right(points, idx, key=lambda x: x[1])
            points = points[:left] + points[right:]
            used.update([i for i in range(left, right)])
            cnt = (right - left) * (right - left + 1) // 2
            ans = abs(ans * n ** min(cnt, k)) % MOD
            k -= cnt

        return ans % MOD

    # 1055. Shortest Way to Form String
    def shortestWay(self, source: str, target: str) -> int:
        S, T = len(source), len(target)
        tidx, sidx, cnt = 0, 0, 1
        while tidx < T:
            w = target[tidx]
            if source[sidx] == w:
                sidx += 1
                tidx += 1
                if sidx == S:
                    sidx = 0
                    cnt += 1
                continue
            start = sidx
            while sidx < S and source[sidx] != w:
                sidx += 1
            if sidx == S:
                if start == 0:
                    return -1
                sidx = 0
                cnt += 1
                continue
            tidx += 1
        return cnt

    # 763. Partition Labels
    def partitionLabels(self, s: str) -> List[int]:
        counts, hset = Counter(s), set()
        res = []
        left = 0
        for right, w in enumerate(s):
            counts[w] -= 1
            if counts[w] >= 1:
                hset.add(w)
            elif counts[w] == 0:
                hset.discard(w)
            if len(hset) == 0:
                res.append(right - left + 1)
                left = right + 1
        return res

    # 3503. Longest Palindrome After Substring Concatenation I
    def longestPalindrome(self, s: str, t: str) -> int:
        S, T = len(s), len(t)
        ans = 1
        for sl in range(S - 1):
            for sr in range(sl, S):
                w = s[sl : sr + 1]
                for tl in range(T - 1):
                    for tr in range(tl, T):
                        w2 = t[tl : tr + 1]
                        word = w + w2
                        if word == word[::-1]:
                            ans = max(len(word), ans)
        return ans

    # 2551. Put Marbles in Bags
    def putMarbles(self, weights: List[int], k: int) -> int:
        n = len(weights)
        if k == 1:
            return 0

        diffs = [weights[i - 1] + weights[i] for i in range(1, n)]
        diffs.sort()

        base = weights[0] + weights[-1]
        min_cost = base + sum(diffs[: k - 1])
        max_cost = base + sum(diffs[-(k - 1) :])
        return max_cost - min_cost

    # 2140. Solving Questions With Brainpower
    def mostPoints(self, questions: List[List[int]]) -> int:
        N = len(questions)

        @cache
        def dp(idx):
            if idx >= N:
                return 0
            point, skip = questions[idx]
            res = dp(idx + 1 + skip) + point
            res = max(res, dp(idx + 1))
            return res

        return dp(0)

    # 2873. Maximum Value of an Ordered Triplet I
    def maximumTripletValue(self, nums: List[int]) -> int:
        N = len(nums)
        ans = 0
        for i in range(N - 2):
            for j in range(i + 1, N - 1):
                for k in range(j + 1, N):
                    ans = max(ans, (nums[i] - nums[j]) * nums[k])
        return ans

    # 2874. Maximum Value of an Ordered Triplet II
    def maximumTripletValue(self, nums: List[int]) -> int:
        N = len(nums)
        left = [0] * N
        right = [0] * N
        for i in range(1, N):
            left[i] = max(left[i - 1], nums[i - 1])
            right[N - 1 - i] = max(right[N - i], nums[N - i])
        ans = 0
        for i in range(1, N - 1):
            ans = max(ans, (left[i] - nums[i] * right[i]))
        return ans

    def lcaDeepestLeaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return root

        def dfs(node, level):
            if not node:
                return 0
            if not node.left and not node.right:
                return node, level

            left, left_lvl = dfs(node.left, level + 1)
            right, right_lvl = dfs(node.right, level + 1)
            if left_lvl > right_lvl:
                return left, left_lvl
            if left_lvl < right_lvl:
                return right, right_lvl
            return node, right_lvl

        return dfs(root, 0)

    def getMoneyAmount(self, n: int) -> int:

        @cache
        def dp(low, high):
            if low >= high:
                return 0
            res = float("inf")

            for x in range(low, high + 1):
                cost = x + max(dp(low, x - 1), dp(x + 1, high))
                res = min(res, cost)
            return res

        return dp(1, n)

    # 1863. Sum of All Subset XOR Totals
    def subsetXORSum(self, nums: List[int]) -> int:
        self.total = 0
        N = len(nums)

        def backtrack(arr, idx):
            xors = 0
            for n in arr:
                xors ^= n
            self.total += xors
            if idx == N:
                return
            for i in range(idx, N):
                arr.append(nums[i])
                backtrack(arr, i + 1)
                arr.pop()

        backtrack([], 0)
        return self.total

    # 368. Largest Divisible Subset
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        if not nums:
            return []

        nums.sort()
        n = len(nums)
        dp = [[num] for num in nums]

        for i in range(n):
            for j in range(i):
                if nums[i] % nums[j] == 0 and len(dp[j]) + 1 > len(dp[i]):
                    dp[i] = dp[j] + [nums[i]]

        return max(dp, key=len)

    def countSymmetricIntegers(self, low: int, high: int) -> int:
        res = 0
        n = low
        while n <= high:
            while len(str(n)) % 2 == 1:
                n += 1
            if n <= high:
                st = str(n)
                size = len(st)
                s, e = 0, size - 1
                total = 0
                while s < e:
                    total += int(st[s]) - int(st[e])
                    s += 1
                    e -= 1
                if total == 0:
                    res += 1
            n += 1
        return res

    # 3272. Find the Count of Good Integers
    def countGoodIntegers(self, n: int, k: int) -> int:  # TLE
        start = 10 ** (n - 1)
        end = start * 10
        table = defaultdict(int)
        pset = set()

        def get_key(snum):
            return "".join(sorted(snum))

        def is_palin(snum):
            left, right = 0, len(snum) - 1
            while left < right:
                if snum[left] != snum[right]:
                    return False
                left += 1
                right -= 1
            return True

        cnt = 0
        for i in range(start, end):
            snum = str(i)
            key = get_key(snum)
            if i % k == 0:
                if key in pset or is_palin(snum):
                    pset.add(key)
                    cnt += 1
                else:
                    table[key] += 1
            else:
                table[key] += 1

        for key in pset:
            cnt += table[key]
        return cnt

    # 3272. Find the Count of Good Integers
    def countGoodIntegers(self, n: int, k: int) -> int:  # TLE
        start = 10 ** (n - 1)
        end = start * 10
        table = []
        pset = set()

        def get_key(snum):
            return "".join(sorted(snum))

        def is_palin(snum):
            left, right = 0, len(snum) - 1
            while left < right:
                if snum[left] != snum[right]:
                    return False
                left += 1
                right -= 1
            return True

        cnt = 0
        for i in range(start, end):
            snum = str(i)
            key = get_key(snum)
            if i % k == 0:
                if key in pset or is_palin(snum):
                    pset.add(key)
                    cnt += 1
                else:
                    table.append(key)
            else:
                table.append(key)

        for key in table:
            if key in pset:
                cnt += 1
        return cnt

    # 3522. Calculate Score After Performing Instructions
    def calculateScore(self, instructions: List[str], values: List[int]) -> int:
        visited = set()
        idx, N, score = 0, len(instructions), 0
        while 0 <= idx < N and idx not in visited:
            visited.add(idx)
            if instructions[idx] == "add":
                score += values[idx]
                idx += 1
            else:
                idx += values[idx]
        return score

    # 3523. Make Array Non-decreasing
    def maximumPossibleSize(self, nums: List[int]) -> int:
        stack = []
        for num in nums:
            if stack and num < stack[-1]:
                current_max = stack.pop()
                stack.append(current_max)
            else:
                stack.append(num)
        return len(stack)

    # 3527. Find the Most Common Response
    def findCommonResponse(self, responses: List[List[str]]) -> str:
        table = defaultdict(int)

        for i in range(len(responses)):
            rset = set(responses[i])
            for n in rset:
                table[n] += 1

        arr = [(-cnt, res) for res, cnt in table.items()]
        arr.sort()
        return arr[0][1]

    # 2918. Minimum Equal Sum of Two Arrays After Replacing Zeros
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1, sum2 = sum(nums1), sum(nums2)
        zero1, zero2 = nums1.count(0), nums2.count(0)
        if sum1 > sum2:
            big_s, small_s = sum1, sum2
            big_z, small_z = zero1, zero2
        elif sum2 > sum1:
            big_s, small_s = sum2, sum1
            big_z, small_z = zero2, zero1
        else:
            if zero1 != zero2 and (zero1 == 0 or zero2 == 0):
                return -1
            return sum1 + max(zero1, zero2)
        if small_z == 0:
            return -1
        if big_z == 0 and small_z + small_s > big_s:
            return -1

        return max(big_s + big_z, small_s + small_z)

    # 3541. Find Most Frequent Vowel and Consonant
    def maxFreqSum(self, s: str) -> int:
        v = {"a", "e", "i", "o", "u"}
        vfreq = [0] * 26
        cfreq = [0] * 26
        for w in s:
            if w in v:
                vfreq[ord("a") - ord(w)] += 1
            else:
                cfreq[ord("a") - ord(w)] += 1
        return max(cfreq) + max(vfreq)

    # 3542. Minimum Operations to Convert All Elements to Zero
    def minOperations(self, nums: List[int]) -> int:  # TLE
        cnt, N = 0, len(nums)
        used = set()
        snums = [(n, i) for i, n in enumerate(nums)]
        snums.sort()
        if snums[-1][0] == 0:
            return 0
        left = 0
        for n, i in snums:
            if n == 0:
                used.add(i)
            else:
                break
            left += 1
        while left < N:
            right = left + 1
            n, idx = snums[left][0], snums[left][1]
            used.add(idx)
            while right < N and snums[right][0] == n:
                ridx = snums[right][1]
                rleft, rright = idx + 1, ridx - 1
                while rleft <= rright:
                    if rleft in used or rright in used:
                        cnt += 1
                        break
                    rleft += 1
                    rright -= 1
                idx = ridx
                used.add(ridx)
                right += 1
            cnt += 1
            left = right
        return cnt

    # 3542. Minimum Operations to Convert All Elements to Zero
    def minOperations(self, nums: List[int]) -> int:
        cnt, stack = 0, []
        for n in nums:
            while stack and stack[-1] > n:
                stack.pop()
            if n > 0 and (not stack or stack[-1] < n):
                cnt += 1
                stack.append(n)
        return cnt

    # 3543. Maximum Weighted K-Edge Path
    def maxWeight(self, n: int, edges: List[List[int]], k: int, t: int) -> int:  # MLE
        if not edges and k == 0:
            return 0
        adj = defaultdict(list)
        for u, v, w in edges:
            adj[str(u)].append((w, str(v)))
        res, visited = [-1], set()

        def dfs(cur, path, total):
            if len(path) - 1 == k and total < t:
                res[0] = max(res[0], total)
            for w, nei in adj[cur]:
                if total + w < t and len(path) - 1 < k:
                    path.append(nei)
                    spath = "-".join(path)
                    if spath not in visited:
                        visited.add(spath)
                        dfs(nei, path, total + w)
                    path.pop()
                if k >= 1 and w < t:
                    npath = f"{cur}-{nei}"
                    if npath not in visited:
                        visited.add(npath)
                        dfs(nei, [cur, nei], w)
                if nei not in visited:
                    visited.add(nei)
                    dfs(nei, [nei], 0)

        for u, v, w in edges:
            u = str(u)
            if u not in visited:
                visited.add(u)
                dfs(u, [u], 0)
        return res[0]

    # 3543. Maximum Weighted K-Edge Path
    def maxWeight(self, n: int, edges: List[List[int]], k: int, t: int) -> int:  # pass
        if not edges and k == 0:
            return 0
        adj = defaultdict(list)
        indeg = [0] * n
        for u, v, w in edges:
            adj[u].append((w, v))
            indeg[v] += 1

        dq = deque()
        for i, n in enumerate(indeg):
            if n == 0:
                dq.append(i)
        torder = []
        while dq:
            cur = dq.popleft()
            torder.append(cur)
            for w, nei in adj[cur]:
                indeg[nei] -= 1
                if indeg[nei] == 0:
                    dq.append(nei)

        res = -1

        @cache
        def dfs(cur, total, cnt):
            if total < t and cnt == k:
                nonlocal res
                res = max(res, total)
            if total >= t or cnt > k:
                return
            for w, nei in adj[cur]:
                dfs(nei, w + total, cnt + 1)

        for start in torder:
            dfs(start, 0, 0)
        return res

    # 3545. Minimum Deletions for At Most K Distinct Characters
    def minDeletion(self, s: str, k: int) -> int:
        counts = Counter(s)
        size, cnt = len(counts), 0
        sarr = [n for n in counts.values()]
        sarr.sort(reverse=True)
        while size > k:
            cnt += sarr.pop()
            size -= 1
        return cnt

    # 3546. Equal Sum Grid Partition I
    def canPartitionGrid(self, grid: List[List[int]]) -> bool:
        R, C = len(grid), len(grid[0])
        total = rtotal = ctotal = 0
        for r in grid:
            total += sum(r)
        cols = [0] * C
        for r in range(R):
            if total - rtotal == rtotal:
                return True
            for c in range(C):
                rtotal += grid[r][c]
                cols[c] += grid[r][c]

        for i, c in enumerate(cols):
            if total - ctotal == ctotal:
                return True
            ctotal += c
        return False

    # 75. Sort Colors
    def sortColors(self, nums: List[int]) -> None:
        red = white = 0
        for n in nums:
            if n == 0:
                red += 1
            elif n == 1:
                white += 1
        for i in range(len(nums)):
            color = 2
            if red:
                color = 0
                red -= 1
            elif white:
                color = 1
                white -= 1
            nums[i] = color
        return nums

    # 1004. Max Consecutive Ones III
    def longestOnes(self, nums: List[int], k: int) -> int:
        best = zeros = left = 0
        N = len(nums)
        for right in range(N):
            if nums[right] == 0:
                zeros += 1
                while left < N and zeros > k:
                    if nums[left] == 0:
                        zeros -= 1
                    left += 1
            if zeros <= k:
                best = max(right - left + 1, best)
        return best

    # 198. House Robber
    def rob(self, nums: List[int]) -> int:
        N = len(nums)

        @cache
        def dp(idx, robbed):
            if idx >= N:
                return 0
            if robbed:
                res = dp(idx + 1, False)
            else:
                res = max(dp(idx + 1, True) + nums[idx], dp(idx + 1, False))
            return res

        return dp(0, False)

    # 844. Backspace String Compare
    def backspaceCompare(self, s: str, t: str) -> bool:
        sstack, tstack = [], []
        for w in s:
            if w == "#":
                if sstack:
                    sstack.pop()
            else:
                sstack.append(w)
        for w in t:
            if w == "#":
                if tstack:
                    tstack.pop()
            else:
                tstack.append(w)
        return "".join(sstack) == "".join(tstack)

    # 222. Count Complete Tree Nodes
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        dq = deque([root])
        cnt = 1
        while dq:
            cur = dq.popleft()
            if cur.left:
                cnt += 1
                dq.append(cur.left)
            if cur.right:
                cnt += 1
                dq.append(cur.right)
        return cnt

    # 562. Longest Line of Consecutive One in Matrix
    def longestLine(self, mat: List[List[int]]) -> int:
        R, C = len(mat), len(mat[0])
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

        @cache
        def dfs(r, c, d):
            dir = directions[d]
            res = 1
            nr = dir[0] + r
            nc = dir[1] + c
            if 0 <= nr < R and 0 <= nc < C:
                if mat[nr][nc] == 1:
                    res += dfs(nr, nc, d)
            return res

        res = 0

        for r in range(R):
            for c in range(C):
                if mat[r][c] == 1:
                    for i in range(len(directions)):
                        res = max(dfs(r, c, i), res)
        return res

    # 3552. Grid Teleportation Traversal
    def minMoves(self, matrix: List[str]) -> int:
        # tried bfs, but got TLE - not sure why
        directions = ((0, 1), (1, 0), (-1, 0), (0, -1))
        R, C = len(matrix), len(matrix[0])
        table = defaultdict(list)
        used = [False] * 26
        distances = [[inf] * C for _ in range(R)]

        for r in range(R):
            for c in range(C):
                w = matrix[r][c]
                if w.isalpha():
                    table[w].append((r, c))

        heap = [(0, 0, 0, [False] * 26)]
        while heap:
            dist, r, c, used = heapq.heappop(heap)
            if r == R - 1 and c == C - 1:
                return dist
            val = matrix[r][c]
            if val.isalpha():
                idx = ord(val) - ord("A")
                if not used[idx]:
                    cused = used.copy()
                    cused[idx] = True
                    for tr, tc in table[val]:
                        if not (r == tr and c == tc):
                            if distances[tr][tc] > dist:
                                distances[tr][tc] = dist
                                heapq.heappush(heap, (dist, tr, tc, cused))
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < R and 0 <= nc < C:
                    val = matrix[nr][nc]
                    if val != "#" and distances[nr][nc] > dist + 1:
                        distances[nr][nc] = dist + 1
                        heapq.heappush(heap, (dist + 1, nr, nc, used))
        return -1

    # 3556. Sum of Largest Prime Substrings
    def sumOfLargestPrimes(self, s: str) -> int:
        def is_prime(n):
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

        primes = set()
        N = len(s)
        for i in range(N):
            if s[i] == "0":
                continue
            for j in range(i, N):
                num = int(s[i : j + 1])
                if is_prime(num):
                    primes.add(num)
        primes = list(primes)
        primes.sort(reverse=True)
        return sum(primes[:3])

    # 3557. Find Maximum Number of Non Intersecting Substrings
    def maxSubstrings(self, word: str) -> int:
        N = len(word)

        @cache
        def dp(idx):
            if idx >= N:
                return 0
            cur = word[idx]
            res = dp(idx + 1)
            for j in range(idx + 3, N):
                if word[j] == cur:
                    res = max(dp(j + 1) + 1, res)
                    break
            return res

        return dp(0)

    # 3558. Number of Ways to Assign Edge Weights I
    def assignEdgeWeights(self, edges: List[List[int]]) -> int:
        MOD = 10**9 + 7
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        visited = set([1])
        dq = deque([(1, 0)])
        path = 0
        while dq:
            cur, dist = dq.popleft()
            path = max(dist, path)
            for nei in adj[cur]:
                if nei not in visited:
                    visited.add(nei)
                    dq.append((nei, dist + 1))

        @cache
        def dp(idx, is_even):
            if idx == path:
                if not is_even:
                    return 1
                return 0
            res = dp(idx + 1, not is_even) + dp(idx + 1, is_even)
            return res % MOD

        return (dp(1, True) + dp(1, False)) % MOD

    # 2131. Longest Palindrome by Concatenating Two Letter Words
    def longestPalindrome(self, words: List[str]) -> int:
        sames, opps = False, 0
        counts = Counter(words)
        for w, cnt in counts.items():
            if cnt == 0:
                continue
            rev = w[::-1]
            if rev == w:
                opps += (cnt // 2) * 4
                if cnt % 2 == 1:
                    sames = True
            else:
                if rev in counts and counts[rev] > 0:
                    rcnt = min(cnt, counts[rev])
                    counts[rev] -= rcnt
                    counts[w] -= rcnt
                    opps += (rcnt * 2) * 2
        if sames:
            opps += 2
        return opps

    # 2131. Longest Palindrome by Concatenating Two Letter Words
    def longestPalindrome(self, words: List[str]) -> int:
        counts = defaultdict(int)
        cnt = 0
        for w in words:
            rev = w[::-1]
            if rev in counts and counts[rev] > 0:
                cnt += 4
                counts[rev] -= 1
            else:
                counts[w] += 1
        for w, v in counts.items():
            if v > 0 and w[0] == w[1]:
                cnt += 2
                break
        return cnt

    def minCuttingCost(self, n: int, m: int, k: int) -> int:
        big = max(n, m)
        if big <= k:
            return 0
        return (big - k) * k

    def resultingString(self, s: str) -> str:
        res = []
        for i in range(len(s)):
            ch1 = ord(s[i])
            if res:
                prev = ord(res[-1])
                diff = abs(prev - ch1)
                if diff == 1 or diff == 25:
                    res.pop()
                else:
                    res.append(s[i])
            else:
                res.append(s[i])
        return "".join(res)

    # 909. Snakes and Ladders
    def snakesAndLadders(self, board: List[List[int]]) -> int:  # tricky
        """board
        6 5 4
        1 2 3
        """
        R, C = len(board), len(board[0])
        target = R * C
        table, idx, opps = {}, 1, True
        for r in range(R - 1, -1, -1):
            if opps:
                for c in range(C):
                    table[idx] = (r, c)
                    idx += 1
            else:
                for c in range(C - 1, -1, -1):
                    table[idx] = (r, c)
                    idx += 1
            opps = not opps

        dq = deque([(0, 1)])
        visited = set([1])
        while dq:
            cnt, pos = dq.popleft()
            r, c = table[pos]
            val = board[r][c]
            if val != -1:
                pos = val
            if pos == target:
                return cnt
            nxt = pos
            for i in range(6):
                nxt += 1
                if nxt <= target:
                    if nxt not in visited:
                        dq.append((cnt + 1, nxt))
                        visited.add(nxt)
                else:
                    break
        return -1

    # 3572. Maximize Y‑Sum by Picking a Triplet of Distinct X‑Values
    def maxSumDistinctTriplet(
        self, x: List[int], y: List[int]
    ) -> int:  # O(NlogN) - 303ms
        arr = [(n, i) for i, n in enumerate(y)]
        arr.sort(reverse=True)  # O(NlogN)
        N = len(x)
        a = 0
        while a <= N - 2:  # O(3N)
            b = a + 1
            while b < N and x[arr[a][1]] == x[arr[b][1]]:
                b += 1
            c = b + 1
            while c < N and (
                x[arr[b][1]] == x[arr[c][1]] or x[arr[c][1]] == x[arr[a][1]]
            ):
                c += 1
            if c < N:
                return arr[a][0] + arr[b][0] + arr[c][0]
            a = b + 1
        return -1

    # 3572. Maximize Y‑Sum by Picking a Triplet of Distinct X‑Values
    def maxSumDistinctTriplet(
        self, x: List[int], y: List[int]
    ) -> int:  # O(NlogN) - 94ms
        nmap = defaultdict(int)
        for i, n in enumerate(x):
            if n not in nmap or nmap[n] < y[i]:
                nmap[n] = y[i]
        arr = sorted([val for val in nmap.values()], reverse=True)
        if len(arr) < 3:
            return -1
        return sum(arr[:3])

    # 3572. Maximize Y‑Sum by Picking a Triplet of Distinct X‑Values
    def maxSumDistinctTriplet(
        self, x: List[int], y: List[int]
    ) -> int:  # O(N + klogk) - 87ms
        nmap = defaultdict(int)
        for i, n in enumerate(x):
            if n not in nmap or nmap[n] < y[i]:
                nmap[n] = y[i]
        heap = [-val for val in nmap.values()]
        if len(heap) < 3:
            return -1
        heapq.heapify(heap)
        res = 0
        for i in range(3):
            res -= heapq.heappop(heap)
        return res

    # 3572. Maximize Y‑Sum by Picking a Triplet of Distinct X‑Values
    def maxSumDistinctTriplet(self, x: List[int], y: List[int]) -> int:  # O(N) - 152ms
        nmap = defaultdict(int)
        for i, n in enumerate(x):
            nmap[n] = max(nmap[n], y[i])
        if len(nmap) < 3:
            return -1
        a = b = c = -1
        for n in nmap.values():
            if n > a:
                c = b
                b = a
                a = n
            elif n > b:
                c = b
                b = n
            elif n > c:
                c = n
        return a + b + c

    # 347. Top K Frequent Elements
    def topKFrequent(self, nums, k):  # O(N * k) - new approach, good for small k
        counts = Counter(nums)
        cands = [(-inf, -1) for _ in range(k)]
        for key, n in counts.items():
            for i, c in enumerate(cands):
                if n > c[0]:
                    idx = k - 1
                    while idx > i:
                        cands[idx], cands[idx - 1] = cands[idx - 1], cands[idx]
                        idx -= 1
                    cands[i] = (n, key)
                    break
        res = [c[1] for c in cands]
        return res

    # 3588. Find Maximum Area of a Triangle
    def maxArea(self, coords: List[List[int]]) -> int:
        N = len(coords)
        xtable, ytable = defaultdict(list), defaultdict(list)
        xcoords, ycoords = sorted(coords), sorted([(y, x) for x, y in coords])

        for x, y in coords:
            xtable[x].append(y)
            ytable[y].append(x)

        def find(table, coord):
            best = 0
            for x, ylist in table.items():
                sy, by = min(ylist), max(ylist)
                base = by - sy
                i = 0
                x1 = coord[i][0]

                while i < N - 1 and x1 == x:
                    i += 1
                    x1 = coord[i][0]
                height = abs(x1 - x)
                best = max(base * height, best)

                i = N - 1
                x2 = coord[-1][0]
                while i > 0 and x2 == x:
                    i -= 1
                    x2 = coord[i][0]
                height = abs(x2 - x)
                best = max(base * height, best)
            return best

        res = max(find(xtable, xcoords, 0), find(ytable, ycoords, 1))
        if res == 0:
            return -1
        return res

    # 3481. Apply Substitutions
    def applySubstitutions(self, replacements: List[List[str]], text: str) -> str:
        dq, wmap = deque(replacements), {}
        while dq:
            key, val = dq.popleft()
            words = val.split("%")
            good = True
            for i in range(len(words)):
                if len(words[i]) == 1 and words[i].isupper():
                    if words[i] in wmap:
                        words[i] = wmap[words[i]]
                    else:
                        good = False
                        words[i] = f"%{words[i]}%"
            if not good:
                dq.append((key, "".join(words)))
            else:
                wmap[key] = "".join(words)

        tsplits, res = text.split("%"), []
        for val in tsplits:
            if val in wmap:
                res.append(wmap[val])
            else:
                res.append(val)
        return "".join(res)

    # 3597. Partition String
    def partitionString(self, s: str) -> List[str]:
        seen, cur, res = set(), "", []
        for i in range(len(s)):
            cur += s[i]
            if cur not in seen:
                res.add(cur)
                seen.add(cur)
                cur = ""
        return res

    # 3598. Longest Common Prefix Between Adjacent Strings After Removals
    def longestCommonPrefix(self, words: List[str]) -> List[int]:
        if len(words) == 1:
            return [0]

        memo = {}

        def prefix_len(str1, str2):
            w1 = str1 if str1 > str2 else str2
            w2 = str1 if str1 <= str2 else str2
            if (w1, w2) in memo:
                return memo[(w1, w2)]
            idx = 0
            while idx < len(w1) and idx < len(w2) and w1[idx] == w2[idx]:
                idx += 1
            memo[(w1, w2)] = idx
            return idx

        N = len(words)
        sizes, res = [], []
        for i in range(N - 1):
            size = prefix_len(words[i], words[i + 1])
            sizes.append((size, i, i + 1))
            if i == N - 2:
                sizes.append((size, i + 1, i))

        sizes.sort(reverse=True)
        best = 0
        for size, idx1, idx2 in sizes:
            if 0 != idx1 and 0 != idx2:
                best = size
                break
        res.append(best)

        for i in range(1, N - 1):
            best = prefix_len(words[i - 1], words[i + 1])
            if best >= sizes[0][0]:
                res.append(best)
            else:
                for size, idx1, idx2 in sizes:
                    if i != idx1 and i != idx2:
                        best = max(best, size)
                        break
                res.append(best)
        best = 0
        for size, idx1, idx2 in sizes:
            if N - 1 != idx1 and N - 1 != idx2:
                best = size
                break
        res.append(best)
        return res

    # 3603. Minimum Cost Path with Alternating Directions II
    def minCost(self, m: int, n: int, waitCost: List[List[int]]) -> int:  # dijkstra
        R, C = m, n
        directions = ((0, 1), (1, 0))
        costs = defaultdict(lambda: float("inf"))
        heap = [(1, 1, 0, 0)]

        while heap:
            cost, s, r, c = heapq.heappop(heap)
            if r == R - 1 and c == C - 1:
                return cost
            if s % 2 == 1:
                for dr, dc in directions:
                    nr, nc = dr + r, dc + c
                    if nr < R and nc < C:
                        ncost = cost + (nr + 1) * (nc + 1)
                        if costs[(nr, nc)] > ncost:
                            costs[(nr, nc)] = ncost
                            heapq.heappush(heap, (ncost, s + 1, nr, nc))
            else:
                ncost = cost + waitCost[r][c]
                heapq.heappush(heap, (ncost, s + 1, r, c))
        return -1

    # 3603. Minimum Cost Path with Alternating Directions II
    def minCost(self, m: int, n: int, waitCost: List[List[int]]) -> int:  # dp
        R, C = m, n

        @cache
        def dp(r, c, s):
            if r == R - 1 and c == C - 1:
                return 0
            res, cost = inf, 0
            if s % 2 == 0:
                cost = waitCost[r][c]
                s += 1
            if r + 1 < R:
                res = dp(r + 1, c, s + 1) + (r + 2) * (c + 1)
            if c + 1 < C:
                res = min(dp(r, c + 1, s + 1) + (r + 1) * (c + 2), res)
            return res + cost

        return dp(0, 0, 1) + 1

    # 3604. Minimum Time to Reach Destination in Directed Graph
    def minTime(self, n: int, edges: List[List[int]]) -> int:
        if not edges and n == 1:
            return 0
        adj = defaultdict(list)
        for u, v, s, e in edges:
            adj[u].append((v, s, e))

        times = defaultdict(lambda: inf)
        heap = [(0, 0)]
        while heap:
            t, cur = heapq.heappop(heap)
            if cur == n - 1:
                return t
            for nei, s, e in adj[cur]:
                if s > t and times[nei] > s:
                    times[nei] = s
                    heapq.heappush(heap, (s, nei))
                elif s <= t <= e and times[nei] > t + 1:
                    times[nei] = t + 1
                    heapq.heappush(heap, (t + 1, nei))

        return -1


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
