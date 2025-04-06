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


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
