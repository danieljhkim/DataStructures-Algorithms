"""operations

1. check if i-th bit is set
if mask & (1 << i):

2. set i-th bit
mask = mask | (1 << i)

3. clear i-th bit
mask = mask & ~(1 << i)

4. toggle i-th bit
mask = mask ^ (1 << i)

5. Check if subset is contained
subset = 0b00101  # {0, 2}
if (mask & subset) == subset:

6. Iterate through all subsets of a mask
submask = mask
while submask:
    print(bin(submask))
    submask = (submask - 1) & mask

7. Count set bits (population count)
count = bin(mask).count("1")
"""

from typing import *
from functools import cache
from collections import Counter, deque, defaultdict, OrderedDict


# 526. Beautiful Arrangement
class Q526:
    """
    Suppose you have n integers labeled 1 through n.
    A permutation of those n integers perm (1-indexed) is considered a beautiful arrangement if for every i (1 <= i <= n), either of the following is true:

    1. perm[i] is divisible by i.
    2. i is divisible by perm[i].

    Given an integer n, return the number of the beautiful arrangements that you can construct.
    """

    def countArrangement(self, n: int) -> int:
        # backtrack - notice here that we aren't able to utilize those already computed
        dq = deque([i for i in range(1, n + 1)])

        def backtrack(arr, dq):
            if len(arr) == n:
                return 1
            l, idx = len(dq), len(arr) + 1
            cnt = 0
            for _ in range(l):
                out = dq.popleft()
                if out % idx == 0 or idx % out == 0:
                    arr.append(out)
                    cnt += backtrack(arr, dq)
                    arr.pop()
                dq.append(out)
            return cnt

        return backtrack([], dq)

    def countArrangement(self, n: int) -> int:
        # bitmask dp - now we are able to cache
        @cache
        def dp(mask, idx):
            if idx == n + 1:
                return 1
            res = 0
            for i in range(n):
                if mask & (1 << i) == 0:
                    if (i + 1) % idx == 0 or idx % (i + 1) == 0:
                        mask = mask | (1 << i)
                        res += dp(mask, idx + 1)
                        mask = mask ^ (1 << i)
            return res

        mask = 1 << (n)
        return dp(mask, 1)


# 1066. Campus Bikes II
class Q1066:
    """
    On a campus represented as a 2D grid, there are n workers and m bikes, with n <= m. Each worker and bike is a 2D coordinate on this grid.

    We assign one unique bike to each worker so that the sum of the Manhattan distances between each worker and their assigned bike is minimized.

    Return the minimum possible sum of Manhattan distances between each worker and their assigned bike.

    The Manhattan distance between two points p1 and p2 is Manhattan(p1, p2) = |p1.x - p2.x| + |p1.y - p2.y|.
    """

    # backtrack - TLE
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        distances = []
        N = len(workers)

        for w in workers:
            dists = []
            for b in bikes:
                dist = abs(w[0] - b[0]) + abs(w[1] - b[1])
                dists.append(dist)
            distances.append(dists)

        self.ans = float("inf")

        def backtrack(w, b, total):
            if len(w) == N:
                self.ans = min(total, self.ans)
                return
            for i in range(N):
                if i not in w:
                    w.add(i)
                    for j in range(N):
                        if j not in b:
                            b.add(j)
                            backtrack(w, b, total + distances[i][j])
                            b.remove(j)
                    w.remove(i)

        backtrack(set(), set(), 0)
        return self.ans

    # bitmask dp
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        distances = []
        N, B = len(workers), len(bikes)

        for w in workers:
            dists = []
            for b in bikes:
                dist = abs(w[0] - b[0]) + abs(w[1] - b[1])
                dists.append(dist)
            distances.append(dists)

        @cache
        def dp(w_idx, b_mask):
            if w_idx == N:
                return 0
            res = float("inf")
            for j in range(B):
                if b_mask & (1 << j) == 0:
                    res = min(
                        dp(w_idx + 1, b_mask | (1 << j)) + distances[w_idx][j], res
                    )
            return res

        b_mask = 1 << (len(bikes) + 1)
        return dp(0, b_mask)
