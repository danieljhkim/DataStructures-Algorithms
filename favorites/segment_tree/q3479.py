import math
from typing import *
from collections import Counter, deque

"""3479. Fruits Into Baskets III

You are given two arrays of integers, fruits and baskets, each of length n, where fruits[i] represents the quantity of the ith type of fruit, and baskets[j] represents the capacity of the jth basket.

From left to right, place the fruits according to these rules:

Each fruit type must be placed in the leftmost available basket with a capacity greater than or equal to the quantity of that fruit type.
Each basket can hold only one type of fruit.
If a fruit type cannot be placed in any basket, it remains unplaced.
Return the number of fruit types that remain unplaced after all possible allocations are made.

Constraints:
    - 1 <= n <= 10^5
    - 1 <= fruits[i], baskets[i] <= 10^9
"""


class SegTree:
    def __init__(self, arr):
        l = len(arr)
        self.n = 2 ** math.ceil(math.log2(l))
        self.tree = [0] * (self.n * 2)
        for i, val in enumerate(arr):
            self.update(i + self.n, val)

    def update(self, idx, val):
        self.tree[idx] = val
        while idx > 1:
            idx //= 2
            self.tree[idx] = max(self.tree[idx * 2], self.tree[idx * 2 + 1])

    def query(self, target):
        if self.tree[1] < target:
            return -1
        idx = 1
        while idx < self.n:
            left, right = idx * 2, idx * 2 + 1
            if self.tree[left] >= target:
                idx = left
            elif self.tree[right] >= target:
                idx = right
            else:
                return idx - self.n
        return idx - self.n


class Solution:
    # seg tree solution - O(n logn)
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        seg_tree = SegTree(baskets)
        ans = 0

        for i, n in enumerate(fruits):
            idx = seg_tree.query(n)
            if idx == -1:
                ans += 1
            else:
                seg_tree.update(idx, -1)
        return ans

    """
    Someone else's solution I found. 
    deque approach - O(n^2)
    shouldn't pass, but passes
    """

    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        ans = 0
        dq = deque()
        max_b = max(baskets)

        for b in baskets:
            dq.append(b)

        for fruit in fruits:
            if fruit > max_b:
                ans += 1
                continue
            buffer = []
            found = False
            while dq:
                basket = dq.popleft()
                if basket >= fruit:
                    found = True
                    break
                else:
                    buffer.append(basket)
            if not found:
                ans += 1
            while buffer:
                dq.appendleft(buffer.pop())
        return ans

    """
    Cool solution I found:
    sqrt decomposition approach - O(n sqrt(n))
    """

    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        n = len(fruits)
        ret = 0

        # sqrt decomposition
        bucket_sz = int(math.ceil(math.sqrt(n)))

        buckets = [[] for _ in range(bucket_sz)]
        for i, basket in enumerate(baskets):
            bucket_idx = i // bucket_sz
            buckets[bucket_idx].append((basket, i))

        for bucket in buckets:
            bucket.sort()

        for cnt in fruits:
            for bucket in buckets:
                if bucket and bucket[-1][0] >= cnt:
                    chosen = min((i, basket) for basket, i in bucket if basket >= cnt)
                    bucket.remove((chosen[1], chosen[0]))
                    break
            else:
                ret += 1

        return ret


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
