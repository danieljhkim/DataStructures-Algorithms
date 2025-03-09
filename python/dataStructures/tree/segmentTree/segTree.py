import math


class SegmentTreeRangeSum:
    """
    - range sum queries

    1. nums = [1, 2, 3, 4]
    2. init tree = [0 ,0, 0, 0, 1, 2, 3, 4]
    3. update tree = [0, 10, 3, 7, 1, 2, 3, 4]

    - even idx -> left child
    - odd idx -> right child
    """

    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [0] * (2 * self.n)

        # build
        for i in range(self.n):
            self.tree[self.n + i] = nums[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, idx, val):
        idx += self.n
        self.tree[idx] = val
        while idx > 1:
            idx //= 2
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def query(self, left, right):
        res = 0
        left += self.n
        right += self.n + 1

        while left < right:
            if left % 2 == 1:
                res += self.tree[left]
                left += 1
            if right % 2 == 1:
                res += self.tree[right - 1]
                right -= 1
            left //= 2
            right //= 2

        return res


class SegmentTreeFirstMaxVal:
    """
    - query left most val >= target
    """

    def __init__(self, arr):
        l = len(arr)
        # since the tree may not be balanced
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
                return idx
        return idx
