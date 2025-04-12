from typing import *

# 3513. Number of Unique XOR Triplets I


class Solution:
    """
    You are given an integer array nums of length n, where nums is a permutation of the numbers in the range [1, n].

    A XOR triplet is defined as the XOR of three elements nums[i] XOR nums[j] XOR nums[k] where i <= j <= k.

    Return the number of unique XOR triplet values from all possible triplets (i, j, k).

    """

    def uniqueXorTriplets(self, nums: List[int]) -> int:
        N = len(nums)
        nset = set()
        top = 0
        for i in range(1, N + 1):
            xn = N ^ i
            top = max(xn, top)
        start = 0 if N >= 3 else 1
        for i in range(start, N + 1):
            nset.add(i ^ top)
        nset.update(nums)
        res = len(nset)
        if N >= 3 and 0 not in nset:
            res += 1
        return res

    def uniqueXorTriplets(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n
        # 2^bit_len
        return 1 << (n.bit_length())

    def uniqueXorTriplets(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n
        s = set(nums)
        for i in range(32 - 1, -1, -1):
            if 1 << i in s:
                return 1 << (i + 1)

    def uniqueXorTriplets(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n
        i = 1
        while i <= n:
            i = i * 2
        return i

    def uniqueXorTriplets(self, nums: List[int]) -> int:
        n = len(nums)
        bits = [2**i for i in range(33)]
        if n <= 2:
            return n
        for i in bits:
            if i > n:
                return i


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
