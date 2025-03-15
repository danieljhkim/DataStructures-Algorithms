from typing import *


class Solution:

    # 1010. Pairs of Songs With Total Durations Divisible by 60
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        freq = [0] * 60
        ans = 0
        for t in time:
            rem = t % 60
            comp = (60 - rem) % 60
            ans += freq[comp]
            freq[rem] += 1
        return ans


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
