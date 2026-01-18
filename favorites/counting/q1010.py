from typing import *

"""1010. Pairs of Songs With Total Durations Divisible by 60

You are given a list of songs where the ith song has a duration of time[i] seconds.
Return the number of pairs of songs for which their total duration in seconds is divisible by 60. 
Formally, we want the number of indices i, j such that i < j with (time[i] + time[j]) % 60 == 0.
"""


class Solution:
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
