from typing import *


class Solution:

    # 1358. Number of Substrings Containing All Three Characters

    # sliding window
    def numberOfSubstrings(self, s: str) -> int:
        N = len(s)
        ans = cnt = right = left = 0
        counts = [0, 0, 0]
        while right < N:
            while right < N and cnt < 3:
                w = ord(s[right]) - ord("a")
                counts[w] += 1
                if counts[w] == 1:
                    cnt += 1
                right += 1
            while left < right and cnt == 3:
                ans += N - right + 1  # this right here
                w = ord(s[left]) - ord("a")
                counts[w] -= 1
                if counts[w] == 0:
                    cnt -= 1
                    left += 1
                    break
                left += 1
        return ans

    # cool move
    def numberOfSubstrings(self, s: str) -> int:
        N = len(s)
        ans = 0
        counts = [-1, -1, -1]
        for i in range(N):
            w = ord(s[i]) - ord("a")
            counts[w] = i
            ans += 1 + min(counts)
        return ans
