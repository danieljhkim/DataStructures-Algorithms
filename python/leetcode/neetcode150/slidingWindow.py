from ast import List
from typing import Optional
from collections import defaultdict


class Solution:

    # 567. Permutation in String
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s2) < len(s1):
            return False
        count_2 = [0] * 26
        count_1 = [0] * 26
        r = len(s1)
        l = 1
        for i in range(0, r):
            pos_2 = ord(s2[i]) - ord("a")
            pos_1 = ord(s1[i]) - ord("a")
            count_2[pos_2] += 1
            count_1[pos_1] += 1

        if count_1 == count_2:
            return True

        while r < len(s2):
            # calculate
            count_2[ord(s2[l - 1]) - ord("a")] -= 1
            count_2[ord(s2[r]) - ord("a")] += 1
            if count_1 == count_2:
                return True
            r += 1
            l += 1
        return False
