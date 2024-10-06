from ast import List
from typing import Optional
from collections import defaultdict, Counter


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

    # 424. Longest Repeating Character Replacement
    def characterReplacement(self, s: str, k: int) -> int:
        n = len(s)
        ans = 0
        left = 0
        max_count = 0
        count = defaultdict(int)
        for right in range(n):
            count[s[right]] += 1
            max_count = max(count[s[right]], max_count)
            if (right - left + 1) - max_count > k:
                count[s[left]] -= 1
                left += 1
            ans = max(ans, right - left + 1)
        return ans

    # 567. Permutation in String
    def checkInclusion(self, s1: str, s2: str) -> bool:
        n = len(s1)
        if n > len(s2):
            return False
        s2_count = [0] * 26
        s1_count = [0] * 26
        for i in range(n):
            s2_count[ord(s2[i]) - ord("a")] += 1
            s1_count[ord(s1[i]) - ord("a")] += 1
        left = 0
        right = n - 1
        while right < len(s2):
            if s2_count == s1_count:
                return True
            s2_count[ord(s2[left]) - ord("a")] -= 1
            left += 1
            right += 1
            if right < len(s2):
                s2_count[ord(s2[right]) - ord("a")] += 1
        return s2_count == s1_count

    # 3. Longest Substring Without Repeating Characters
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans = 0
        nset = set()
        left = 0
        for right in range(len(s)):
            if s[right] in nset:
                ans = max(ans, len(nset))
                while s[right] in nset:
                    nset.remove(s[left])
                    left += 1
                nset.add(s[right])
            else:
                nset.add(s[right])
        return max(ans, len(nset))
