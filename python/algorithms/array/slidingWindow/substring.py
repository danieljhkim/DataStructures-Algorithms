from collections import Counter, defaultdict


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        counter = Counter(t)
        left = 0
        table = defaultdict(list)
        for right in range(len(s)):

            while left < right and (s[left] not in counter or table[s[left]] > 1):
                if s[left] in counter and counter[s[left]] > 0:
                    counter[s[left]] -= 1
                left += 1
            letter = s[right]
            if letter in counter:
                table[letter] += 1
                counter[letter] -= 1
