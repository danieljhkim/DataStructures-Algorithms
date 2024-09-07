from ast import List
from typing import Optional
from collections import defaultdict


class Solution:

    # 217. Contains Duplicate
    def containsDuplicate(self, nums: List[int]) -> bool:
        n_set = set()
        for num in nums:
            if num in n_set:
                return True
            else:
                n_set.add(num)
        return False

    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(set(nums)) < len(nums)

    # 242. Valid Anagram
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        nmap = defaultdict(int)
        for c in s:
            nmap[c] += 1
        for c in t:
            nmap[c] -= 1
        for n in nmap.values():
            if n != 0:
                return False
        return True

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = defaultdict(list)
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord("a")] += 1
            ans[tuple(count)].append(s)
        return ans.values()
