from math import inf
from collections import defaultdict
from typing import List, Tuple, Optional
import heapq


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    pass

    def reportSpam(self, message: List[str], bannedWords: List[str]) -> bool:
        count = 0
        hset = set(bannedWords)
        for m in message:
            if m in hset:
                count += 1
            if count == 2:
                return True
        return False

    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        """_summary_
        2 1 1
        2
        """

        s = 1
        hmap = defaultdict(int)
        top = 0
        for i in range(len(workerTimes)):
            hmap[workerTimes[i]] += 1
            top = max(workerTimes[i], top)
        while mountainHeight:
            for i in range(top):
                mountainHeight -= hmap[i] * s

    def validSubstringCount(self, word1: str, word2: str) -> int:
        if len(word1) < len(word2):
            return 0

        def is_valid(word):
            nMap2 = nMap.copy()
            nonlocal word2
            for c in range(len((word))):
                nMap2[word[c]] += 1
            for i in range(len(word2)):
                if nMap2[word2[i]] > 0:
                    nMap2[word2[i]] -= 1
                else:
                    return False
            return True

        ans = 0

        for size in range(len(word2), len(word1)):
            start = 0
            end = size
            nMap = defaultdict(int)
            for c in range(size):
                nMap[word1[c]] += 1
            while end <= len(word1):
                if is_valid(word1[start:end]):
                    ans += 1
                nMap[word1[end]] += 1
                nMap[word1[start]] -= 1
                start += 1
                end += 1
        return ans


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
