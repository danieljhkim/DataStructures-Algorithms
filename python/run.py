from math import inf
from collections import Counter, defaultdict, deque
from typing import List, Tuple, Optional
import heapq
import math


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
    def stringSequence(self, target: str) -> List[str]:
        ans = ["a"]
        idx = 0

        for i in range(len(target)):
            let = target[i]
            nlet = ans[-1][-1]
            neww = list(ans[-1])
            if ord(let) > ord(nlet):
                while ord(let) > ord(nlet):
                    nlet = chr(ord(nlet) + 1)
                    neww[len(neww) - 1] = nlet
                    ans.append("".join(neww))
            else:
                ans.append(ans[-1] + let)
        return ans

    def numberOfSubstrings(self, s: str, k: int) -> int:
        def cal(arr):
            counts = [0] * 26
            for i in arr:
                counts[ord(i) - ord("a")] += 1
            return counts

        ans = 0
        left = 0
        counts = cal(s[:k])
        right = k - 1

        while right < len(s):
            if max(counts) >= k:
                ans += len(s) - right
                counts[ord(s[left]) - ord("a")] -= 1
                left += 1
            else:
                right += 1
                if right < len(s):
                    counts[ord(s[right]) - ord("a")] += 1

        return ans

    def possibleStringCount(self, word: str) -> int:
        ans = 1
        words = list(word)
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                ans += 1
        return ans


def test_solution():
    s = Solution()
    s.maxScore(3, 2, [[3, 4, 2], [2, 1, 2]], [[0, 2, 1], [2, 0, 4], [3, 2, 0]])


if __name__ == "__main__":
    test_solution()
