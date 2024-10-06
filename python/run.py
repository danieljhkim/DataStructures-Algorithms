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

    def int_to_bin(num):
        binary = ""
        while num > 0:
            remainder = num % 2
            num = num // 2
            binary = str(remainder) + binary
        return binary

    def bin_to_int(binary):
        num = 1
        for i, v in enumerate(binary):
            num = num * 2 + int(v)
        return num

    def maxGoodNumber(self, nums: List[int]) -> int:
        def bin_to_int(binary):
            num = 0
            for i, v in enumerate(binary):
                num = num * 2 + int(v)
            return num

        ans = float(-inf)
        arr = [None] * 3
        for i in range(3):
            arr[0] = str(bin(nums[i]))[2:]
            idx = 1
            for j in range(3):
                if i == j:
                    continue
                arr[idx] = str(bin(nums[j]))[2:]
                idx += 1

            bnum = bin_to_int("".join(arr))
            arr[1], arr[2] = arr[2], arr[1]
            cnum = bin_to_int("".join(arr))
            ans = max(bnum, ans, cnum)
        return ans

    # for nn in nmap[s]:
    #     if nn not in sus:
    #         sus.add(nn)
    #         changed = True

    def remainingMethods(
        self, n: int, k: int, invocations: List[List[int]]
    ) -> List[int]:
        """_summary_
        0: 1 2 3
        1: 0 1 2
        """
        ans = []
        nmap = defaultdict(list)
        sus = set()
        sus.add(k)
        good = set()
        for i in invocations:
            if i[0] == k:
                sus.add(i[1])
            else:
                good.add(i[1])
                good.add(i[0])
            nmap[i[0]].append(i[1])

        changed = True
        good = good.difference(sus)
        while changed:
            changed = False
            nset = set()
            prev = len(sus)
            for s in sus:
                nset.update(nmap[s])
            sus.update(nset)
            if prev != len(sus):
                changed = True
        for s in sus:
            if s not in good:
                good.remove(s)

        return list(good)


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
