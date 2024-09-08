from ast import List
from typing import Optional


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
    def convertDateToBinary(self, date: str) -> str:
        def binary(num):
            return bin(num)[2:]

        date = date.split("-")
        year = binary(int(date[0]))
        month = binary(int(date[1]))
        day = binary(int(date[2]))
        return year + "-" + month + "-" + day

    # def maxPossibleScore(self, start: List[int], d: int) -> int:
    #     pass

    def findMaximumScore(self, nums):
        def cal_score(i, j):
            return (j - i) * nums[i]

        outcomes = []
        tops = []
        for i in range(len(nums) - 1):
            outcome = []
            top = 0
            for j in range(i + 1, len(nums)):
                score = cal_score(i, j)
                top = max(top, score)
                outcome.append((j, score))
            outcomes.append(outcome)
            if i > 0:
                final = top
            else:
                final = top
            tops.append(final)
        print(outcomes)
        print(tops)


def test_solution():
    # s = Solution()
    # s.findMaximumScore([4, 3, 1, 3, 25])
    # s.findMaximumScore([1, 3, 1, 5])
    print(int(11 / -12))
    print(11 // -12)


if __name__ == "__main__":
    test_solution()
