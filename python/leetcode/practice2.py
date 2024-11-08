import heapq
from typing import Optional, List
import random
from collections import Counter, deque, defaultdict, OrderedDict
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

    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        intervals.sort()
        for i in range(1, len(intervals)):
            cur = intervals[i - 1]
            nex = intervals[i]
            if cur[1] > nex[0]:
                return False
        return True

    def convertToTitle(self, columnNumber: int) -> str:
        col = columnNumber
        ans = ""
        while col:
            col -= 1
            rem = col % 26
            col = col // 26
            ans += chr(rem + ord("A"))
        return ans[::-1]

    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []
        for i, v in enumerate(asteroids):
            if not stack or v > 0:
                stack.append(v)
            else:
                if v < 0 and stack[-1] > 0:
                    while stack and stack[-1] > 0 and stack[-1] < abs(v):
                        stack.pop()
                    if not stack:
                        stack.append(v)
                    else:
                        if stack[-1] < 0:
                            stack.append(v)
                        elif stack[-1] == abs(v):
                            stack.pop()
                elif (v < 0 and stack[-1] < 0) or (v > 0 and stack[-1] > 0):
                    stack.append(v)
        return stack

    def str2tree(self, s: str) -> Optional[TreeNode]:
        # parent(child_left)(child_right)
        stack = []
        while stack:
            if not stack:
                node = TreeNode()

    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        if not root:
            return False
        table = set()
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if k - node.val in table:
                return True
            table.add(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return False

    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left = 0
        pos = 0
        for left in range(len(nums)):
            if nums[left] == 0:
                pos = max(pos, left)
                while pos < len(nums) and nums[pos] == 0:
                    pos += 1
                if pos < len(nums) and nums[pos] != 0:
                    nums[left], nums[pos] = nums[pos], nums[left]
                    pos += 1
                else:
                    break


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
