from email.policy import default
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


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
