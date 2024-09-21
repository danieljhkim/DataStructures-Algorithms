from ast import List
from math import inf
from typing import Optional
from collections import defaultdict
from typing import List
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
    """
    given a p array of integers and q array of integers, for each element in q,
    return an array with the sum of absolute difference between each q element with p array
    i.e.
    p = [1,2,3]
    q = [5, 0]
    return [(|5-1| + |5-2| + |5-3|), (|0-1| + |0-2| + |0-3|)
    """


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
