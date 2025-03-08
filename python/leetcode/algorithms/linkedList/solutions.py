import heapq
import random
import math
import bisect
from typing import *
from math import inf, factorial, gcd, lcm
from functools import lru_cache, cache
from heapq import heapify, heappush, heappop
from itertools import accumulate, permutations, combinations
from collections import Counter, deque, defaultdict, OrderedDict
from sortedcontainers import SortedSet, SortedList, SortedDict


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:

    # 82. Remove Duplicates from Sorted List II
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        dummy = ListNode(-1000)
        dummy.next = head
        prev = dummy
        cur = head
        while cur:
            dup = False
            while cur.next and cur.val == cur.next.val:
                cur = cur.next
                dup = True
            if dup:
                prev.next = self.deleteDuplicates(cur.next)
                break
            else:
                prev = cur
                cur = cur.next
        return dummy.next

    # 86. Partition List
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        big = ListNode(-1000)
        small = ListNode(-1000)
        bcur = big
        scur = small

        while head:
            nxt = head.next
            if head.val >= x:
                bcur.next = head
                bcur = bcur.next
                bcur.next = None
            else:
                scur.next = head
                scur = scur.next
                scur.next = None
            head = nxt
        scur.next = big.next
        return small.next


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
