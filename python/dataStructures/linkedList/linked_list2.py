from typing import Optional
from dataclasses import dataclass
from __future__ import annotations
from typing import Optional, List


@dataclass
class Node:

    def __init__(self, val: int, next: Optional[Node]):
        self.val = val
        self.next = next


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class LinkedList:

    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0

    def insert(self, val: int) -> None:
        new_node = Node(val)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.length += 1

    def delete_all(self, val: int) -> int:
        dummy = Node(-1)
        dummy.next = self.head
        cur = dummy
        deleted_count = 0
        while cur and cur.next:
            if cur.next.val == val:
                cur.next = cur.next.next
                deleted_count += 1
                self.length -= 1
            cur = cur.next
        # head update
        self.head = dummy.next

        # tail udpate
        if not self.head:
            self.tail = None
        else:
            cur = self.head
            while cur.next:
                cur = cur.next
            self.tail = cur
        return deleted_count

    def reverse(self):
        """
        1 -> 2 -> 3 -> None
        None <- 1 <- 2 <- 3
        """
        self.tail = self.head
        cur = self.head
        prev = None
        while cur:
            cur2 = cur.next
            cur.next = prev
            prev = cur
            cur = cur2
        self.head = prev

    def mergeNodes(self, head: Optional[Node]) -> Optional[Node]:
        cur = head
        zeros = 0
        total = 0
        zero_node = None
        prev = None
        while cur:
            if cur.val == 0:
                zeros += 1
                if zeros == 2:
                    zeros = 1
                    prev.next = None  # cut
                    zero_node.val = total
                    zero_node.next = cur.next
                    total = 0
                    cur = zero_node
                elif zeros == 1:
                    zero_node = cur
                    total = 0
            if zeros == 1:
                total += cur.val
            prev = cur
            cur = cur.next
        return head

    def mergeNodes(self, head: Optional[Node]) -> Optional[Node]:

        cur = head.next
        prev = head
        zero_node = head
        total = 0
        while cur:
            total += cur.val
            if cur.val == 0:

                zero_node.val = total
                zero_node.next = cur.next
                zero_node = cur

                total = 0
            prev = cur
            cur = cur.next
        return head.next

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:

        pos = {}
        for i, nodes in enumerate(lists):
            pos[i] = nodes
        head = ListNode(-1)
        cur = head
        while pos:
            smallest = float("inf")
            idx = -1
            empty = []
            for i in pos.keys():
                if not pos[i]:
                    empty.append(i)
                    continue
                num = pos[i].val
                if num < smallest:
                    idx = i
                    smallest = num
            if idx > -1:
                cur.next = pos[idx]
                cur = cur.next
                pos[idx] = pos[idx].next
            else:
                break
            if empty:
                for i in empty:
                    del pos[i]
        return head.next

    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:

        def find_middle(node):
            slow = node
            fast = node
            while fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            return slow

        def merge_sort(node):
            if not node or not node.next:
                return node
            mid = find_middle(node)
            left = node
            right = mid.next
            mid.next = None  # IMPORTMAT!!!! cut it!!

            left = merge_sort(left)
            right = merge_sort(right)

            dummy = ListNode(-1, node)
            cur = dummy
            while left and right:
                if left.val > right.val:
                    cur.next = right
                    right = right.next
                else:
                    cur.next = left
                    left = left.next
                cur = cur.next
            cur.next = left if left else right
            return dummy.next

        return merge_sort(head)
