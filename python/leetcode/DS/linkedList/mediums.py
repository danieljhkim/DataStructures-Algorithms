from ast import List
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solutions:

    # 92. Reverse Linked List II
    def reverseBetween(
        self, head: Optional[ListNode], left: int, right: int
    ) -> Optional[ListNode]:

        # 0 -> 1 -> 2
        dummy = ListNode(0, head)
        curr = dummy.next
        prev = dummy
        left_prev = None
        right_node = None
        stack = []
        start = False
        count = 1
        while curr:
            if count == right:
                right_node = curr.next
                stack.append(curr)
                break
            if count == left:
                left_prev = prev
                start = True
            if start:
                stack.append(curr)
            count += 1
            prev = curr
            curr = curr.next
        curr = left_prev

        while stack and curr:
            next_node = stack.pop()
            if not stack:
                next_node.next = None
            curr.next = next_node
            curr = curr.next
        if curr:
            curr.next = right_node
        return dummy.next
