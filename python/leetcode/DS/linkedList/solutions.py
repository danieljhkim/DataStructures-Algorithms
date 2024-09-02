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

    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        num1 = ""
        num2 = ""
        head1 = l1
        head2 = l2
        while head1:
            num1 += str(head1.val) + num1
            head1 = head1.next
        while head2:
            num2 += str(head2.val) + num2
            head2 = head2.next
        total = str(int(num1) + int(num2))
        ans = ListNode(int(total[-1]))
        head3 = ans
        for i in range(len(total) - 2, -1, -1):
            head3.next = ListNode(int(total[i]))
            head3 = head3.next
        return ans
