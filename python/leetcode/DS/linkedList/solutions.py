from ast import List
from calendar import c
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

    # 725. Split Linked List in Parts
    def splitListToParts(
        self, head: Optional[ListNode], k: int
    ) -> List[Optional[ListNode]]:
        ans = []
        size = 0
        curr = head
        while curr:
            size += 1
            curr = curr.next

        split_size = size // k
        remainder = size % k
        curr = head

        while k > 0:
            cur_size = split_size
            if remainder > 0:
                cur_size += 1
                remainder -= 1
            node = curr
            head_node = node
            while cur_size > 1:
                if node:
                    node = node.next
                cur_size -= 1
            if node:
                curr = node.next
                node.next = None
            ans.append(head_node)
            k -= 1
        return ans

    def insert(self, head: "Optional[Node]", insertVal: int) -> "Node":
        new_node = Node(insertVal)
        if not head:
            head = new_node
            head.next = head
            return head
        cur = head
        if cur.next == cur:
            cur.next = new_node
            new_node.next = cur
            return head
        smallest = float("inf")
        biggest = float("-inf")
        is_full_loop = False
        while not is_full_loop:
            smallest = min(cur.val, smallest)
            biggest = max(cur.val, biggest)
            if cur.next.val >= insertVal and insertVal >= cur.val:
                temp = cur.next
                new_node.next = temp
                cur.next = new_node
                return head
            cur = cur.next
            if cur == head:
                is_full_loop = True

        if biggest == smallest:
            temp = cur.next
            cur.next = new_node
            new_node.next = temp
            return head

        while cur.val != biggest:
            cur = cur.next
        if cur.val == cur.next.val:
            while cur.val == cur.next.val:
                cur = cur.next
        temp = cur.next
        cur.next = new_node
        new_node.next = temp
        return head
