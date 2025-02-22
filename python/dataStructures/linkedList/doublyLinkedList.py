class ListNode:

    def __init__(self, val, next, prev):
        self.val = val
        self.next = next
        self.prev = prev


class MyLinkedList:

    def __init__(self):
        self.head = ListNode(-1, prev=None, next=None)
        self.tail = ListNode(val=-1, prev=self.head, next=None)
        self.head.next = self.tail
        self.size = 0

    def get(self, index: int) -> int:
        if index >= self.size:
            return -1

        mid = self.size // 2
        if mid > index:
            idx = self.size - index - 1
            cur = self.tail.prev
            while cur and idx > 0:
                cur = cur.prev
                idx -= 1
            return cur.val
        else:
            idx = index
            cur = self.head.next
            while cur and idx > 0:
                cur = cur.next
                idx -= 1
            return cur.val

    def getNode(self, index: int) -> ListNode:
        if index >= self.size:
            return None

        mid = self.size // 2
        if mid > index:
            idx = self.size - index - 1
            cur = self.tail.prev
            while cur and idx > 0:
                cur = cur.prev
                idx -= 1
            return cur
        else:
            idx = index
            cur = self.head.next
            while cur and idx > 0:
                cur = cur.next
                idx -= 1
            return cur

    def addAtHead(self, val: int) -> None:
        first_node = self.head.next
        new_node = ListNode(val, prev=self.head, next=first_node)
        self.head.next = new_node
        first_node.prev = new_node
        self.size += 1

    def addAtTail(self, val: int) -> None:
        last_node = self.tail.prev
        new_node = ListNode(val, prev=last_node, next=self.tail)
        self.tail.prev = new_node
        last_node.next = new_node
        self.size += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index > self.size:
            return
        if index == self.size:
            self.addAtTail(val)
        elif index == 0:
            self.addAtHead(val)
        else:
            node = self.getNode(index)
            prev = node.prev
            new_node = ListNode(val, prev=prev, next=node)
            prev.next = new_node
            node.prev = new_node
            self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        if index >= self.size:
            return
        node = self.getNode(index)
        prev = node.prev
        nxt = node.next
        prev.next = nxt
        nxt.prev = prev
        node.next = None
        node.prev = None
        self.size -= 1
