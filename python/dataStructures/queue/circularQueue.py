class ListNode:
    def __init__(self, val=0, next=None, prev=None):
        self.val = val
        self.next = next
        self.prev = prev


class MyCircularQueue:

    def __init__(self, k: int):
        self.k = k
        self.head = ListNode(-1)
        self.tail = ListNode(-1, prev=self.head)
        self.head.next = self.tail
        self.size = 0

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        new = ListNode(value)
        prev = self.tail.prev
        prev.next = new
        new.prev = prev
        new.next = self.tail
        self.tail.prev = new
        self.size += 1
        return True

    def deQueue(self) -> bool:
        if self.size == 0:
            return False
        first = self.head.next
        second = first.next
        self.head.next = second
        second.prev = self.head
        first.prev = None
        first.next = None
        self.size -= 1
        return True

    def Front(self) -> int:
        if self.size == 0:
            return -1
        return self.head.next.val

    def Rear(self) -> int:
        if self.size == 0:
            return -1
        return self.tail.prev.val

    def isEmpty(self) -> bool:
        return self.size == 0

    def isFull(self) -> bool:
        return self.size == self.k
