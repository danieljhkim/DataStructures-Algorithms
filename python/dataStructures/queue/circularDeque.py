class MyCircularDeque:

    def __init__(self, k: int):
        self.capacity = k
        self.deque = [None] * k
        self.head_idx = 0
        self.size = 0

    def insertFront(self, value: int) -> bool:
        if self.size == self.capacity:
            return False
        self.head_idx = (self.head_idx + 1) % self.capacity
        self.deque[self.head_idx] = value
        self.size += 1
        return True

    def insertLast(self, value: int) -> bool:
        """ "
        [2,3,head,1]
        """
        if self.size == self.capacity:
            return False
        idx = (self.head_idx - self.size) % self.capacity
        self.size += 1
        self.deque[idx] = value
        return True

    def deleteFront(self) -> bool:
        if self.size == 0:
            return False
        self.head_idx = (self.head_idx - 1) % self.capacity
        self.size -= 1
        return True

    def deleteLast(self) -> bool:
        if self.size == 0:
            return False
        self.size -= 1
        return True

    def getFront(self) -> int:
        if self.size == 0:
            return -1
        return self.deque[self.head_idx]

    def getRear(self) -> int:
        if self.size == 0:
            return -1
        idx = (self.head_idx - self.size + 1) % self.capacity
        return self.deque[idx]

    def isEmpty(self) -> bool:
        return self.size == 0

    def isFull(self) -> bool:
        return self.size == self.capacity
