class Node:
    def __init__(self, val, next=None, prev=None):
        self.words = set()
        self.freq = 1
        self.prev = prev
        self.next = next


class FreqMinMax:

    def __init__(self):
        self.map = {}
        self.head = Node(0)
        self.tail = Node(0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def inc(self, key: str) -> None:
        if key not in self.map:
            if self.tail.prev.freq > 1:
                new_tail = Node(1)
                new_tail.words.add(key)
                self._add_to_tail(new_tail)
                self.map[1] = new_tail
            elif self.tail.prev.freq == 1:
                self.tail.prev.freq += 1
                self.tail.words.add(key)

    def dec(self, key: str) -> None:
        pass

    def getMaxKey(self) -> str:
        pass

    def getMinKey(self) -> str:
        pass

    def _add_to_tail(self, new_tail):
        prev_tail = self.tail.prev
        prev_tail.next = new_tail
        self.tail.prev = new_tail
        new_tail.prev = prev_tail
        new_tail.next = self.tail
