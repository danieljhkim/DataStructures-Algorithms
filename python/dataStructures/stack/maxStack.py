import heapq
from sortedcontainers import SortedDict


class MaxStack:

    def __init__(self):
        self.idx = 0
        self.idx_dict = SortedDict()
        self.heap = []

    def push(self, x: int) -> None:
        heapq.heappush(self.heap, (-x, -self.idx))
        self.idx_dict[self.idx] = x
        self.idx += 1

    def pop(self) -> int:
        idx, val = self.idx_dict.popitem()
        return val

    def top(self) -> int:
        idx, val = self.idx_dict.peekitem()
        return val

    def peekMax(self) -> int:
        while self.heap and -self.heap[0][1] not in self.idx_dict:
            heapq.heappop(self.heap)
        return -self.heap[0][0]

    def popMax(self) -> int:
        while self.heap and -self.heap[0][1] not in self.idx_dict:
            heapq.heappop(self.heap)
        out, idx = heapq.heappop(self.heap)
        del self.idx_dict[-idx]
        return -out
