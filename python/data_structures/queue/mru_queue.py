from sortedcontainers import SortedList


class MRUQueue:
    """
    Most Recently Used Queue
    """

    def __init__(self, n: int):
        self.queue = SortedList([(i, i + 1) for i in range(n)])

    def fetch(self, k: int) -> int:
        idx, val = self.queue.pop(k - 1)
        pos = self.queue[-1][0] + 1 if self.queue else 0
        self.queue.add((pos, val))
        return val
