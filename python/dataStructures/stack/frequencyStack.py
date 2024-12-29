from collections import defaultdict, Counter
import heapq


class FreqStack:

    def __init__(self):
        self.heap = []
        self.idx = 0
        self.table = defaultdict(int)

    def push(self, val: int) -> None:
        self.table[val] -= 1
        self.idx -= 1
        entry = (self.table[val], self.idx, val)
        heapq.heappush(self.heap, entry)

    def pop(self) -> int:
        freq, idx, val = heapq.heappop(self.heap)
        self.table[val] += 1
        return val


class FreqStack2(object):

    def __init__(self):
        self.freq = Counter()
        self.group = defaultdict(list)
        self.maxfreq = 0

    def push(self, x):
        f = self.freq[x] + 1
        self.freq[x] = f
        if f > self.maxfreq:
            self.maxfreq = f
        self.group[f].append(x)

    def pop(self):
        x = self.group[self.maxfreq].pop()
        self.freq[x] -= 1
        if not self.group[self.maxfreq]:
            self.maxfreq -= 1

        return x
