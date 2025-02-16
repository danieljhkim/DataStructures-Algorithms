from collections import OrderedDict
from sortedcontainers import SortedDict


class LFUCache:

    def __init__(self, capacity: int):
        self.size = 0
        self.cap = capacity
        self.freq_table = SortedDict()
        self.key_table = {}

    def _update_freq(self, key):
        val, freq = self.key_table[key]
        self.freq_table[freq].pop(key)
        if len(self.freq_table[freq]) == 0:
            del self.freq_table[freq]
        freq += 1
        if freq not in self.freq_table:
            self.freq_table[freq] = OrderedDict()
        self.freq_table[freq][key] = val
        self.key_table[key] = (val, freq)

    def get(self, key: int) -> int:
        if key not in self.key_table:
            return -1
        self._update_freq(key)
        return self.key_table[key][0]

    def put(self, key: int, value: int) -> None:
        if key in self.key_table:
            val, freq = self.key_table[key]
            if val != value:
                self.key_table[key] = (value, freq)
            self._update_freq(key)
        else:
            if self.size == self.cap:
                k, out = self.freq_table.popitem(index=0)
                idx, val = out.popitem(last=False)
                if len(out) > 0:
                    self.freq_table[k] = out
                del self.key_table[idx]
            else:
                self.size += 1
            if 1 not in self.freq_table:
                self.freq_table[1] = OrderedDict()
            self.key_table[key] = (value, 1)
            self.freq_table[1][key] = value
