from collections import defaultdict


class CharCount:

    def __init__(self, string):
        self.str = string
        self.prefix = defaultdict(list)
        self.counter = defaultdict(int)
        self.build_prefix()

    def build_prefix(self):
        for w in self.str:
            self.counter[w] += 1
            for k in self.prefix:
                self.prefix[k].append(self.counter[k])

    def char_count(self, char, start, end):
        """
        - start & end are inclusive ranges
        """
        if start == 0:
            return self.prefix[char][end]
        return self.prefix[end] - self.prefix[start - 1]
