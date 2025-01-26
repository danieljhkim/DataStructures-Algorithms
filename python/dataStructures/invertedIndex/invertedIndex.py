from collections import defaultdict


class InvertedIndex:

    def __init__(self):
        self.table = {}
        self.index = defaultdict(set)
        self.idx = 1

    def add_index(self, texts):
        parsed = self.parse_text(texts)
        self.table[self.idx] = parsed
        for w in parsed:
            self.index[w].add(str(self.idx))
        self.idx += 1
        return self.idx - 1

    def query_partial_match(self, words):
        wset = set()
        for w in words:
            wset.update(self.index[w.lower()])
        return list(wset)

    def query_all_match(self, words):
        wset = self.index[words[0]].copy()
        for w in words[1:]:
            wset.intersection_update(self.index[w.lower()])
            if len(wset) == 0:
                return []
        return list(wset)

    def parse_text(self, texts):
        buffer = []
        N = len(texts)
        words = set()
        for i, n in enumerate(texts):
            if n.isalnum() or n == "'":
                buffer.append(n)
                if i != N - 1:
                    continue
            if buffer:
                word = "".join(buffer)
                buffer.clear()
                words.add(word.lower())
        return words
