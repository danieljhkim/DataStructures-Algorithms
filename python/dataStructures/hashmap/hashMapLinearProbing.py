class SimpleHashMapLinearProbing:
    def __init__(self):
        self.size = 1000
        self.table = [None] * self.size

    def _index(self, key: str) -> int:
        return hash(key) % self.size

    def put(self, key: str, value: int) -> None:
        idx = self._index(key)
        original_idx = idx
        while self.table[idx] is not None:
            if self.table[idx][0] == key:
                self.table[idx] = (key, value)
                return
            idx = (idx + 1) % self.size
            if idx == original_idx:
                raise Exception("HashMap is full")
        self.table[idx] = (key, value)

    def get(self, key: str) -> int:
        idx = self._index(key)
        original_idx = idx
        while self.table[idx] is not None:
            if self.table[idx][0] == key:
                return self.table[idx][1]
            idx = (idx + 1) % self.size
            if idx == original_idx:
                break
        return -1

    def remove(self, key: str) -> None:
        idx = self._index(key)
        original_idx = idx
        while self.table[idx] is not None:
            if self.table[idx][0] == key:
                self.table[idx] = None
                return
            idx = (idx + 1) % self.size
            if idx == original_idx:
                break


class HashMapLinearProbing:
    def __init__(self, initial_capacity=10):
        self.capacity = initial_capacity
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity

    def _hash(self, key):
        return hash(key) % self.capacity

    def _probe(self, key):
        index = self._hash(key)
        start_index = index  # To detect full table
        while self.keys[index] is not None and self.keys[index] != key:
            index = (index + 1) % self.capacity
            if index == start_index:
                raise Exception("HashMap is full")
        return index

    def put(self, key, value):
        if self.size >= self.capacity * 0.7:
            self._resize()
        index = self._probe(key)
        if self.keys[index] is None:
            self.size += 1
        self.keys[index] = key
        self.values[index] = value

    def get(self, key):
        index = self._probe(key)
        if self.keys[index] == key:
            return self.values[index]
        return None

    def delete(self, key):
        index = self._probe(key)
        if self.keys[index] == key:
            # Set the slot as deleted
            self.keys[index] = None
            self.values[index] = None
            self.size -= 1

    def _resize(self):
        old_keys = self.keys
        old_values = self.values
        self.capacity *= 2
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        for i in range(len(old_keys)):
            if old_keys[i] is not None:
                self.put(old_keys[i], old_values[i])
