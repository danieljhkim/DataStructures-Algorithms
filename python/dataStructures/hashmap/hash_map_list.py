class HashMapLinearProbing:
    def __init__(self):
        self.size = 1000
        self.table = [[]] * self.size

    def _index(self, key: str) -> int:
        # Compute the initial index using the hash of the key
        return hash(key) % self.size

    def put(self, key: str, value: int) -> None:
        idx = self._index(key)
        # Check if the key already exists and update it
        for i, (k, v) in enumerate(self.table[idx]):
            if k == key:
                self.table[idx][i] = (key, value)
                return
        # If the key does not exist, append the new key-value pair
        self.table[idx].append((key, value))

    def get(self, key: str) -> int:
        idx = self._index(key)
        # Iterate through the list at the index to find the key
        for k, v in self.table[idx]:
            if k == key:
                return v
        return -1

    def remove(self, key: str) -> None:
        idx = self._index(key)
        # Iterate through the list at the index to find and remove the key
        for i, (k, v) in enumerate(self.table[idx]):
            if k == key:
                del self.table[idx][i]
                return
