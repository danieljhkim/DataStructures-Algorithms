from collections import defaultdict, OrderedDict


class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.min_freq = 0
        self.key_to_val_freq = {}
        # {key: (value, frequency)}
        self.freq_to_keys = defaultdict(OrderedDict)
        # {freq: {key: None (maintains order)}}

    def _update_freq(self, key):
        """Helper function to update the frequency of a given key."""
        value, freq = self.key_to_val_freq[key]

        del self.freq_to_keys[freq][key]
        if not self.freq_to_keys[freq]:
            # If no keys with this frequency, delete the frequency list
            del self.freq_to_keys[freq]
            if freq == self.min_freq:
                self.min_freq += 1

        # Add key to the next frequency list
        new_freq = freq + 1
        self.freq_to_keys[new_freq][key] = None
        self.key_to_val_freq[key] = (value, new_freq)

    def get(self, key: int) -> int:
        """Retrieve value associated with the key, updating its frequency."""
        if key not in self.key_to_val_freq:
            return -1

        self._update_freq(key)
        return self.key_to_val_freq[key][0]

    def put(self, key: int, value: int) -> None:
        """Add a key-value pair or update the value and frequency of an existing key."""
        if self.capacity == 0:
            return

        if key in self.key_to_val_freq:
            # Update the value and frequency of the existing key
            self.key_to_val_freq[key] = (value, self.key_to_val_freq[key][1])
            self._update_freq(key)
        else:
            # Check if cache is full
            if self.size == self.capacity:
                # Evict the least frequently used key
                evict_key, _ = self.freq_to_keys[self.min_freq].popitem(
                    last=False
                )  # Remove the oldest key in min_freq
                del self.key_to_val_freq[evict_key]
                self.size -= 1

            # Add the new key-value pair
            self.key_to_val_freq[key] = (value, 1)
            self.freq_to_keys[1][key] = None
            self.min_freq = 1  # Reset the min frequency to 1 for the new item
            self.size += 1


# Example Usage
lfu = LFUCache(3)
lfu.put(1, 1)
lfu.put(2, 2)
lfu.put(3, 3)
print(lfu.get(1))  # Output: 1 (updates frequency of key 1)
lfu.put(4, 4)  # Evicts key 2 (least frequently used)
print(lfu.get(2))  # Output: -1 (key 2 has been evicted)
print(lfu.get(3))  # Output: 3
print(lfu.get(4))  # Output: 4
