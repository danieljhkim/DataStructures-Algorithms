"""
Design a HashMap without using any built-in hash table libraries.

Implement the MyHashMap class:

MyHashMap() initializes the object with an empty map.
void put(int key, int value) inserts a (key, value) pair into the HashMap. If the key already exists in the map, update the corresponding value.
int get(int key) returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key.
void remove(key) removes the key and its corresponding value if the map contains the mapping for the key.

"""


class ListNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None


class HashMapLinkedList:

    def __init__(self, key, value):
        self.size = 1000
        self.table = [None] * self.size

    def _index(self, key: int) -> int:
        return key % self.size

    def _index_str(self, key: str) -> int:
        # for string keys
        return hash(key) % self.size

    def put(self, key: int, value: int) -> None:
        idx = self._index(key)
        if not self.table[idx]:
            self.table[idx] = ListNode(key, value)
        else:
            current = self.table[idx]
            while current:
                if current.key == key:
                    current.value = value
                    return
                if not current.next:
                    current.next = ListNode(key, value)
                    return
                current = current.next

    def get(self, key: int) -> int:
        idx = self._index(key)
        current = self.table[idx]
        while current:
            if current.key == key:
                return current.value
            current = current.next
        return -1

    def remove(self, key: int) -> None:
        idx = self._index(key)
        current = self.table[idx]
        if not current:
            return
        if current.key == key:
            self.table[idx] = current.next
            return
        while current.next:
            if current.next.key == key:
                current.next = current.next.next
                return
            current = current.next
