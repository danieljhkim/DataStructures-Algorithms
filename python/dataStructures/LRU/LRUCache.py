from dataclasses import dataclass
from __future__ import annotations


@dataclass
class ListNode:
    def __init__(self, key, val=0):
        self.val = val
        self.key = key
        self.next = None
        self.prev = None


@dataclass
class LRUCache:
    """
    - when a node is accessed, the node is placed back to the tail.
    - when a node is added, it is added to the tail - in case of over capacity, head is removed.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tail = ListNode(-1, -1)  # dummy
        self.head = ListNode(-1, -1)  # dummy
        self.head.next = self.tail
        self.tail.prev = self.head
        self.dict = {}

    def get(self, key: int) -> int:
        if key not in self.dict:
            return -1
        node = self.dict[key]
        self._remove(node)
        self._add(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.dict:
            node = self.dict[key]
            node.val = value
            self._remove(node)
            self._add(node)
        else:
            new_node = ListNode(key, value)
            self._add(new_node)
            self.dict[key] = new_node
            if len(self.dict) > self.capacity:
                front_node = self.head.next
                self._remove(front_node)
                del self.dict[front_node.key]

    def _add(self, node: ListNode):
        # puts node to the end of the list
        prev_end = self.tail.prev
        prev_end.next = node
        node.prev = prev_end
        node.next = self.tail
        self.tail.prev = node

    def _remove(self, node: ListNode):
        node.prev.next = node.next
        node.next.prev = node.prev
