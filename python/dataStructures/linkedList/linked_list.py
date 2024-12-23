from dataclasses import dataclass
from __future__ import annotations
from typing import Optional, List


@dataclass
class Node:
    value: int
    next: Node | None = None

    def __init__(self, value):
        self.value = value
        self.next = None


@dataclass
class LinkedList:
    head: Node | None = None
    tail: Node | None = None
    length: int

    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0

    def __len__(self):
        return self.length

    def add(self, value):
        node = Node(value)
        if self.head:
            self.tail.next = node
            self.tail = node
        else:
            self.head = node
            self.tail = node
        self.length += 1

    def remove_first(self):
        if self.head:
            value = self.head.value
            self.head = self.head.next
            self.length -= 1
            return value


def reverse(head):
    """
    1 <- 2 <- 3 <- 4
    """
    curr = head
    prev = None
    while curr:
        curr_next = curr.next
        curr.next = prev
        prev = curr
        curr = curr_next
