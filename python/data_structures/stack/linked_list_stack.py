class Node:
    def __init__(self, value):
        self.val = value
        self.next = None


class Stack:
    """LIFI
    head -> one -> two
    """

    def __init__(self):
        self.head = None
        self.size = 0

    def push(self, val):
        new_node = Node(val)
        if not self.head:
            self.head = new_node
        else:
            curr = self.head
            self.head = new_node
            self.head.next = curr
        self.size += 1

    def pop(self):
        if not self.head:
            raise Exception("Stack is empty")
        value = self.head.val
        self.head = self.head.next
        self.size -= 1
        return value

    def peek(self):
        if self.head:
            return self.head.val
        return None
