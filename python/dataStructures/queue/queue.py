from ..linkedList.linked_list import LinkedList


class Queue:
    def __init__(self, size=0):
        self.list = LinkedList()
        self.size = size

    def enqueue(self, value):
        self.list.add(value)

    def dequeue(self):
        return self.list.remove_first()

    def is_empty(self):
        return self.list.head == None
