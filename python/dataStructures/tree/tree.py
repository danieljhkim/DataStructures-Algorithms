"""
Tree
"""


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


def print_tree(node, level=0):
    if node is not None:
        print(" " * level * 2 + str(node.val))
        if node.children:
            for child in node.children:
                print_tree(child, level + 1)
