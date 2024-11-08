"""
Tree
    Definition: A tree is a special type of graph that is connected and acyclic.
    
    Structure: Consists of nodes connected by edges, with one node designated as the root.
    
    Properties:
        - Acyclic: No cycles (i.e., no path where you can start from a node and return to it by traversing edges).
        - Connected: There is exactly one path between any two nodes.
        - Hierarchical: Nodes are organized in a parent-child relationship.
        - Rooted: Has a single root node from which all other nodes descend.
        - Edges: If there are n nodes, there are n-1 edges.
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
