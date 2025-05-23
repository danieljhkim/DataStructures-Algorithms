from dataclasses import dataclass
from __future__ import annotations
from typing import Dict, List, Optional

"""
                16
         /              \
       8                24
     /   \            /    \
   4      12       20      28
  / \    /  \     /  \    /  \
 2   6  10  14  18   22  26   30
"""


@dataclass
class TreeNode(object):
    val: int
    left: TreeNode | None = None
    right: TreeNode | None = None

    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None
        self.level = None


@dataclass
class BinarySearchTree:

    root: TreeNode | None = None

    def __init__(self, values: list):
        self.root = self.build(values)

    def build(self, values: list):
        # values must be sorted to be balanced
        N = len(values)
        if N == 0:
            return None
        mid = N // 2
        root = TreeNode(mid)
        root.left = self.build(values[:mid])
        root.right = self.build(values[mid + 1 :])
        return root

    def iterative_insert(self, root, val):
        if root == None:
            root = TreeNode(val)
        current = root
        while True:
            if val < current.val:
                if not current.left:
                    current.left = TreeNode(val)
                    break
                current = current.left
            elif val > current.val:
                if not current.right:
                    current.right = TreeNode(val)
                    break
                current = current.right
            else:
                break
        return root

    def insert(self, root, val):
        if root is None:
            return TreeNode(val)
        if val < root.val:
            root.left = self.insert(root.left, val)
        elif val > root.val:
            root.right = self.insert(root.right, val)
        return root

    def get_min_node(self, node: TreeNode):
        cur = node
        while cur.left:
            cur = cur.left
        return cur

    def get_max_node(self, node: TreeNode):
        cur = node
        while cur.right:
            cur = cur.right
        return cur

    def delete(self, node: TreeNode, val):
        """
        1. when its a leaf node
        2. when it has one child - replace the node with the child
        3. when it has 2 children - find predecessor and replace the node's val with the predecessor and delete the predecessor
        """
        if not node:
            return node
        if node.val > val:
            node.left = self.delete(node.left, val)
        elif node.val < val:
            node.right = self.delete(node.right, val)
        else:
            if not node.left:
                return node.right
            elif not node.right:
                return node.left

            node_max_of_left = self.get_max_node(node.left)
            node.val = node_max_of_left.val
            node.left = self.delete(node.left, node.val)
        return node

    def max_depth(self) -> int:
        return self._get_depth(self.root, 0)

    def _get_depth(self, node: TreeNode, level: int) -> int:
        if not node:
            return level
        left_depth = self._get_depth(node.left, level + 1)
        right_depth = self._get_depth(node.right, level + 1)
        return max(left_depth, right_depth)


################################### helpers #################################


def print_tree(root):
    # only works for balanced binary trees
    levels = {}

    def bfs(root):
        queue = [(root, 0)]
        level = 0
        while queue:
            curr, level = queue.pop(0)
            if level not in levels:
                levels[level] = []
            levels[level].append(curr.val)
            if curr.left:
                queue.append((curr.left, level + 1))
            if curr.right:
                queue.append((curr.right, level + 1))

    bfs(root)
    for i in range(len(levels)):
        space = (len(levels) * 4) // ((i + 1) * 3) * " "
        stuff = space
        for node in levels[i]:
            stuff += str(node) + space
        print(stuff)


def pathSum(root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
    result = []

    def dfs(node, path, total):
        if not node:
            return
        path.append(node.val)
        total += node.val
        if not node.left and not node.right:
            if targetSum == total:
                result.append(path[::])
        dfs(node.left, path, total)
        dfs(node.right, path, total)
        path.pop()

    dfs(root, [], 0)
    return result


if __name__ == "__main__":
    # print_tree(root)
    pass
