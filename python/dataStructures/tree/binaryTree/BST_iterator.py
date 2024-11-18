from typing import Optional, List
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.root = root
        self.queue = deque()
        self.inorder(root)

    def inorder(self, node):
        if not node:
            return
        self.inorder(node.left)
        self.queue.append(node.val)
        self.inorder(node.right)

    def next(self) -> int:
        return self.queue.popleft()

    def hasNext(self) -> bool:
        return len(self.queue) > 0


class BSTIterator:
    """_summary_
    - iteratively controlled
    """

    def __init__(self, root: Optional[TreeNode]):
        self.root = root
        self.stack = []
        self.inorder(root)

    def inorder(self, node):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        cur = self.stack.pop()
        if cur.right:
            self.inorder(cur.right)
        return cur.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0
