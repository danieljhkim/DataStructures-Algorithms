from ast import List
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:

    # 230. Kth Smallest Element in a BST
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        values = []

        def inorder(root):
            if not root:
                return
            inorder(root.left)
            values.append(root.val)
            inorder(root.right)

        inorder(root)
        return values[k - 1]

    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        ans = None
        found = False

        def inorder(root, count):
            nonlocal ans, found
            if not root or found:
                return count
            count = inorder(root.left, count)
            if found:
                return count
            if count == k:
                ans = root.val
                found = True
                return count
            count += 1
            count = inorder(root.right, count)
            return count

        inorder(root, 1)
        return ans

    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        while root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val
            root = root.right
