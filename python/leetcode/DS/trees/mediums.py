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

    # 98. Validate Binary Search Tree
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        stack = []
        prev_val = float("-inf")
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if root.val <= prev_val:
                return False
            prev_val = root.val
            root = root.right
        return True

    # 114. Flatten Binary Tree to Linked List
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        pre-order traversal
        """
        queue = []

        def pre_order(root):
            if not root:
                return
            queue.append(root)
            pre_order(root.left)
            pre_order(root.right)

        pre_order(root)
        if queue:
            curr = queue.pop(0)
        while queue:
            curr.left = None
            curr.right = queue.pop(0)
            curr = curr.right

    # 102. Binary Tree Level Order Traversal
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        def bfs(root):
            if not root:
                return
            level = 1
            queue = [(root, level)]
            levels = []
            while queue:
                node, level = queue.pop(0)
                if len(levels) < level:
                    n = level - len(levels)
                    for _ in range(n):
                        levels.append([])
                levels[level - 1].append(node.val)
                if node.left:
                    queue.append((node.left, level + 1))
                if node.right:
                    queue.append((node.right, level + 1))
            return levels

        return bfs(root)
