from typing import List


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class PostFixTreeBuilder:

    operands = set(["+", "*", "/", "-"])

    def buildTree(self, postfix: List[str]):
        """
        ["3","4","+","2","*","7","/"]
        """
        N = len(postfix)
        self.idx = N - 1

        def recurs():
            if self.idx < 0:
                return None
            cur = postfix[self.idx]
            node = TreeNode(cur)
            self.idx -= 1
            if cur not in self.operands:
                node.val = int(node.val)
                return node
            node.right = recurs()
            node.left = recurs()
            return node

        root = recurs()
        return root

    def evaluate(self, root) -> int:
        def dfs(node):
            if node.val not in self.operands:
                return node.val
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            if node.val == "+":
                return left + right
            if node.val == "-":
                return left - right
            if node.val == "*":
                return left * right
            if node.val == "/":
                return left / right

        return int(dfs(root))
