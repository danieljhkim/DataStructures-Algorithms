"""_summary_
            0
        1       2
      3   4    5  6
    7  8   7    9   11
    
"""


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class BinaryTree:

    def __init__(self):
        self.pfound = False
        self.qfound = False

    # 1644
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        """
        - p and q may not exist
        - in this case, we want to explore all branches
        """

        def dfs(node):
            if not node:
                return node
            left = dfs(node.left)
            right = dfs(node.right)
            if node == p:
                self.pfound = True
                return node
            elif node == q:
                self.qfound = True
                return node
            if left and right:
                return node
            if left:
                return left
            else:
                return right

        lca = dfs(root)
        if self.pfound and self.qfound:
            return lca
        return None

    # 236
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        """_summary_
        - p and q are guarantee to exist
        - so, we don't have to explore deeper, just return early when p or q when found
        - if left and right both returns, current node is the lca
        - otherwise, whichever is first found is the lca
        """

        def dfs(node):
            if not node or node == p or node == q:
                # return early since no need to explore deeper
                return node

            left = dfs(node.left)
            right = dfs(node.right)
            if left and right:
                return node
            if left:
                return left
            return right

        lca = dfs(root)
        return lca
