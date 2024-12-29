from dataStructures.tree.binaryTree.binarySearchTree import TreeNode

################################### BFS #################################

"""BFS
1. level order traversal - process nodes in order
2. Uses Queue
"""


def BFS(root: TreeNode):
    if root is None:
        return

    queue = [root]
    while queue:
        node = queue.pop(0)
        # process(node)
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)
