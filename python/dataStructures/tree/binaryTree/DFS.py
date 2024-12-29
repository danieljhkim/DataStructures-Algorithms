from dataStructures.tree.binaryTree.binarySearchTree import TreeNode


""" ordered 
         1
     2       3
   4   5   6   7
  
"""

""" binary search tree
left is smaller and right is bigger
         4
     2       6
   1   3   5   7
  
"""

################################### Recursion #################################


def inorderTraversal(root: TreeNode):
    # tranverses nodes in non-decreasing order
    # 4 2 5 1 6 3 7
    # 1 2 3 4 5 6 7
    if root:
        inorderTraversal(root.left)
        print(root.val)
        inorderTraversal(root.right)


def preorderTraversal(root):
    # used to create a copy/mirror of the tree
    # used to get prefix expressions of an expression tree
    # 1 2 4 5 3 6 7
    if root:
        print(root.val)
        preorderTraversal(root.left)
        preorderTraversal(root.right)


def postorderTraversal(root):
    # used to delete a tree
    # useful for getting postrix expression
    # 4 5 2 6 7 3 1
    if root:
        postorderTraversal(root.left)
        postorderTraversal(root.right)
        print(root.val)


################################### Stacks #################################

"""
    1
  2  3
 4 5 6  7

"""


def preorder_stack(root):
    """inverse pre-order
    stack : [1] -> [2, 3] -> [2, 6, 7] -> [2, 6] -> [2] -> [4,5]

    processing order: 1 -> 3 -> 7 -> 6 -> 2 -> 5 -> 4
    """
    levels = {}
    stack = [(root, 0)]
    level = 0
    while stack:
        curr, level = stack.pop()  # pop the last one
        if level not in levels:
            levels[level] = []
        levels[level].append(curr.val)
        print(f"{curr.val} - {level}")
        if curr.left:
            stack.append((curr.left, level + 1))
        if curr.right:
            stack.append((curr.right, level + 1))


def inorder_stack(root):
    """_summary_
    stack: [1] [1, 3] [1, 3, 7]
    """
    stack = []
    curr = root
    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        print(curr.val)
        curr = curr.right


def postorder_stack(root):
    if not root:
        return
    stack1 = [root]
    stack2 = []
    while stack1:
        curr = stack1.pop()
        stack2.append(curr)
        if curr.left:
            stack1.append(curr.left)
        if curr.right:
            stack1.append(curr.right)
    while stack2:
        curr = stack2.pop()
        print(curr.val)  # Process the current node
