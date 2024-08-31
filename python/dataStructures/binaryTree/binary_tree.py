# Definition for a binary tree node.


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.level = None


class BinarySearchTree:
    def __init__(self):
        self.root = None

    def create(self, val):
        if self.root == None:
            self.root = TreeNode(val)
        else:
            current = self.root
            while True:
                if val < current.val:
                    if current.left:
                        current = current.left
                    else:
                        current.left = TreeNode(val)
                        break
                elif val > current.val:
                    if current.right:
                        current = current.right
                    else:
                        current.right = TreeNode(val)
                        break
                else:
                    break


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

################################### DFS #################################


def inorderTraversal(root):
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


################################### BFS #################################

"""BFS
1. level order traversal - process nodes in order
2. Uses Queue
"""


def BFS(root):
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


def insert_into_bst(root, val):
    if root is None:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_into_bst(root.left, val)
    else:
        root.right = insert_into_bst(root.right, val)
    return root


def create_bst():
    root = None
    values = [4, 2, 6, 1, 3, 5, 7]
    for val in values:
        root = insert_into_bst(root, val)
    return root


# Test case
if __name__ == "__main__":
    root = create_bst()
    inorder_stack(root)
    # print_tree(root)
