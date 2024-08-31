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
    # inorder_stack(root)
    # print_tree(root)
