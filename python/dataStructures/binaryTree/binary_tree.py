# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def treeNodeToString(root):
    if not root:
        return "[]"
    output = ""
    queue = [root]
    current = 0
    while current != len(queue):
        node = queue[current]
        current = current + 1

        if not node:
            output += "null, "
            continue

        output += str(node.val) + ", "
        queue.append(node.left)
        queue.append(node.right)
    return "[" + output[:-2] + "]"

def stringToTreeNode(input):
    input = input.strip()
    input = input[1:-1]
    if not input:
        return None

    inputValues = [s.strip() for s in input.split(',')]
    root = TreeNode(int(inputValues[0]))
    nodeQueue = [root]
    front = 0
    index = 1
    while index < len(inputValues):
        node = nodeQueue[front]
        front = front + 1

        item = inputValues[index]
        index = index + 1
        if item != "null":
            leftNumber = int(item)
            node.left = TreeNode(leftNumber)
            nodeQueue.append(node.left)

        if index >= len(inputValues):
            break

        item = inputValues[index]
        index = index + 1
        if item != "null":
            rightNumber = int(item)
            node.right = TreeNode(rightNumber)
            nodeQueue.append(node.right)
    return root

def prettyPrintTree(node, prefix="", isLeft=True):
    if not node:
        print("Empty Tree")
        return

    if node.right:
        prettyPrintTree(node.right, prefix + ("│   " if isLeft else "    "), False)

    print(prefix + ("└── " if isLeft else "┌── ") + str(node.val))

    if node.left:
        prettyPrintTree(node.left, prefix + ("    " if isLeft else "│   "), True)

def main():
    import sys

    def readlines():
        for line in sys.stdin:
            yield line.strip('\n')

    lines = readlines()
    while True:
        try:
            line = lines.next()
            node = stringToTreeNode(line)
            prettyPrintTree(node)
        except StopIteration:
            break


if __name__ == '__main__':
    main()

''' ordered 
         1
     2       3
   4   5   6   7
  
'''

''' left is smaller and right is bigger
         4
     2       6
   1   3   5   7
  
'''

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



