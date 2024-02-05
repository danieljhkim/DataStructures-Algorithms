

class Node:
  def __init__(self, key):
    self.left = None
    self.right = None
    self.val = key

'''
         1
     2       3
   4   5   6   7
  
'''

def inorderTraversal(root):
  # tranverses nodes in non-decreasing order
  # 4 2 5 1 6 3 7  
  if root:
    inorderTraversal(root.left)
    print(root.val)
    inorderTraversal(root.right)

def preorderTraversal(root):
  # used to create a copy of the tree
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



