
from ast import List
from typing import Optional

class TreeNode:
  def __init__(self, val=0, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

class Solution:

  # https://leetcode.com/problems/count-pairs-whose-sum-is-less-than-target/
  def countPairs(self, nums: List[int], target: int) -> int:
    ans = 0
    for i in range(0, len(nums)-1):
      ii = nums[i]
      for j in range(i+1, len(nums)):
        if ii + nums[j] < target:
          ans += 1
    return ans
  
  # https://leetcode.com/problems/left-and-right-sum-differences/
  def leftRightDifference(self, nums: List[int]) -> List[int]:
    length = len(nums)
    ans = []
    left = [0] * length
    right = [0] * length
    lsum = nums[0]
    rsum = nums[length-1]
    for i in range(1, length):
      left[i] = lsum
      lsum += nums[i]

    for i in range(length-2, -1, -1):
      right[i] = rsum
      rsum += nums[i]

    for i in range(0,length):
      ans.append(abs(left[i]-right[i]))
    return ans
  
  def letterCombinations(self, digits: str) -> List[str]:
    if len(digits) == 0:
      return []
    dmap = {
      "2": "abc",
      "3": "def",
      "4": "ghi",
      "5": "jkl",
      "6": "mno",
      "7": "pqrs",
      "8": "tuv",
      "9": "wxyz"
    }
    ans = [""]
    for digit in digits:
      dstr = dmap[digit]
      comb = []
      for ansStr in ans:
        for dchar in dstr:
          comb.append(ansStr + dchar)
      ans = comb
    return ans
  
  def maxDepth(self, root: Optional[TreeNode]) -> int:
    def dfs(root, depth):
      if not root: return depth
      bigger = max(dfs(root.left, depth+1), dfs(root.right, depth+1))
      return bigger
    return dfs(root, 0)
  
  def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    def isSame(root_p, root_q):
      if root_p is None and root_q is None:
        return True
      if root_p is None or root_q is None:
        return False
      if root_p.val != root_q.val:
        return False
      left = isSame(root_p.left, root_q.left)
      right = isSame(root_p.right, root_q.right)
      return left and right
    return isSame(p, q)
      
  def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    if root is None:
      return root
    root.left, root.right = root.right, root.left
    self.inverTree(root.left)
    self.inverTree(root.right)
    return root

  def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    def isMirror(root_left, root_right):
      if root_left is None and root_right is None:
        return True
      if root_right is None or root_left is None:
        return False
      same_val = root_right.val == root_left.val
      left = isMirror(root_left.left, root_right.right)
      right = isMirror(root_left.right, root_right.left)
      return same_val and left and right
    if not root:
      return True
    return isMirror(root.left, root.right)

  def isBalanced(self, root: Optional[TreeNode]) -> bool:
    def getHeight(root) -> int:
      if root is None:
        return 0
      left_depth = getHeight(root.left)
      right_depth = getHeight(root.right)
      return max(left_depth, right_depth) + 1
    if root is None:
      return True
    left_depth = getHeight(root.left)
    right_depth = getHeight(root.right)
    height_diff = abs(left_depth - right_depth)
    if height_diff < 2 and self.isBalanced(root.left) and self.isBalanced(root.right):
      return True
    return False
  
  def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
    total = len(nums)
    if not total:
      return None
    mid = total // 2
    return TreeNode(nums[mid], self.sortedArrayToBST(nums[:mid]), self.sortedArrayToBST(nums[mid+1:]))
  
  def countNodes(self, root: Optional[TreeNode]) -> int:
    if root is None:
      return 0
    return 1 + self.countNodes(root.left) + self.countNodes(root.right)


  def minDepth(self, root: Optional[TreeNode]) -> int:
    if root is None:
      return 0
    if root.left is None and root.right is None:
      return 1
    leftDepth = self.minDepth(root.left)
    rightDepth = self.minDepth(root.right)
    if root.left is None:
      return rightDepth + 1
    if root.right is None:
      return leftDepth + 1
    return min(rightDepth, leftDepth) + 1