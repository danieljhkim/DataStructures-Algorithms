
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
  

  def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    length = len(flowerbed)
    i = 0
    while i < length:
      if flowerbed[i] == 0:
        _next = flowerbed[i + 1] if i < length - 1 else 0
        if _next == 0:
          i += 2
          n -= 1
        else:
          i += 1
        if n == 0:
          return True
      else:
        i += 1

    return i <= 0
  

  def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    # def postorder(root, val: int):
    #   if not root:
    #       return None
    #   return postorder(root.left, val)
    #   return postorder(root.right, val)
    #   if root.val == val:
    #       return root
    if not root:
      return None
    if root.val == val:
      return root
    elif root.val > val:
      return self.searchBST(root.left, val)
    elif root.val < val:
      return self.searchBST(root.right, val)
    
  def twoSum(self, nums: List[int], target: int) -> List[int]:
    def _merge_sort(arr):
      if len(arr) == 1:
        return arr
      mid = len(arr) // 2
      left = _merge_sort(arr[:mid])
      right = _merge_sort(arr[mid:])
      l = r = k = 0
      while l < len(left) and r < len(right):
        if left[l][1] <= right[r][1]:
          arr[k] = left[l]
          k += 1
          l += 1
        else:
          arr[k] = right[r]
          r += 1
          k += 1
      while l < len(left):
        arr[k] = left[l]
        k += 1
        l += 1
      while r < len(right):
        arr[k] = right[r]
        r += 1
        k += 1
      return arr
    
    nums_dict = [(i, nums[i]) for i in range(0, len(nums))]
    sorted_nums = _merge_sort(nums_dict)
    left = 0
    right = len(nums) - 1

    while left < right:
      l_value = sorted_nums[left][1]
      r_value = sorted_nums[right][1]
      total = l_value + r_value
      if total < target:
        left += 1
      elif total > target:
        right -= 1
      else:
        return [sorted_nums[left][0], sorted_nums[right][0]]
    
  def threeSum_brute_force(self, nums: List[int]) -> List[List[int]]:
    n = len(nums)
    ans = set()
    nums.sort()
    for i in range(0, n-2):
      for j in range(i+1, n-1):
        for k in range(j+1, n):
          total = nums[i] + nums[j] + nums[k]
          if total == 0:
            ans.add((nums[i], nums[j], nums[k]))
    return ans

  def threeSum(self, nums: List[int]) -> List[List[int]]: # optimized
    n = len(nums)
    ans = set()
    nums.sort()
    for l in range(n-2):
      m = l + 1
      r = n - 1
      while m < r:
        total = nums[l] + nums[r] + nums[m]
        if total == 0:
          ans.add((nums[l], nums[m], nums[r]))
          m += 1
        elif total > 0:
          r -= 1
        else:
          m += 1
    return ans
  
  
  def isValidSudoku(self, board: List[List[str]]) -> bool:
    #check horizontal for duplicates
    for row in board:
      hmap = {}
      for col in row:
        if col != ".":
          if hmap.get(col) is None:
            hmap[col] = 1
          else:
            return False
    
    #check verticle
    for i in range(9):
      rmap = {}
      for row in board:
        if row[i] != ".":
          if rmap.get(row[i]) is None:
            rmap[row[i]] = 1
          else:
            return False
    
    #check 3X3
    for r in range(0, 9, 3):
      for c in range(0, 9, 3):
        cmap = {}
        for i in range(c, c+3):
          for j in range(r, r+3):
            rnum = board[j][i]
            if rnum != ".":
              if cmap.get(rnum) is None:
                cmap[rnum] = 1
              else:
                return False
    return True

  def longestConsecutive(self, nums: List[int]) -> int:
    numSet = set(nums)
    ans = 0
    for n in numSet:
      if n-1 not in numSet:
        length = 1
        while n+length in numSet:
          length += 1
        ans = max(length, ans)
    return ans

  def permute(self, nums: List[int]) -> List[List[int]]:
    all_perms = []
    def _permute(a_list, pos):
      length = len(a_list)
      if length == pos:
        all_perms.append(a_list[:])
      else:
        for i in range(pos, length):
          a_list[pos], a_list[i] = a_list[i], a_list[pos]
          _permute(a_list, pos+1)
          a_list[pos], a_list[i] = a_list[i], a_list[pos] # backtrack
    _permute(nums, 0)
    return all_perms


  def permuteUnique(self, nums: List[int]) -> List[List[int]]:
    all_perms = []
    hmap = {}
    def _permute(a_list, pos):
      length = len(a_list)
      if length == pos:
        key = "".join(str(i) for i in a_list)
        if not hmap.get(key, False):
          all_perms.append(a_list[:])
          hmap[key] = True
      else:
        for i in range(pos, length):
          a_list[i], a_list[pos] = a_list[pos], a_list[i]
          _permute(a_list, pos+1)
          a_list[i], a_list[pos] = a_list[pos], a_list[i] 
    _permute(nums, 0)
    return all_perms


        
    