
from ast import List


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