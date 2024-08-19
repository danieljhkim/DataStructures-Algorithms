

from ast import List
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:

    # https://leetcode.com/problems/largest-number/description/
    def largestNumber(self, nums: List[int]) -> str:
        # Time complexity: O(n!)
        biggest = 0
        def sum_num(arr):
            total = ""
            for a in arr:
                total += str(a)
            return int(total)
            
        def permute(a_list, pos):
            nonlocal biggest
            length = len(a_list)
            if length == pos:
                total = sum_num(a_list)
                biggest = max(biggest, total)
            else:
                for i in range(pos, length):
                    a_list[pos], a_list[i] = a_list[i], a_list[pos]
                    permute(a_list, pos+1)
                    a_list[pos], a_list[i] = a_list[i], a_list[pos] # backtrack
        permute(nums, 0)
        return str(biggest)
    

    def largestNumber2(self, nums: List[int]) -> str: 
        # Time complexity: O(n^2)
        for i in range(len(nums) - 1):
            for j in range(i+1, len(nums)):
                if str(nums[i]) + str(nums[j]) < str(nums[j]) + str(nums[i]):
                    nums[i], nums[j] = nums[j], nums[i]
        total = ''.join(map(str, nums))
        if int(total) == 0:
            return '0'
        return total