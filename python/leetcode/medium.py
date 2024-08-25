

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
    
    
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        num1 = ""
        num2 = ""
        head1 = l1
        head2 = l2
        while head1:
            num1 += str(head1.val) + num1
            head1 = head1.next
        while head2:
            num2 += str(head2.val) + num2
            head2 = head2.next
        total = str(int(num1) + int(num2))
        ans = ListNode(int(total[-1]))
        head3 = ans
        for i in range(len(total)-2, -1, -1):
            head3.next = ListNode(int(total[i]))
            head3 = head3.next
        return ans
            
        

    """
            0
        1       2
      2  3    4  5
    
    """
    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        queu = [(root, 0)]
        levels = {}
        while queu:
            curr, level = queu.pop(0)
            if level not in levels:
                levels[level] = []
            levels[level].append(curr)
            if curr.right:
                queu.append((curr.right, level+1))
            if curr.left:
                queu.append((curr.left, level+1))
        
        for level in levels:
            if level % 2 == 1:
                nodes = levels[level]
                vals = [node.val for node in nodes]
                j = 0
                for i in range(len(vals)-1, -1, -1):
                    nodes[i].val = vals[j]
                    j += 1
        return root

                
    def findShortestSubArray(self, nums: List[int]) -> int:
        degree_map = {}
        degree = 0
        
        for num in nums:
            if num not in degree_map:
                degree_map[num] = 0
            degree_map[num] += 1
            degree = max(degree_map[num], degree)
        degree_map = {key: val for key, val in degree_map.items() if val == degree}
        
        def check_max_deg(num, degree_map_c):
            nonlocal degree
            degree_map_c[num] -= 1
            for x in degree_map_c.values():
                if x == degree:
                    return True
            degree_map_c[num] += 1
            return False
        
        def find_it(left_no_more, right_no_more, degree_map_c):
            l = 0
            r = len(nums) - 1
            count = 0
            while count < 2 and l < r:
                left = nums[l]
                right = nums[r]
                if not left_no_more:
                    if left not in degree_map_c:
                        l += 1
                    elif check_max_deg(left, degree_map_c):
                        l += 1
                    else:
                        left_no_more = True
                        right_no_more = False
                        count += 1
                if not right_no_more:
                    if right not in degree_map_c:
                        r -= 1
                    elif check_max_deg(right, degree_map_c):
                        r -= 1
                    else:
                        right_no_more = True
                        left_no_more = False
                        count += 1
            return r - l + 1
        
        return min(find_it(True, False, degree_map.copy()), find_it(False, True, degree_map.copy()))
        
        