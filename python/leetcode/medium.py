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
                    permute(a_list, pos + 1)
                    a_list[pos], a_list[i] = a_list[i], a_list[pos]  # backtrack

        permute(nums, 0)
        return str(biggest)

    def largestNumber2(self, nums: List[int]) -> str:
        # Time complexity: O(n^2)
        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                if str(nums[i]) + str(nums[j]) < str(nums[j]) + str(nums[i]):
                    nums[i], nums[j] = nums[j], nums[i]
        total = "".join(map(str, nums))
        if int(total) == 0:
            return "0"
        return total

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

        return min(
            find_it(True, False, degree_map.copy()),
            find_it(False, True, degree_map.copy()),
        )

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        start: 1, 2, 8  ,15
        end:   3, 6, 10, 18
        """
        has_changed = False

        while not has_changed:
            i = 0
            has_changed = False
            while i < len(intervals):
                start = intervals[i][0]
                end = intervals[i][1]
                s_min = start
                e_max = end
                j = i + 1
                while j < len(intervals):
                    j_start = intervals[j][0]
                    j_end = intervals[j][1]
                    if e_max >= j_start:
                        s_min = min(s_min, j_start)
                        e_max = max(j_end, e_max)
                        intervals.pop(j)
                    j += 1
                    intervals[i] = [s_min, e_max]
                    has_changed = True
                i += 1
        return intervals

    def minLength(self, s: str) -> int:
        ar = list(s)
        i = 0
        while i < len(ar) - 1:
            if (ar[i] == "A" and ar[i + 1] == "B") or (
                ar[i] == "C" and ar[i + 1] == "D"
            ):
                del ar[i : i + 2]
                if i >= len(ar):
                    return len(ar)
                i = i - 1 if i > 0 else 0
                break
            else:
                i += 1
        return len(ar)

    # 560. Subarray Sum Equals K
    # wrong - redo
    def subarraySum(self, nums: List[int], k: int) -> int:
        if k == 0 and len(nums) == 1 and nums[0] == 0:
            return 0
        right = 0
        ans = 0
        cur_sum = 0
        n = len(nums)
        for left in range(n):
            while cur_sum < k and right < n:
                cur_sum += nums[right]
                right += 1
            if cur_sum == k:
                ans += 1
            cur_sum -= nums[left]
        return ans
