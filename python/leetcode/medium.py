from ast import List
from curses.ascii import FF
from typing import Optional
import random
from collections import deque


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

    # 50
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        if x == 0:
            return 0

        if n < 0:
            x = 1 / x
            n = -n

        result = 1
        current_product = x

        while n > 0:
            if n % 2 == 1:
                result *= current_product
            current_product *= current_product
            n //= 2

        return result

    # 921
    def minAddToMakeValid(self, s: str) -> int:
        left = 0
        right = 0
        for p in s:
            if p == "(":
                left += 1
            else:
                if left < 1:
                    right += 1
                else:
                    left -= 1
        return left + right

    # 227
    def calculate(self, s: str) -> int:
        stack = []
        num = 0
        prev_op = "+"
        s += "."
        for i in range(len(s)):
            ch = s[i]
            if ch.isdigit():
                num = num * 10 + int(ch)
            elif ch != " ":
                if prev_op == "-":
                    stack.append(-num)
                elif prev_op == "+":
                    stack.append(num)
                elif prev_op == "*":
                    stack.append(stack.pop() * num)
                elif prev_op == "/":
                    stack.append(int(stack.pop() / num))
                prev_op = ch
                num = 0
        ans = 0
        while stack:
            ans += stack.pop()
        return ans

    # 162
    def findPeakElement(self, nums: List[int]) -> int:
        low = 0
        high = len(nums) - 1
        while low < high:
            mid = (low + high) // 2
            if nums[mid] > nums[mid + 1]:
                high = mid
            else:
                low = mid + 1
        return low

    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        max_row = len(grid) - 1
        max_col = len(grid[0]) - 1
        if grid[0][0] != 0 or grid[max_row][max_col] != 0:
            return -1
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        def get_paths(row, col):
            paths = []
            for d in directions:
                new_row = d[0] + row
                new_col = d[1] + col
                if not 0 <= new_row <= max_row:
                    continue
                if not 0 <= new_col <= max_col:
                    continue
                if grid[new_row][new_col] != 0:
                    continue
                paths.append((new_row, new_col))
            return paths

        que = deque()
        que.append((0, 0))
        grid[0][0] = 1
        while que:
            row, col = que.popleft()
            dist = grid[row][col]
            if (row, col) == (max_row, max_col):
                return dist
            paths = get_paths(row, col)
            for path in paths:
                que.append(path)
                grid[path[0]][path[1]] += dist + 1
        return -1

    def findBuildings(self, heights: List[int]) -> List[int]:
        mx_h = 0
        n = len(heights) - 1
        ans = []
        for i in range(n, -1, -1):
            if heights[i] > mx_h:
                ans.append(i)
            mx_h = max(mx_h, heights[i])
        return ans.reverse()

    def longestOnes(self, nums: List[int], k: int) -> int:
        left = 0
        n = len(nums)
        flip = 0
        ans = 0
        for right in range(n):
            if nums[right] == 0:
                flip += 1
                if flip > k:
                    while left <= right and flip > k:
                        flip -= 1 - nums[left]
                        left += 1
            ans = max(right - left + 1, ans)
        return ans

    def coinChange(self, coins: List[int], amount: int) -> int:
        coins.sort(reverse=True)
        ans = 0
        for coin in coins:
            ans += amount // coin
            amount = amount % coin
        if amount > 0:
            return -1
        return ans

    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(r, c):
            if (
                r >= max_row
                or r < 0
                or c >= max_col
                or c < 0
                or visited[r][c] == True
                or grid[r][c] == "0"
            ):
                return
            visited[r][c] = True
            dfs(r - 1, c)
            dfs(r + 1, c)
            dfs(r, c - 1)
            dfs(r, c + 1)

        ans = 0
        max_row = len(grid)
        max_col = len(grid[0])
        visited = [[False] * max_col for _ in range(max_row)]

        for r in range(max_row):
            for c in range(max_col):
                if grid[r][c] == "1" and not visited:
                    ans += 1
                    dfs(r, c)
        return ans


# 528
class Solution:

    def __init__(self, w: List[int]):
        self.prefix_sum = []
        self.total = 0
        for i in w:
            self.total += i
            self.prefix_sum.append(self.total)

    def pickIndex(self) -> int:
        target = random.random() * self.total
        left = 0
        right = len(self.prefix_sum)
        while left < right:
            mid = (left + right) // 2
            val = self.prefix_sum[mid]
            if val < target:
                left = mid + 1
            else:
                right = mid
        return left
