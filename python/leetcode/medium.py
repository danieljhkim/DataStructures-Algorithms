import heapq
from typing import Optional, List
import random
from collections import deque, defaultdict
import math


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

    def rob(self, nums: List[int]) -> int:
        memo = {}

        def recursion(i):
            if i >= len(nums):
                return 0
            if i in memo:
                return memo[i]
            cur = nums[i] + recursion(i + 2)
            skip = recursion(i + 1)
            chosen = max(cur, skip)
            memo[i] = chosen
            return chosen

        return recursion(0)

    def maxScore(self, cardPoints: List[int], k: int) -> int:
        total = sum(cardPoints[:k])
        max_score = total

        for i in range(1, k + 1):
            total = total - cardPoints[k - i] + cardPoints[-i]
            max_score = max(max_score, total)

        return max_score

    def maxKelements(self, nums: List[int], k: int) -> int:
        heap = [-num for num in nums]
        heapq.heapify(heap)
        ans = 0
        while k:
            num = -heapq.heappop(heap)
            ans += num
            num = math.ceil(num / 3)
            heapq.heappush(heap, -num)
            k -= 1
        return ans

    def maximumSwap(self, num: int) -> int:
        if num < 10:
            return num
        snum = list(str(num))
        nmap = {}
        for i in range(len(snum)):
            nmap[int(snum[i])] = i
        for s in range(len(snum)):
            for i in range(9, int(snum[s]), -1):
                if i in nmap and nmap[i] > s:
                    snum[s], snum[nmap[i]] = snum[nmap[i]], snum[s]
                    return int("".join(snum))
        return num

    def removeDuplicates(self, s: str) -> str:
        head = ListNode("0")
        cur = head
        for i in s:
            cur.next = ListNode(i)
            cur = cur.next
        cur = head
        changed = True
        while changed:
            changed = False
            while cur and cur.next and cur.next.next:
                if cur.next.val == cur.next.next.val:
                    cur.next = cur.next.next.next
                    changed = True
                cur = cur.next
            cur = head
        ans = ""
        cur = head.next
        while cur:
            ans += cur.val
            cur = cur.next
        return ans

    def removeDuplicates(self, s: str) -> str:
        arr = list(s)

        def helper(left, right):
            if right < len(s) and left >= 0:
                if arr[right] == 0 or arr[left] == 0:
                    if arr[right] == 0:
                        right += 1
                    if arr[left] == 0:
                        left -= 1
                    helper(left, right)

                elif arr[right] == arr[left]:
                    arr[left] = 0
                    arr[right] = 0
                    helper(left - 1, right + 1)

        for i in range(len(s) - 1):
            helper(i, i + 1)
        ans = ""
        for letter in arr:
            if letter != 0:
                ans += letter
        return ans

    def removeDuplicates(self, s: str) -> str:
        stack = []
        for ch in s:
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                stack.append(ch)
        return "".join(stack)

    def findMissingRanges(
        self, nums: List[int], lower: int, upper: int
    ) -> List[List[int]]:
        if len(nums) == 0:
            return [lower, upper]
        ans = []
        if lower < nums[0]:
            ans.append([lower, nums[0] - 1])
        for i in range(len(nums) - 1):
            if (nums[i + 1] - nums[i]) > 1:
                ans.append([nums[i] + 1, nums[i + 1] - 1])
        if upper > nums[-1]:
            ans.append([nums[-1] + 1, upper])
        return ans

    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        col_map = defaultdict(list)
        queue = deque()
        queue.append((root, 0))
        max_col = min_col = 0
        while queue:
            node, col = queue.popleft()
            col_map[col].append(node.val)
            if node.left:
                min_col = min(col - 1, min_col)
                queue.append((node.left, col - 1))
            if node.right:
                max_col = max(col + 1, max_col)
                queue.append((node.right, col + 1))
        ans = []
        for i in range(min_col, max_col + 1):
            ans.append(col_map[i])
        return ans

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        stack = [intervals[0]]
        idx = 1
        while idx < len(intervals):
            cur = stack.pop()
            second = intervals[idx]
            if cur[1] >= second[0]:
                cur[1] = max(second[1], cur[1])
                stack.append(cur)
            else:
                stack.append(cur)
                stack.append(second)
            idx += 1
        return stack

    def subarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += nums[j]
                if total == k:
                    ans += 1

        return ans

    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        nmap = defaultdict(list)
        for r in range(len(mat)):
            for c in range(len(mat[0])):
                diag = r + c
                nmap[diag].append(mat[r][c])
        ans = []
        for i, v in nmap.items():
            if i % 2 == 0:
                ans.extend(v[::-1])
            else:
                ans.extend(v)
        return ans

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        ans = []
        heap = []
        for i in range(len(points)):
            dis = (points[i][0] ** 2 + points[i][1] ** 2) ** (0.5)
            if len(heap) < k:
                heapq.heappush(heap, (-dis, points[i]))
            else:
                if -heap[0][0] > dis:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (-dis, points[i]))
        for p in heap:
            ans.append(p[1])
        return ans

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        def helper(mid, remain, dists):
            closer = []
            fars = []
            for i in remain:
                if dists[i] <= mid:
                    closer.append(i)
                else:
                    fars.append(i)
            return closer, fars

        ans = []
        dists = [p[0] ** 2 + p[1] ** 2 for p in points]
        remain = [i for i in range(len(points))]
        high = max(dists)
        low = 0
        closest = []
        while k > 0:
            mid = (high + low) / 2
            closer, fars = helper(mid, remain, dists)
            if len(closer) <= k:
                closest.extend(closer)
                remain = fars
                k -= len(closer)
                low = mid
            else:
                remain = closer
                high = mid
        for idx in closest:
            ans.append(points[idx])
        return ans

    def depthSum(self, nestedList: List["NestedInteger"]) -> int:
        def find_sum(arr, depth):
            total = 0
            for a in arr:
                if a.isInteger():
                    total += a.getInteger() * depth
                else:
                    total += find_sum(a.getList(), depth + 1)
            return total

        ans = find_sum(nestedList, 1)
        return ans

    def nextPermutation(self, nums: List[int]) -> None:
        def find_closest(num, idx):
            n = len(nums)
            ans = float("inf")
            jdx = -1
            for i in range(idx, n):
                if nums[i] > num:
                    dif = nums[i] - num
                    if dif < ans:
                        ans = dif
                        jdx = i
            return jdx

        right = len(nums) - 1
        start = -1
        while right > 0:
            if nums[right] > nums[right - 1]:
                start = right - 1
                break
            right -= 1

        if start == -1:
            nums.reverse()
            return

        closest = find_closest(nums[start], start + 1)
        nums[start], nums[closest] = nums[closest], nums[start]
        nums[start + 1 :] = nums[start + 1 :][::-1]

    def intervalIntersection(
        self, firstList: List[List[int]], secondList: List[List[int]]
    ) -> List[List[int]]:
        f = 0
        s = 0
        ans = []
        while f < len(firstList) and s < len(secondList):
            low = max(firstList[f][0], secondList[s][0])
            high = min(firstList[f][1], secondList[s][1])
            if low <= high:
                ans.append([low, high])
            if firstList[f][1] > secondList[s][1]:
                s += 1
            else:
                f += 1
        return ans

    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        def shift_string(s: str) -> str:
            shift = ord(s[0]) - ord("a")
            shifted = []
            for char in s:
                sft = ord(char) - ord("a")
                if sft <= shift:
                    shifted_char = chr(ord("a") + sft - shift + 26)
                else:
                    shifted_char = chr(ord("a") + sft - shift)
                shifted.append(shifted_char)
            return "".join(shifted)

        groups = defaultdict(list)

        for s in strings:
            shifted = shift_string(s)
            groups[shifted].append(s)

        return list(groups.values())

    def isAlienSorted(self, words: List[str], order: str) -> bool:
        map = {}
        for i, w in enumerate(order):
            map[w] = i
        for i in range(len(words) - 1):
            fw = words[i]
            sw = words[i + 1]
            for fx in range(max(len(fw), len(sw))):
                second = map[sw[fx]] if fx < len(sw) else 0
                first = map[fw[fx]] if fx < len(fw) else 0
                if first < second:
                    break
                elif second == first:
                    continue
                else:
                    return False
        return True

    def compress(self, chars: List[str]) -> int:
        idx = 0
        i = 0
        while i < len(chars):
            cur = chars[i]
            j = i
            while j < len(chars) and chars[j] == cur:
                j += 1
            chars[idx] = cur
            idx += 1
            if j - i > 1:
                for s in str(j - i):
                    chars[idx] = s
                    idx += 1
            i = j
        return idx

    def cloneGraph(self, node: Optional["Node"]) -> Optional["Node"]:
        if not node:
            return node
        clones = {}

        def dfs(node):
            if node in clones:
                return clones[node]
            cloned = Node(node.val, [])
            clones[node] = cloned
            if node.neighbors:
                cloned.neighbors = [dfs(n) for n in node.neighbors]
            return cloned

        return dfs(node)

    def insert(
        self, intervals: List[List[int]], newInterval: List[int]
    ) -> List[List[int]]:
        if not intervals:
            return [newInterval]
        ans = []
        found = False
        for i, v in enumerate(intervals):
            if found:
                prev = ans[-1]
                if prev[1] >= v[0]:
                    prev[1] = max(v[1], prev[1])
                else:
                    ans.append(v)
            else:
                if newInterval[0] >= v[0] and newInterval[0] <= v[1]:
                    found = True
                    v[1] = max(newInterval[1], v[1])
                    ans.append(v)
                elif newInterval[0] <= v[0] and newInterval[1] >= v[0]:
                    v[0] = newInterval[0]
                    v[1] = max(newInterval[1], v[1])
                    while len(ans) > 0 and v[0] <= ans[-1][1]:
                        prev = ans.pop()
                        v[0] = min(prev[0], v[0])
                        v[1] = max(prev[1], v[1])
                    found = True
                    ans.append(v)
                elif newInterval[1] < v[0]:
                    found = True
                    ans.append(newInterval)
                    ans.append(v)
                else:
                    ans.append(v)
        if not found:
            ans.append(newInterval)
        return ans

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0:
            return [-1, -1]

        low = 0
        high = len(nums) - 1
        if target > nums[high] or target < nums[low]:
            return [-1, -1]

        while low <= high:
            mid = (high + low) // 2
            midVal = nums[mid]
            if midVal < target:
                low = mid + 1
            else:
                high = mid - 1

        if nums[low] != target:
            return [-1, -1]
        if low < len(nums) - 1 and nums[low + 1] != target:
            return [low, low]

        high = len(nums) - 1
        lo = low
        while lo <= high:
            mid = (high + lo) // 2
            if nums[mid] <= target:
                lo = mid + 1
            else:
                high = mid - 1
        if nums[high] == target:
            return [low, high]

        return [-1, -1]

    def search(self, nums: List[int], target: int) -> int:
        shift = 0
        n = len(nums)
        for i in range(n - 1):
            if nums[i] > nums[i + 1]:
                shift = i + 1
                break

        low = 0
        high = n - 1
        while low <= high:
            mid = (low + high) // 2
            new_mid = (mid + shift) % n
            mid_num = nums[new_mid]
            if mid_num < target:
                low = mid + 1
            elif mid_num > target:
                high = mid - 1
            else:
                return new_mid
        return -1
