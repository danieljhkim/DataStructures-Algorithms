import heapq
from typing import Optional, List
import random
from collections import Counter, deque, defaultdict, OrderedDict
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
    def minEatingSpeed(self, piles: List[int], h: int) -> int:

        def helper(k):
            time = 0
            for pile in piles:
                time += math.ceil(pile / k)
            if time > h:
                return False
            return True

        low = min(piles)
        high = max(piles)
        while low < high:
            mid = (high + low) // 2
            ate_all = helper(mid)
            if ate_all:
                high = mid
            else:
                low = mid + 1
        return high

    def minEatingSpeed(self, woods: List[int], k: int) -> int:
        low = 1
        high = max(woods)
        while low <= high:
            mid = (high + low) // 2
            pieces = 0
            for w in woods:
                pieces += w // mid
            if pieces > k:
                high = mid - 1
            else:
                low = mid + 1

    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        left = 0
        ans = []
        for w in s:
            if w == ")":
                if left > 0:
                    left -= 1
                    stack.append(w)
            elif w == "(":
                left += 1
                stack.append(w)
            else:
                stack.append(w)

        while stack:
            w = stack.pop()
            if w == "(" and left > 0:
                left -= 1
                continue
            ans.append(w)
        return "".join(ans.reverse())

    def depthSum(self, nestedList: List["NestedInteger"]) -> int:
        def dfs(nest, depth):
            total = 0
            for n in nest:
                if n.isInteger():
                    total += n.getInteger() * depth
                else:
                    total += dfs(n.getList(), depth + 1)
            return total

        return dfs(nestedList, 1)

    def depthSum(self, nestedList: List["NestedInteger"]) -> int:
        stack = [(1, nestedList)]
        total = 0
        while stack:
            depth, nest = stack.pop()
            if isinstance(nest, list):
                for i in range(len(nest)):
                    stack.append((depth, nest[i]))
            else:
                if nest.isInteger():
                    total += nest.getInteger() * depth
                else:
                    stack.append((depth + 1, nest.getList()))
        return total

    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = deque()
        queue.append((0, root))
        colMap = defaultdict(list)
        low = 0
        high = 0
        while queue:
            col, node = queue.popleft()
            colMap[col].append(node.val)
            if node.left:
                low = min(low, col - 1)
                queue.append((col - 1, node.left))
            if node.right:
                high = max(high, col + 1)
                queue.append((col + 1, node.right))
        ans = []
        for i in range(low, high + 1):
            ans.append(colMap[i])
        return ans

    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        nmap = defaultdict(list)
        mxCol = 0
        minCol = 0

        def dfs(node, col, row):
            if node.left:
                nonlocal minCol
                minCol = min(minCol, col - 1)
                dfs(node.left, col - 1, row + 1)
            nmap[col].append((row, node.val))
            if node.right:
                nonlocal mxCol
                mxCol = max(mxCol, col + 1)
                dfs(node.right, col + 1, row + 1)

        dfs(root, 0, 0)
        ans = []
        for i in range(minCol, mxCol + 1):
            nmap[i].sort(key=lambda x: x[0])
            ans.append([x[1] for x in nmap[i]])
        return ans

    def lowestCommonAncestor(self, p: "Node", q: "Node") -> "Node":
        pset = set()
        cur_p = p
        while cur_p:
            pset.add(cur_p)
            cur_p = cur_p.parent

        while q:
            if q in pset:
                return q
            q = q.parent

        return None

    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for i in nums:
            if len(heap) < k:
                heapq.heappush(heap, i)
            else:
                if heap[0] < i:
                    heapq.heappop(heap)
                    heapq.heappush(heap, i)
        return heap[0]

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        num1_copy = list(nums1[:m])
        i_1 = 0
        i_2 = 0
        idx = 0
        while i_1 < m and i_2 < n:
            if num1_copy[i_1] >= nums2[i_2]:
                nums1[idx] = nums2[i_2]
                i_2 += 1
            else:
                nums1[idx] = num1_copy[i_1]
                i_1 += 1
            idx += 1
        while i_1 < m:
            nums1[idx] = num1_copy[i_1]
            i_1 += 1
            idx += 1
        while i_2 < n:
            nums1[idx] = nums2[i_2]
            i_2 += 1
            idx += 1

    def maximumCandies(self, candies: List[int], k: int) -> int:
        def calcCandies(size):
            count = 0
            for c in candies:
                count += c // size
                if count >= k:
                    return True
            return False

        low = 1
        high = max(candies)
        while low <= high:
            mid = (low + high) // 2
            if calcCandies(mid):
                low = mid + 1
            else:
                high = mid - 1
        return high

    def minimizedMaximum(self, n: int, quantities: List[int]) -> int:

        def canDistribute(amount):
            count = 0
            for q in quantities:
                count += (q + amount - 1) // amount
            return count <= n

        low = 1
        high = max(quantities)
        while low <= high:
            mid = (low + high) // 2
            if canDistribute(mid):
                high = mid - 1
            else:
                low = mid + 1
        return low

    def repairCars(self, ranks: List[int], cars: int) -> int:
        def howManyCars(time):
            total = 0
            for rank in ranks:
                total += math.isqrt(time // rank)
            return total

        low = 1
        high = 100 * cars * cars
        while low <= high:
            mid = (high + low) // 2
            finished = howManyCars(mid)
            if finished < cars:
                low = mid + 1
            else:
                high = mid - 1
        return low

    def searchInsert(self, nums: List[int], target: int) -> int:
        low = 0
        high = len(nums) - 1
        while low <= high:
            mid = (high + low) // 2
            if nums[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return low

    def isStrobogrammatic(self, num: str) -> bool:
        # 0, 1, 8
        # 6, 9
        arr = list(num)
        left = 0
        right = len(arr) - 1
        badNums = set([2, 3, 4, 5, 7])
        specialNums = [6, 9]
        while left <= right:
            lnum = int(arr[left])
            rnum = int(arr[right])
            if lnum in badNums or rnum in badNums:
                return False
            if lnum != rnum:
                if lnum in specialNums and rnum in specialNums:
                    left += 1
                    right -= 1
                    continue
                return False
            else:
                if lnum in specialNums and rnum in specialNums:
                    return False
            if left == right:
                if lnum in specialNums:
                    return False
            left += 1
            right -= 1
        return True

    # def can_complete(idx):
    #     tank = gas[idx]
    #     i = idx
    #     while tank >= 0:
    #         i = (i + 1) % n
    #         if cost[i] > tank:
    #             return False
    #         tank += gas[i] - cost[i]
    #         if i == idx:
    #             return True
    # def canDoIt(idx):
    # tank = gas[idx]
    # for _ in range(n):
    #     idx = (idx + 1) % n
    #     tank += costArr[idx]
    #     if tank < 0:
    #         return False
    # return True

    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        # 1 2 3 4 5
        # 3 4 5 1 1
        #
        n = len(gas)
        costArr = []
        sumArr = [0] * n
        for i in range(n):
            takes = gas[i] - cost[(i) % n]
            costArr.append(takes)
        if sum(costArr) < 0:
            return False
        sumArr[0] = costArr[0]
        for i in range(1, n):
            sumArr[i] = sumArr[i - 1] + costArr[i]
        print(sumArr)
        print(costArr)
        return -1

    def permute(self, nums: List[int]) -> List[List[int]]:
        ans = []

        def backtrack(arr, pos):
            if pos == len(nums):
                ans.append(arr[:])
            for i in range(pos, len(nums)):
                arr[i], arr[pos] = arr[pos], arr[i]
                backtrack(arr, pos + 1)
                arr[i], arr[pos] = arr[pos], arr[i]

        backtrack(nums, 0)
        return ans

    def removeSubfolders(self, folder: List[str]) -> List[str]:
        result = []
        paths = set(folder)
        for f in folder:
            is_sub = False
            tempf = f
            while tempf != "":
                slash_pos = tempf.rfind("/")
                if slash_pos == -1:
                    break
                tempf = tempf[:slash_pos]
                if tempf in paths:
                    is_sub = True
                    break
            if not is_sub:
                result.append(f)
        return result

    def subarraySum(self, nums: List[int], k: int) -> int:
        ans = 0
        scores = [1] * len(nums)
        totals = []

        for i in range(len(nums)):
            cur_num = nums[i]
            totals.append(0)
            found = False
            for total_idx in range(len(totals)):
                new_total = totals[total_idx] + cur_num
                totals[total_idx] = new_total
                if new_total == k and not found:
                    found = True
                    ans += scores[total_idx]
                    scores[total_idx] += 1

        return ans

    def subarraySum(self, nums: List[int], k: int) -> int:
        ans = 0
        scores = [1] * len(nums)
        totals = OrderedDict(int)

        for i in range(len(nums)):
            cur_num = nums[i]
            totals[i] = cur_num
            for total_idx in totals.keys():
                new_total = totals[total_idx] + cur_num
                totals[total_idx] = new_total
                if new_total == k:
                    ans += scores[total_idx]
                    scores[total_idx] += 1
        return ans

    def subarraySum2(self, nums: List[int], k: int) -> int:
        dic = defaultdict(int)  # sum 0 occurs once
        dic[0] = 1
        runningSum = 0
        count = 0
        for num in nums:
            runningSum += num
            difference = runningSum - k
            if difference in dic:
                count += dic[difference]
            dic[runningSum] += 1
        return count

    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        def dfs(node, count):
            if not node:
                return float("inf")
            if not node.left and not node.right:
                return count + 1
            left = dfs(node.left, count + 1)
            right = dfs(node.right, count + 1)
            return min(left, right)

        result = dfs(root, 0)
        return result if result != float("inf") else 0


class Solution:

    def __init__(self, nums: List[int]):
        self.table = defaultdict(list)
        for i, n in enumerate(nums):
            self.table[n].append(i)

    def pick(self, target: int) -> int:
        choices = self.table[target]
        n = len(choices)
        chosen = math.floor(n * random.random())
        return choices[chosen]

    def simplifyPath(self, path: str) -> str:
        stack = []
        paths = path.split("/")
        for p in paths:
            if p == "..":
                if stack:
                    stack.pop()
            elif p == " " or p == "" or p == ".":
                continue
            else:
                stack.append("/" + p)
        result = "".join(stack)
        return result or "/"

    def closestValue(self, root: Optional[TreeNode], target: float) -> int:
        heap = []  # (abs(root.val - target), root.val)

        def dfs(node):
            if not node:
                return
            dfs(node.left)
            diff = (abs(node.val - target), node.val)
            if heap:
                if heap[0][0] >= diff[0]:
                    heapq.heappush(heap, diff)
            else:
                heapq.heappush(heap, diff)
            dfs(node.right)

        dfs(root)
        return heap[0][1]

    def isAlienSorted(self, words: List[str], order: str) -> bool:
        def is_sorted(first, second):
            length = min(len(first), len(second))
            first_idx = 0
            second_idx = 0
            while first_idx < length and second_idx < length:
                first_order = diction[first[first_idx]]
                second_order = diction[second[second_idx]]
                if first_order < second_order:
                    return True
                elif first_order == second_order:
                    first_idx += 1
                    second_idx += 1
                else:
                    return False
            if len(first) == length:
                return True
            return False

        diction = {}
        for i, v in enumerate(order):
            diction[v] = i
        for i in range(1, len(words)):

            first = words[i - 1]
            second = words[i]
            if not is_sorted(first, second):
                return False
        return True

    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        queue = deque()
        queue.append((0, root))
        levels = defaultdict(list)
        while queue:
            level, node = queue.popleft()
            levels[level].append(node.val)
            if node.left:
                queue.append((level + 1, node.left))
            if node.right:
                queue.append((level + 1, node.right))

        result = []
        max_depth = max(list(levels.keys()))
        for i in range(max_depth + 1):
            if i in levels:
                val = levels[i][-1]
                result.append(val)
        return result

    def checkInclusion(self, s1: str, s2: str) -> bool:
        def check(arr):
            for i, v in enumerate(arr):
                if table[i] < arr[i]:
                    return False
            return True

        if len(s1) > len(s2):
            return False
        table = [0] * 27
        for s in s1:
            table[ord(s) - ord("a")] += 1

        arr = [0] * 27
        for i in range(0, len(s1)):
            arr[ord(s2[i]) - ord("a")] += 1

        for i in range(len(s1), len(s2)):
            if check(arr):
                return True
            arr[ord(s2[i - len(s1)]) - ord("a")] -= 1
            arr[ord(s2[i]) - ord("a")] += 1
        return check(arr)

    def sortedSquares(self, nums: List[int]) -> List[int]:
        for i, v in enumerate(nums):
            nums[i] = v * v

        def radix_sort(arr):
            max_digits = len(str(max(arr)))

            for i in range(max_digits):
                buckets = [[] for _ in range(10)]
                for num in arr:
                    idx = (num // 10**i) % 10
                    buckets[idx].append(num)
                j = 0
                for bucket in buckets:
                    for n in bucket:
                        arr[j] = n
                        j += 1
            return arr

        return radix_sort(nums)

    def customSortString(self, order: str, s: str) -> str:
        table = defaultdict(int)
        for i, v in enumerate(order):
            table[v] = i

        arr = list(s)
        arr.sort(key=lambda x: table[x])
        return "".join(arr)

    def treeToDoublyList(self, root: "Optional[Node]") -> "Optional[Node]":
        if not root:
            return root
        head = Node(-1)
        arr = [head]

        def dfs(node):
            if not node:
                return
            dfs(node.left)
            arr.append(node)
            dfs(node.right)

        dfs(root)
        for i in range(1, len(arr)):
            pre_node = arr[i - 1]
            suc_node = arr[i]
            pre_node.right = suc_node
            suc_node.left = pre_node
        if len(arr) == 2:
            return root
        if len(arr) > 2:
            last_node = arr[-1]
            first_node = arr[1]
            last_node.right = first_node
            first_node.left = last_node
        return arr[0].right

    def arraysIntersection(
        self, arr1: List[int], arr2: List[int], arr3: List[int]
    ) -> List[int]:
        counts = Counter(arr1, arr2, arr3)
        return

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        table = {}
        for n in nums2:
            table[n] = -1
        for i in range(len(nums2)):
            cur = nums2[i]
            while stack and stack[-1] < cur:
                num = stack.pop()
                table[num] = cur
            stack.append(cur)
        ans = []
        for n in nums1:
            ans.append(table[n])
        return ans

    def isPalindrome(self, s: str) -> bool:
        left = 0
        right = len(s) - 1
        while left < right:
            if not s[left].isalnum():
                left += 1
                continue
            if not s[right].isalnum():
                right -= 1
                continue
            if s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1
        return True

    def validPalindrome(self, s: str) -> bool:
        def valid(ss: list, chance: int, left, right):
            while left < right:
                if ss[left] != ss[right]:
                    if chance == 0:
                        chance += 1
                        one = valid(ss, chance, left + 1, right)
                        two = valid(ss, chance, left, right - 1)
                        return one or two
                    else:
                        return False
                left += 1
                right -= 1
            return True

        return valid(list(s), 0, 0, len(s) - 1)

    def addOperators(self, num: str, target: int) -> List[str]:
        operators = ["+", "-", "*"]

        def calculate(express):
            pass

        ans = set()
        nums = list(num)

        def backtrack(idx, expression):
            if idx == len(num):
                total = eval(expression)
                if total == target:
                    ans.add(expression)
                return
            elif idx >= len(num):
                return
            for size in range(idx, len(num)):
                if expression == "":
                    new_exp = "".join(nums[idx : size + 1])
                    backtrack(size + 1, new_exp)
                else:
                    for o in operators:
                        new_exp = expression + o + "".join(nums[idx : size + 1])
                        backtrack(size + 1, new_exp)
                if nums[idx] == "0":
                    break

        backtrack(0, "")
        return list(ans)

    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 1:
            return 0
        profit = 0
        min_price = float("inf")
        for price in prices:
            if price < min_price:
                min_price = price
            elif price - min_price > profit:
                profit = price - min_price
        return profit

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        total = 1
        zeros = []
        for i, n in enumerate(nums):
            if n == 0:
                zeros.append(i)
            else:
                total *= n
        if len(zeros) > 1:
            return [0] * len(nums)
        if len(zeros) == 1:
            zero_resp = [0] * len(nums)
            zero_resp[zeros[0]] = total
            return zero_resp
        ans = []
        for i, v in enumerate(nums):
            num = v**-1
            ans.append(int(num * total))
        return ans

    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        trie = {}

        def insert_in_trie(word: str, id: str) -> None:
            cur = trie
            for c in word:
                if c not in cur:
                    cur[c] = {}
                cur = cur[c]
                cur[id] = {}
            cur["*"] = {}

        def add_nums(arr, id):
            for n in arr:
                word = str(n)
                insert_in_trie(word, id)

        def find_longest(cur, level):
            top_outcome = level
            for k, v in cur.items():
                if "a1" in v and "a2" in v:
                    outcome = find_longest(v, level + 1)
                    top_outcome = max(top_outcome, outcome)
            return top_outcome

        add_nums(arr1, "a1")
        add_nums(arr2, "a2")
        cur = trie
        ans = find_longest(cur, 0)
        return ans

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        diff = float("inf")
        ans = 0

        for i in range(len(nums) - 2):
            left = i + 1
            right = len(nums) - 1

            while left < right:
                result = nums[i] + nums[left] + nums[right]

                if abs(result - target) < diff:
                    ans = result
                    diff = abs(result - target)

                if result < target:
                    left += 1
                elif result > target:
                    right -= 1
                else:
                    return target
        return ans

    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        heap = []
        for r in range(len(matrix)):
            for c in range(len(matrix[0])):
                if len(heap) < k:
                    heapq.heappush(heap, -matrix[r][c])
                else:
                    if heap[0] < -matrix[r][c]:
                        heapq.heappop(heap)
                        heapq.heappush(heap, -matrix[r][c])
        return heap[0]

    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        num1_count = Counter(nums1)
        num2_count = Counter(nums2)
        ans = []
        for n in num1_count:
            if n in num2_count:
                total = min(num1_count[n], num2_count[n])
                ans.extend([n] * total)
        return ans

    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        if k == 0:
            return 0
        table = defaultdict(int)
        left = 0
        ans = 0
        for right in range(len(s)):
            char = s[right]
            table[char] += 1
            while left < right and len(table) > k:
                l_char = s[left]
                table[l_char] -= 1
                if table[l_char] <= 0:
                    del table[l_char]
                left += 1
            ans = max(right - left + 1, ans)
        return ans

    def mergeTwoLists(
        self, list1: Optional[ListNode], list2: Optional[ListNode]
    ) -> Optional[ListNode]:
        head = ListNode(-1)
        curr = head
        cur1 = list1
        cur1 = list2
        while cur1 and cur2:
            if cur1.val <= cur2.val:
                curr.next = cur1
                cur1 = cur1.next
            else:
                curr.next = cur2
                cur2 = cur2.next
            curr = curr.next
        if cur2:
            curr.next = cur2
        if cur1:
            curr.next = cur1
        return head.next

    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        table = {}
        total = 0
        ans = 0
        for i in range(len(nums)):
            total += nums[i]
            if total == k:
                ans = i + 1
            if total - k in table:
                ans = max(ans, i - table[total - k])
            if total not in table:
                table[total] = i
        return ans

    def threeSum(self, nums: List[int]) -> List[List[int]]:

        nums.sort()
        ans = []
        seen = set()
        table = defaultdict(dict)
        for i, v in enumerate(nums):
            table[v][i] = 0
        i = 0
        while i < len(nums) - 2:
            for idx in range(i + 1, len(nums) - 1):
                total = nums[i] + nums[idx]
                cur_t = table[-total]
                if -total in table:
                    if len(table[-total]) > 2 or (idx not in cur_t and i not in cur_t):
                        right = list(table[-total].keys())
                        right = right[0]
                        candidate = [nums[i], nums[idx], nums[right]]
                        candidate.sort()
                        if tuple(candidate) not in seen:
                            ans.append(candidate)
                            seen.add(tuple(candidate))
            i += 1
        return ans


def test_solution():
    s = Solution()
    # print(s.subarraySum([1, -1, 0], 0))
    # print(s.subarraySum([1, 1, 1], 2))
    # print(s.subarraySum2(test, -93))


if __name__ == "__main__":
    test_solution()
