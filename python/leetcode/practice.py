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
        ans = set()
        dups = set()
        for i, n1 in enumerate(nums):
            if n1 in dups:
                continue
            dups.add(n1)
            table = {}
            for j in range(i + 1, len(nums)):
                n2 = nums[j]
                target = -n1 - n2
                if target in table:
                    ans.add(tuple(sorted([n1, n2, target])))
                table[n2] = j
        return [list(x) for x in ans]

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans = []
        i = 0
        while i < len(nums) - 2:
            n1 = nums[i]
            table = {}
            j = i + 1
            while j < len(nums):
                n2 = nums[j]
                target = -n1 - n2
                if target in table:
                    ans.append([n1, n2, target])
                    table[n2] = j
                    while j < len(nums) and nums[j] == n2:
                        j += 1
                table[n2] = j
                j += 1
            while i < len(nums) and nums[i] == n1:
                i += 1
        return ans

    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        letters = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }
        ans = []

        def backtrack(word, digit_pos):
            if len(word) == len(digits):
                ans.append("".join(word))
                return
            letter = letters[digits[digit_pos]]
            for i in range(len(letter)):
                word.append(letter[i])
                backtrack(word, digit_pos + 1)
                word.pop()

        backtrack([], 0)
        return ans

    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        total_rem = 0
        table = {}
        for i, n1 in enumerate(nums):
            total_rem = (n1 + total_rem) % k
            if i > 0 and total_rem == 0:
                return True
            if total_rem in table:
                if i - table[total_rem] > 1:
                    return True
            else:
                table[total_rem] = i
        return False

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        word_set = set(wordDict)
        cache = {}

        def check(word):
            if word in cache:
                return cache[word]
            if len(word) == 0:
                return True
            for i in range(len(word)):
                if word[: i + 1] in word_set:
                    found = check(word[i + 1 :])
                    if found:
                        cache[word] = True
                        return found
            cache[word] = False
            return False

        return check(s)

    def depthSumInverse(self, nestedList: List[NestedInteger]) -> int:
        max_depth = 1
        table = defaultdict(list)

        def explore(nest, level):
            nonlocal max_depth
            if nest:
                max_depth = max(level - 1, max_depth)
            else:
                max_depth = max(level, max_depth)
            for n in nest:
                if n.isInteger():
                    table[level].append(n.getInteger())
                else:

                    explore(n.getList(), level + 1)

        explore(nestedList, 1)
        if not table:
            return 0
        ans = 0
        for i, v in table.items():
            for j in v:
                ans += j * (max_depth - i + 1)
        return ans

    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        total = 0

        def dfs(node):
            if not node:
                return
            nonlocal total
            if node.val > low:
                dfs(node.left)
            if node.val <= high and node.val >= low:
                total += node.val
            if node.val < high:
                dfs(node.right)

        dfs(root)
        return total

    def bulbSwitch(self, n: int) -> int:

        bulb = [True] * (n + 1)
        ans = 0
        for i in range(n, 0, -1):
            count = 1
            j = 2
            while j < i**0.5:
                if i % j == 0:
                    count += 1
                    j += j
                else:
                    j += 1

            if count % 2 == 1:
                ans += 1
        return ans


class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.ROW = len(matrix)
        self.COL = len(matrix[0])
        self.matrix = matrix
        self.total = 0
        self.calculate_row()
        self.calculate_col()
        self.rsum = []
        self.csum = []

    def calculate_row(self):
        for r in self.matrix:
            total = 0
            for c in range(self.COL):
                total += self.matrix[r][c]
            self.total += total
            self.rsum.append(total)

    def calculate_col(self):
        for c in self.matrix:
            total = 0
            for r in range(self.ROW):
                total += self.matrix[r][c]
                self.matrix[r][c] = total
            self.csum.append(total)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        # up

        total = self.rsum[row1 : row2 + 1]
        cols = 0
        for i, c in enumerate(self.csum):
            if i < col1 and i > col2:
                cols += self.csum[i]
        rows = 0
        for i, c in enumerate(self.rsum):
            if i < row1 and i > row2:
                rows += self.rsum[i]

    def isBalanced(self, num: str) -> bool:
        even_sum = 0
        odd_sum = 0
        for i, v in enumerate(list(num)):
            if i % 2 == 0:
                even_sum += int(v)
            else:
                odd_sum += int(v)
        return even_sum == odd_sum

    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        ROW = len(moveTime)
        COL = len(moveTime[0])
        ans = float("inf")
        visited = [[False] * COL for _ in range(ROW)]

        def dfs(time, r, c):
            nonlocal ans
            if r == ROW - 1 and c == COL - 1:
                ans = min(ans, time)
                return
            if time >= ans:
                return
            visited[r][c] = True

            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROW and 0 <= nc < COL and not visited[nr][nc]:
                    next_time = max(moveTime[nr][nc], time) + 1
                    dfs(next_time, nr, nc)

            visited[r][c] = False

        dfs(0, 0, 0)
        return ans

    def reverseVowels(self, s: str) -> str:
        vowels = ["a", "e", "i", "o", "u"]
        vowels = set(vowels)

        pos = []
        s = list(s)
        for i, v in enumerate(s):
            if v.lower() in vowels:
                pos.append(v)

        for i, v in enumerate(s):
            if v.lower() in vowels:
                s[i] = pos.pop()
        return "".join(s)

    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        queue = deque()
        queue.append((0, root))
        levels = {}
        while queue:
            level, node = queue.popleft()
            if level not in levels:
                levels[level] = node.val
            levels[level] = max(levels[level], node.val)
            if node.left:
                queue.append((level + 1, node.left))
            if node.right:
                queue.append((level + 1, node.right))
        ans = []
        top = max(list(levels.keys()))
        for i in range(top + 1):
            ans.append(levels[i])
        return ans

    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        ans = set(nums1)
        ans.intersection_update(nums2)
        return list(ans)

    def subtreeWithAllDeepest(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """_summary_
            0
        1       3
          2
        """

        def dfs(node, level):
            if not node:
                return level - 1, None
            left_l, left = dfs(node.left, level + 1)
            right_l, right = dfs(node.right, level + 1)

            if left_l == right_l:
                return left_l, node
            elif left_l > right_l:
                return left_l, left
            else:
                return right_l, right

        return dfs(root, 0)[1]

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        heap = []
        intervals.sort(key=lambda x: x[0])
        i = 0
        ans = 0
        while i < len(intervals):
            v = intervals[i]
            begin = v[0]
            end = v[1]
            if heap and heap[0][0] <= begin:
                heapq.heappop(heap)
            heapq.heappush(heap, (end, begin, i))
            ans = max(ans, len(heap))
            i += 1
        return ans

    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        parent = {}
        names = {}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x_root, y):
            rootx = find(x_root)
            rooty = find(y)
            if rootx != rooty:
                parent[rooty] = rootx

        for acc in accounts:
            name = acc[0]
            first_email = acc[1]
            for email in acc[1:]:
                parent[email] = email
                names[email] = name
                union(first_email, email)

        merged_emails = defaultdict(list)
        for email in parent:
            root = find(email)
            merged_emails[root].append(email)

        result = []
        for emails in merged_emails.values():
            result.append([names[emails[0]] + sorted(emails)])

        return result

    def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
        idx1 = 0
        idx2 = 0
        p1 = 0
        p2 = 0

        while p1 < len(word1) or p2 < len(word2):

            if idx1 >= len(word1[p1]):
                idx1 = 0
                p1 += 1

            if idx2 >= len(word2[p2]):
                idx2 = 0
                p2 += 1
            if idx2 == 0 and idx1 == 0:
                if p2 == len(word2) and p1 == len(word1):
                    return True
            if p2 >= len(word2) or p1 >= len(word1):
                return False

            w1 = word1[p1][idx1]
            w2 = word2[p2][idx2]

            idx1 += 1
            idx2 += 1
            if w1 != w2:
                return False

        return True

    def lowestCommonAncestor(self, p: "Node", q: "Node") -> "Node":
        curp = p
        curq = q
        pset = set()
        qset = set()
        while curp and curq:
            if curp.val == curq.val:
                return curp
            if curp.val in qset:
                return curp
            if curq.val in pset:
                return curq
            pset.add(curp.val)
            qset.add(curq.val)
            curp = curp.parent
            curq = curq.parent

        lower = p if curp else q
        hset = qset if curp else pset

        while lower.parent:
            if lower.val in hset:
                return lower
            lower = lower.parent
        return lower

    def longestArithSeqLength(self, nums: List[int]) -> int:
        """_summary_
        i1 - i = target
        i = i1 - target
        """
        cache = defaultdict(dict)

        def recursion(idx, target):
            if idx in cache:
                if target in cache[idx]:
                    return cache[idx][target]
            top = 1
            for i in range(idx + 1, len(nums)):
                diff = nums[i] - nums[idx]
                if diff == target:
                    res = recursion(i, target) + 1
                    top = max(res, top)
            cache[idx][target] = top
            return top

        ans = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                target = nums[j] - nums[i]
                res = recursion(j, target) + 1
                ans = max(res, ans)
        return ans

    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        queue = deque()
        queue.append((0, root))
        levels = defaultdict(list)
        max_depth = 0
        while queue:
            level, node = queue.popleft()
            max_depth = max(max_depth, level)
            if node.left:
                queue.append((level + 1, node.left))
                levels[level + 1].append(node)
            if node.right:
                queue.append((level + 1, node.right))
                levels[level + 1].append(node)

        last_row = levels[max_depth]
        if not last_row:
            return True
        if len(last_row) < max_depth * 2 - 1:
            return False
        for i in range(len(last_row) - 2):
            if not last_row[i].left or not last_row[i].right:
                return False
        if not last_row[-1].left and not last_row[-1].right:
            return True
        return False

    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        queue = deque([root])
        prev = root
        while queue:
            node = queue.popleft()
            if node:
                if not prev:
                    return False
                queue.append(node.left)
                queue.append(node.right)
            prev = node
        return True

    def buddyStrings(self, s: str, goal: str) -> bool:
        """_summary_
        aaabb
        aabab
        """
        ns = len(s)
        ng = len(goal)
        if ns != ng:
            return False
        first = None
        second = None
        for i in range(ns):
            if s[i] != goal[i]:
                if first is not None:
                    first = i
                elif second is not None:
                    second = i
                else:
                    return False

        if first is None and second is None:
            diff = set(list(s))
            if diff < ns:
                return True
        if second is None:
            return False
        if s[first] == goal[second] and s[second] == goal[first]:
            return True
        return False

    def minChanges(self, s: str) -> int:
        changes = 0
        for i in range(0, len(s), 2):
            if s[i] != s[i + 1]:
                changes += 1
        return changes


def test_solution():
    s = Solution()
    s.threeSum([])
    # print(s.subarraySum([1, -1, 0], 0))
    # print(s.subarraySum([1, 1, 1], 2))
    # print(s.subarraySum2(test, -93))


if __name__ == "__main__":
    test_solution()
