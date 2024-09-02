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

    # https://leetcode.com/problems/count-pairs-whose-sum-is-less-than-target/
    def countPairs(self, nums: List[int], target: int) -> int:
        ans = 0
        for i in range(0, len(nums) - 1):
            ii = nums[i]
            for j in range(i + 1, len(nums)):
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
        rsum = nums[length - 1]
        for i in range(1, length):
            left[i] = lsum
            lsum += nums[i]

        for i in range(length - 2, -1, -1):
            right[i] = rsum
            rsum += nums[i]

        for i in range(0, length):
            ans.append(abs(left[i] - right[i]))
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
            "9": "wxyz",
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
            if not root:
                return depth
            bigger = max(dfs(root.left, depth + 1), dfs(root.right, depth + 1))
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
        if (
            height_diff < 2
            and self.isBalanced(root.left)
            and self.isBalanced(root.right)
        ):
            return True
        return False

    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        total = len(nums)
        if not total:
            return None
        mid = total // 2
        return TreeNode(
            nums[mid],
            self.sortedArrayToBST(nums[:mid]),
            self.sortedArrayToBST(nums[mid + 1 :]),
        )

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
        for i in range(0, n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    total = nums[i] + nums[j] + nums[k]
                    if total == 0:
                        ans.add((nums[i], nums[j], nums[k]))
        return ans

    def threeSum(self, nums: List[int]) -> List[List[int]]:  # optimized
        n = len(nums)
        ans = set()
        nums.sort()
        for l in range(n - 2):
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
        # check horizontal for duplicates
        for row in board:
            hmap = {}
            for col in row:
                if col != ".":
                    if hmap.get(col) is None:
                        hmap[col] = 1
                    else:
                        return False

        # check verticle
        for i in range(9):
            rmap = {}
            for row in board:
                if row[i] != ".":
                    if rmap.get(row[i]) is None:
                        rmap[row[i]] = 1
                    else:
                        return False

        # check 3X3
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                cmap = {}
                for i in range(c, c + 3):
                    for j in range(r, r + 3):
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
            if n - 1 not in numSet:
                length = 1
                while n + length in numSet:
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
                    _permute(a_list, pos + 1)
                    a_list[pos], a_list[i] = a_list[i], a_list[pos]  # backtrack

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
                    _permute(a_list, pos + 1)
                    a_list[i], a_list[pos] = a_list[pos], a_list[i]

        _permute(nums, 0)
        return all_perms

    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if k == 0:
            return
        elif k < len(nums):
            nums[:] = nums[-k:] + nums[: len(nums) - k]
        else:
            for i in range(k):
                num = nums.pop()
                nums.insert(0, num)

    def maxProfit(self, prices: List[int]) -> int:
        profit = 0

        for i in range(0, len(prices) - 1):
            cur_price = prices[0]
            tmw_price = prices[i + 1]
            if cur_price < tmw_price:
                profit += tmw_price - cur_price
        return profit

    def canJump(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        index = 0
        while index < len(nums):
            max_step = nums[index]
            if max_step == 0:
                return False
            if (max_step + index + 1) >= len(nums):
                return True
            for i in range(index + 1, index + max_step + 1):
                if (max_step - (i - index)) <= nums[i]:
                    index = i
                    break
        return True

    def jump(self, nums: List[int]) -> int:
        jumps = 0
        if len(nums) == 1:
            return 0
        index = 0
        while index < len(nums) - 1:
            max_step = nums[index]
            if (max_step + index + 1) >= len(nums):
                return jumps + 1
            for i in range(index + 1, index + max_step + 1):
                if (max_step - (i - index)) <= nums[i]:
                    max_step = nums[i]
                    index = i
            jumps += 1
        return jumps

    def hIndex(self, citations: List[int]) -> int:
        citations.sort()
        max_index = len(citations)
        for i in citations:
            if i < max_index:
                max_index -= 1
            else:
                return max_index
        return max_index

    def intToRoman(self, num: int) -> str:
        num_map = {
            1: "I",
            5: "V",
            4: "IV",
            10: "X",
            9: "IX",
            50: "L",
            40: "XL",
            100: "C",
            90: "XC",
            500: "D",
            400: "CD",
            1000: "M",
            900: "CM",
        }
        result = ""
        for n in [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]:
            while n <= num:
                result += num_map[n]
                num -= n
        return result

    def reverseWords(self, s: str) -> str:
        s = s.strip()
        words = s.split()
        words.reverse()
        return " ".join(words)

    def reverseWords2(self, s: str) -> str:
        s = s.split()
        left = 0
        right = len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
        return " ".join(s)

    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        if not root.left and not root.right:
            return targetSum == root.val
        leftSum = self.hasPathSum(root.left, targetSum - root.val)
        rightSum = self.hasPathSum(root.right, targetSum - root.val)
        return leftSum or rightSum

    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtrack(first, curr):
            if len(curr) == k:
                output.append(curr[:])
                return
            for i in range(first, n + 1):
                curr.append(i)
                backtrack(i + 1, curr)
                curr.pop()

        output = []
        backtrack(1, [])
        return output

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left = 0
        sum_arr = 0
        min_len = float("inf")

        for right in range(len(nums)):
            sum_arr += nums[right]
            while sum_arr >= target:
                min_len = min(min_len, right - left + 1)
                sum_arr -= nums[left]
                left += 1
        if min_len == float("inf"):
            return 0
        return min_len

    def summaryRanges(self, nums: List[int]) -> List[str]:
        ans = []
        prev = nums[0]
        for i in range(1, len(nums)):
            diff = prev - nums[i]
            if diff > -1:
                ans.append(str(prev))
                prev = nums[i - 1]
            elif diff == 0:
                ans.append(str(prev))
                i += 1

    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        levels = {}

        def levelOrder(node, level):
            if not node:
                return
            if level in levels:
                levels[level][0] += node.val
                levels[level][1] += 1
            else:
                levels[level] = [node.val, 1]
            level += 1
            levelOrder(node.left, level)
            levelOrder(node.right, level)

        levelOrder(root, 0)
        res = []
        for v in levels.values():
            res.append(v[0] / v[1])
        return res

    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        ans = {"total": 0}

        def sumLeft(node, isLeft):
            if not node:
                return
            if not node.left and not node.right and isLeft:
                ans["total"] += node.val
                return
            if node.left:
                sumLeft(node.left, True)
            if node.right:
                sumLeft(node.right, False)

        sumLeft(root, False)

        return ans["total"]

    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        num_map = {}

        def find(node):
            if not node:
                return
            if node.val in num_map:
                num_map[node.val] += 1
            else:
                num_map[node.val] = 1
            find(node.left)
            find(node.right)

        find(root)
        sorted_nums = sorted(num_map.items(), key=lambda x: x[1], reverse=True)
        biggest = sorted_nums[0][1]
        ans = []
        for num in sorted_nums:
            if num[1] >= biggest:
                ans.append(num[0])
            else:
                break
        return ans

    def preorder(self, root: "Node") -> List[int]:
        ans = []

        def traverse(node):
            if not node:
                return
            ans.append(node.val)
            for chil in node.children:
                traverse(chil)

        traverse(root)
        return ans

    def postorder(self, root: "Node") -> List[int]:
        ans = []

        def traverse(node):
            if not node:
                return
            for chil in node.children:
                traverse(chil)
            ans.append(node.val)

        traverse(root)
        return ans

    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        total = [0]

        def postorder(node):
            if not node:
                return
            if node.val > high:
                postorder(node.left)
            elif node.val < low:
                postorder(node.right)
            else:
                total[0] += node.val
                postorder(node.left)
                postorder(node.right)

        postorder(root)
        return total[0]

    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        ans = []

        def traverse(node, path):
            if not node:
                return
            path += "->" + str(node.val)
            if not node.left and not node.right:
                ans.append(path)
            else:
                traverse(node.left, path)
                traverse(node.right, path)

        if not root.right and not root.left:
            return [str(root.val)]
        traverse(root.left, str(root.val))
        traverse(root.right, str(root.val))
        return ans

    def getTargetCopy(
        self, original: TreeNode, cloned: TreeNode, target: TreeNode
    ) -> TreeNode:
        if not original or target == original:
            return cloned
        left = self.getTargetCopy(original.left, cloned.left, target)
        right = self.getTargetCopy(original.right, cloned.right, target)
        return left or right

    def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
        answer = 0
        if not root:
            return 0

        def dfs(node, slate):
            nonlocal answer
            if not node.left and not node.right:
                slate.append(str(node.val))
                answer += int("".join(slate), 2)
            if node.left:
                dfs(node.left, slate + [str(node.val)])
            if node.right:
                dfs(node.right, slate + [str(node.val)])

        dfs(root, [])
        return answer

    def isUnivalTree(self, root: Optional[TreeNode]) -> bool:
        uni_val = root.val
        state = True

        def dfs(node):
            nonlocal state
            if not node:
                return
            if node.val != uni_val:
                state = False
                return
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return state

    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        ans = float("inf")
        val_list = []

        def dfs(node):
            if not node:
                return
            dfs(node.left)
            val_list.append(node.val)
            dfs(node.right)

        dfs(root)
        for i in range(0, len(val_list) - 1):
            ans = min(abs(val_list[i] - val_list[i + 1]))
        return ans

    def mergeTrees(
        self, root1: Optional[TreeNode], root2: Optional[TreeNode]
    ) -> Optional[TreeNode]:
        if root1 and root2:
            new_root = TreeNode(root1.val + root2.val)
            new_root.left = self.mergeTrees(root1.left, root2.left)
            new_root.right = self.mergeTrees(root1.right, root2.right)
            return new_root
        else:
            return root1 or root2

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        maxDiameter = 0

        def _diameter(root):
            if not root:
                return 0
            nonlocal maxDiameter
            left = _diameter(root.left)
            right = _diameter(root.right)
            maxDiameter = max(maxDiameter, left + right)
            return 1 + max(left, right)

        _diameter(root)
        return maxDiameter

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        size = len(nums)
        numMap = {}
        for i in range(size):
            numMap[nums[i]] = i
        for i in range(size):
            diff = target - nums[i]
            if diff in numMap and numMap[diff] != i:
                return [i, numMap[diff]]
        return []

    def isPowerOfFour(self, n: int) -> bool:
        if n <= 0:
            return False
        if n == 1:
            return True
        while n > 1:
            if n % 4 != 0:
                return False
            n //= 4
        return True

    def isPowerOfThree(self, n: int) -> bool:
        if n <= 0:
            return False
        if n == 1:
            return True
        while n > 1:
            if n % 3 != 0:
                return False
            n //= 3
        return True

    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        size = len(matrix)
        for row in range(0, int(size / 2)):
            for col in range(row, size - 1 - row):
                matrix[row][col], matrix[col][size - 1 - row] = (
                    matrix[col][size - 1 - row],
                    matrix[row][col],
                )

                matrix[row][col], matrix[size - 1 - row][size - 1 - col] = (
                    matrix[size - 1 - row][size - 1 - col],
                    matrix[row][col],
                )

                matrix[row][col], matrix[size - 1 - col][row] = (
                    matrix[size - 1 - col][row],
                    matrix[row][col],
                )

    def rotate2(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        for row in range(n):
            for col in range(row, n):
                matrix[row][col], matrix[col][row] = matrix[col][row], matrix[row][col]
        h = len(matrix)
        w = len(matrix[0])
        for row in range(h):
            for col in range(w // 2):
                matrix[row][col], matrix[row][w - col - 1] = (
                    matrix[row][w - col - 1],
                    matrix[row][col],
                )

    def insertGreatestCommonDivisors(
        self, head: Optional[ListNode]
    ) -> Optional[ListNode]:
        # O(nlogm)
        def find_gcd(num1, num2):
            # O(logn)
            while num2:
                num1, num2 = num2, num1 % num2
            return num1

        current = head
        while current.next:  # O(n)
            current.next = ListNode(
                find_gcd(current.val, current.next.val), current.next
            )
            current = current.next.next
        return head

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums = [(i, nums[i]) for i in range(len(nums))]
        nums.sort(key=lambda x: x[1])
        l = 0
        r = len(nums) - 1
        while l < r:
            total = nums[l][1] + nums[r][1]
            if total == target:
                return [nums[l][0], nums[r][0]]
            elif total < target:
                l += 1
            else:
                r -= 1
        return []

    # 3274. Check if Two Chessboard Squares Have the Same Color
    def checkTwoChessboards(self, coordinate1: str, coordinate2: str) -> bool:
        def is_white(coord):
            x = ord(coord[0])
            y = int(coord[1])
            if x % 2 == 0:
                if y % 2 == 0:
                    return False
                else:
                    return True
            else:
                if y % 2 == 0:
                    return True
                else:
                    return False

        c1 = is_white(coordinate1)
        c2 = is_white(coordinate2)
        return c1 == c2
