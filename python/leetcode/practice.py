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


def test_solution():
    s = Solution()
    # print(s.subarraySum([1, -1, 0], 0))
    # print(s.subarraySum([1, 1, 1], 2))
    # print(s.subarraySum2(test, -93))


if __name__ == "__main__":
    test_solution()
