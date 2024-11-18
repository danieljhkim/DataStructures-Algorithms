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

    pass

    def quick_select(self, arr, k):
        """_summary_
        [1,2,3] [4,4,4] [7,8,9]
        """
        pivot = arr[-1]
        left = []
        right = []
        pivots = []
        for n in arr:
            if n < pivot:
                left.append(n)
            elif n > pivot:
                right.append(n)
            else:
                pivots.append(n)
        if len(right) >= k:
            return self.quick_select(right, k)
        elif len(pivots) + len(right) >= k:
            return pivot
        else:
            return self.quick_select(left, k - len(pivots) - len(right))

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        """_summary_
        [1,2,3] [4] [5,6,7]
        """
        distances = [p[0] ** 2 + p[1] ** 2 for p in points]

        def quick_select(arr, k):
            pivot = arr[-1]
            left = []
            right = []
            for n in arr:
                if distances[n] > distances[pivot]:
                    right.append(n)
                elif distances[n] < distances[pivot]:
                    left.append(n)
            if len(left) > k:
                return quick_select(left, k)
            elif len(left) < k:
                return left + quick_select(right + [pivot], k - len(left))
            else:
                return left

        arr = [i for i in range(points)]
        results = quick_select(arr, k)
        ans = []
        for idx in results:
            ans.append(points[idx])
        return ans

    def avg_of_window_size_k(self, arr, k):
        """_summary_
        [0, 1, 2, 3,] 4, 5
        """
        total = 0
        left = 0
        ans = []
        for right, n in enumerate(arr):
            if right < k:
                total += n
                continue
            if right - left + 1 == k:
                total += n
                ans.append(total / n)
                left += 1
                total -= arr[left]

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """_summary_
        1 -> 2 -> 3 -> none
        none <- 1 <- 2 <- 3
        """
        cur = head
        prev = None
        while cur:
            next_n = cur.next
            cur.next = prev
            prev = cur
            cur = next_n
        return prev

    def reverseList(self, head: ListNode) -> ListNode:
        if (not head) or (not head.next):
            return head

        p = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return p

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        result = []

        def dfs(node, path, total):
            if not node:
                return
            path.append(node.val)
            total += node.val
            if not node.left and not node.right:
                if targetSum == total:
                    result.append(path[::])
            dfs(node.left, path, total)
            dfs(node.right, path, total)
            path.pop()

        dfs(root, [], 0)
        return result

    def nextPermutation(self, nums: List[int]) -> None:
        """_summary_
        1 2 3 4 -> 1 2 4 3

        4 3 2 1 -> 1 2 3 4

        1 2 4 3 -> 1 3 4 2
        """

        n = len(nums)
        dec_idx = None
        for i in range(len(nums) - 1, 0, -1):
            prev = nums[i - 1]
            cur = nums[i]
            if prev < cur:
                dec_idx = i - 1
                break
        if dec_idx is None:
            nums.reverse()
            return
        i = n - 1
        while nums[dec_idx] >= nums[i]:
            i -= 1
        nums[dec_idx], nums[i] = nums[i], nums[dec_idx]
        nums[dec_idx + 1 :] = reversed(nums[dec_idx + 1 :])

    def maxLength(self, ribbons: List[int], k: int) -> int:
        """_summary_
        1 2 3 4 4 4 5 6
        """

        def possible(arr, length):
            count = 0
            for r in arr:
                if r >= length:
                    count += r // length
            return count >= k

        low = 1
        high = max(ribbons)
        while low <= high:
            mid = (low + high) // 2
            yes = possible(ribbons, mid)
            if yes:
                low = mid + 1
            else:
                high = mid - 1
        if high > 0 and possible(ribbons, high):
            return high
        return 0

    def sortArray(self, nums: List[int]) -> List[int]:

        def merge_sort(arr):
            if len(arr) < 2:
                return arr
            mid = len(arr) // 2
            left = merge_sort(arr[:mid])
            right = merge_sort(arr[mid:])
            l = r = j = 0
            while l < len(left) and r < len(right):
                if left[l] < right[r]:
                    arr[j] = left[l]
                    l += 1
                else:
                    arr[j] = right[r]
                    r += 1
                j += 1

            while l < len(left):
                arr[j] = left[l]
                l += 1
                j += 1
            while r < len(right):
                arr[j] = right[r]
                r += 1
                j += 1
            return arr

        return merge_sort(nums)

    def findCheapestPrice(
        self, n: int, flights: List[List[int]], src: int, dst: int, k: int
    ) -> int:
        """_summary_
        flights[i] = [from, to, price]
        """

        adj = defaultdict(list)
        for fr, to, pr in flights:
            adj[fr].append((to, pr))

        # (cost, current_node, stops)
        heap = [(0, src, 0)]
        while heap:
            cost, node, stops = heapq.heappop(heap)
            if node == dst:
                return cost
            if stops <= k:
                for neighbor, price in adj[node]:
                    heapq.heappush(heap, (cost + price, neighbor, stops + 1))
        return -1

    def majorityElement(self, nums: List[int]) -> List[int]:
        counter = Counter(nums)
        lim = nums / 3
        ans = []
        for k, v in counter.items():
            if v > lim:
                ans.append(k)
        return ans

    def frequencySort(self, s: str) -> str:
        counts = list(Counter(s).items())
        counts.sort(key=lambda x: x[1], reverse=True)
        ans = []
        for k, v in counts:
            for i in range(v):
                ans.append(k)
        return "".join(ans)

    def frequencySort(self, s: str) -> str:
        counts = list(Counter(s).items())
        low = min(counts, key=lambda x: x[1])[1]
        high = max(counts, key=lambda x: x[1])[1]
        xrange = high - low + 1
        buckets = [[] for _ in range(xrange)]
        for w, count in counts:
            idx = count - low
            buckets[idx].extend([w] * count)
        ans = []
        for bucket in reversed(buckets):
            ans.extend(bucket)
        return "".join(ans)

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        indegree = [0] * numCourses
        adj = defaultdict(list)
        for course, preq in prerequisites:
            adj[preq].append(course)
            indegree[course] += 1

        queue = deque()
        for i, n in enumerate(indegree):
            if n == 0:
                queue.append(i)

        topo_order = []
        while queue:
            cur = queue.popleft()
            topo_order.append(cur)
            for neigh in adj[cur]:
                indegree[neigh] -= 1
                if indegree[neigh] == 0:
                    queue.append(neigh)
        return len(topo_order) == numCourses

    def tribonacci(self, n: int) -> int:
        """_summary_
        0 1 1 2 4
        f4 + f3 + f2

        """
        if n <= 0:
            return 0
        if n <= 2:
            return 1
        return self.tribonacci(n - 1) + self.tribonacci(n - 2) + self.tribonacci(n - 3)

    def toGoatLatin(self, sentence: str) -> str:
        ans = []
        vowels = {"a", "e", "i", "o", "u"}
        sentence = sentence.split(" ")
        for i, w in enumerate(sentence):
            new_word = []
            if w[0].lower() not in vowels:
                new_word.append(w[1:])
                new_word.append(w[0])
            else:
                new_word.append(w)
            new_word.append("ma")
            new_word.append("a" * (i + 1))
            ans.append("".join(new_word))
        return " ".join(ans)


class SparseVector:
    def __init__(self, nums: List[int]):
        self.set = set()
        for i, n in enumerate(nums):
            if n != 0:
                self.set.add(i)
        self.nums = nums

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: "SparseVector") -> int:
        intersects = self.set.intersection(vec.set)
        res = 0
        for i in intersects:
            res += self.nums[i] * vec.nums[i]
        return res


def test_solution():
    s = Solution()
    arr = [1, 2, 3, 4, 5, 6, 7]
    s.allSubarraySum2(arr, 2)


if __name__ == "__main__":
    test_solution()
