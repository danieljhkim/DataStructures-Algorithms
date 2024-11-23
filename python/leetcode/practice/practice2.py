from curses.ascii import isalpha
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

    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        counter = Counter(nums)
        ans = 0
        nums.sort()

        def bsearch_left(target):
            """_summary_
            1 2 3 4 4 4 6 7
            """
            low = 0
            high = len(nums) - 1
            while low <= high:
                mid = (high + low) // 2
                if nums[mid] < target:
                    low = mid + 1
                else:
                    high = mid - 1
            if low < len(nums) and nums[low] >= target:
                return low
            return 0

        def bsearch_right(target):
            low = 0
            high = len(nums) - 1
            while low <= high:
                mid = (high + low) // 2
                if nums[mid] <= target:
                    low = mid + 1
                else:
                    high = mid - 1
            if high >= 0:
                return high + 1
            return 0

        def how_many(target):
            left = bsearch_left(target - k)
            right = bsearch_right(target + k)
            count = right - left - counter[target]
            return min(count, numOperations) + counter[target]

        for n in nums:
            res = max(how_many(n), how_many(n + k))
            ans = max(ans, res)
        return ans

    def addBinary(self, a: str, b: str) -> str:
        def to_int(binary):
            num = 0
            for v in binary:
                num = num * 2 + int(v)
            return num

        def to_binary(num):
            ans = []
            while num:
                rem = num % 2
                num = num // 2
                ans.append(str(rem))
            ans.reverse()
            return "".join(ans)

        num_a = to_int(a)
        num_b = to_int(b)
        res = to_binary(num_a + num_b)
        if not res:
            return "0"
        return res

    def leastInterval(self, tasks: List[str], n: int) -> int:
        counts = list(Counter(tasks))
        counts.sort(key=lambda x: x[1])
        arr = []
        for task, count in counts:
            for i in range(count):
                slot = [task] + [None] * n
                arr.extend(slot)

    def isMatch(self, s: str, p: str) -> bool:
        """_summary_
        "aab"
        "c*a*b"
        """
        dqp = deque(list(p))
        dqs = deque(list(s))
        prevp = p[0]
        prevs = s[0]
        matching = []
        while dqp and (dqs or matching):

            pw = dqp[0]

            if pw == "*":
                matching.clear()
                if not dqs:
                    dqs.append(prevs)
                if prevp == ".":
                    prevp = dqs[0]

                while dqs and dqs[0] == prevp:
                    matching.append(dqs.popleft())
                dqp.popleft()
                continue
            if not dqs:
                dqs.append(matching.pop())
                if not matching and prevs:
                    matching.append(prevs)
            if pw == ".":
                prevp = dqp.popleft()
                prevs = dqs.popleft()
            else:
                if dqs[0] == dqp[0]:
                    prevs = dqs.popleft()
                prevp = dqp.popleft()
            if not dqs:
                if not matching and prevs:
                    matching.append(prevs)
        while dqp and dqp[-1] == "*":
            dqp.pop()
            if dqp:
                dqp.pop()
        while dqp and dqp[0] == "*":
            dqp.popleft()
        if dqp or dqs:
            return False
        return True


class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return "none"
        stuff = []
        queue = deque([root])
        level = 1
        while queue:
            count = 0
            for i in range(level):
                what = "none"
                if queue:
                    node = queue.popleft()
                    what = str(node.val)
                    if node.left:
                        queue.append(node.left)
                    else:
                        queue.append(TreeNode("none"))
                        count += 1
                    if node.right:
                        queue.append(node.right)
                    else:
                        queue.append(TreeNode("none"))
                        count += 1

                stuff.append(what + ":")
            if count == level * 2:
                for i in range(count):
                    queue.pop()
            level *= 2
        return "".join(stuff)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        1-2-3-none-4-none-none-
        """
        dq = deque(data.split(":"))
        empty = dq.pop()
        if empty == "none":
            return None

        root = TreeNode(int(dq.popleft()))
        queue = deque([root])
        while dq:

            left = dq.popleft()
            right = dq.popleft()
            if left == "none" and right == "none":
                if not queue[0]:
                    queue.popleft()
                continue
            node = queue.popleft()
            if left != "none":
                left = TreeNode(left)
            else:
                left = None

            if right != "none":
                right = TreeNode(right)
            else:
                right = None
            queue.append(left)
            queue.append(right)
            node.left = left
            node.right = right
        return root


def test_solution():
    """_summary_
        1
    2       3
        4
    """
    # s = Codec()
    # root = TreeNode(1)
    # root.left = TreeNode(2)
    # root.right = TreeNode(3)
    # root.left.right = TreeNode(4)
    # # aaa = s.serialize(root)
    # aa = "4:-7:-3:none:none:-9:-3:none:none:none:none:9:-7:-4:none:none:none:none:none:none:none:none:none:6:none:-6:-6:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:0:6:none:none:5:none:9:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:-1:-4:none:none:none:none:none:none:none:none:none:-2:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:"
    # s.deserialize(aa)

    # """_summary_
    # 4:-7:-3:none:none:-9:-3:none:none:none:none:9:-7:-4:none:none:none:none:none:none:none:none:none:6:none:-6:-6:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:0:6:none:none:5:none:9:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:-1:-4:none:none:none:none:none:none:none:none:none:-2:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:none:
    # """
    s1 = "mississippi"
    p1 = "mis*is*p*."

    s2 = "aaa"
    p2 = "a*a"
    s = Solution()
    s.isMatch("a", "ab*")


if __name__ == "__main__":
    test_solution()
