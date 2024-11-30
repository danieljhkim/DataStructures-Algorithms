from curses.ascii import isdigit
from functools import lru_cache
import heapq
from typing import Optional, List
import random
from collections import Counter, deque, defaultdict, OrderedDict
import math
import bisect


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

    def restoreString(self, s: str, indices: List[int]) -> str:
        """_summary_
        "codeleet"

        [4,5,6,7,0,2,1,3]
        """
        ans = [None] * len(s)
        for i in range(len(s)):
            ans[indices[i]] = s[i]
        return "".join(ans)

    def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
        """_summary_
        In 1 second, you can either:
        - move up or down
        - move side
        - move up and down
        - visit in order
        - You are allowed to pass through points that appear later in the order, but these do not count as visits.
        """
        ans = 0
        for i in range(len(points) - 1):
            prev = points[i]
            after = points[i + 1]
            x = abs(prev[0] - after[0])
            y = abs(prev[1] - after[1])
            ans += min(x, y) + abs(x - y)
        return ans

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        """_summary_
        # 7 8 9 | 1 2 3 4
        # 1 2 3 | 7 8 9 10

        # 6 7 8 9 | 1 2 3
        # 1 2 3 4 | 6 7 8

        # 0 | 1 2 3 4
        """
        n1 = len(nums1)
        n2 = len(nums2)
        total = n1 + n2
        rem = total % 2
        middle = total // 2

        low1 = 0
        high1 = n1 - 1
        n2_idx = 0
        while low1 <= high1:
            # choose left idx -> search nums2 and get next biggest

            mid1 = (low1 + high1) // 2
            n2_idx = bisect.bisect_left(nums1[mid1])
            length = mid1 + n2_idx
            if length > middle:
                high1 = mid1 - 1
            elif length < middle:
                low1 = mid1 + 1
            else:
                if rem == 0:
                    return nums1[mid1]
                else:
                    return (nums1[mid1] + nums2[n2_idx]) / 2
        mid = middle - low1
        if rem == 0:
            return nums2[mid]
        else:
            return (nums2[mid] + nums1[mid1]) / 2

    def alienOrder(self, words: List[str]) -> str:
        indegre = defaultdict(int)
        adj = defaultdict(set)
        for i, w in enumerate(words):
            gset = set(w)
            for i in range(1, len(w)):
                gset.remove(w[i - 1])
                adj[w[i - 1]].update(gset)
            if i < len(words) - 1:
                adj[words[i][0]].add(words[i + 1][0])
        for k, v in adj.items():
            indegre[k] = len(v)
        queue = deque()
        for k, v in indegre.items():
            if v == 0:
                queue.append(k)

        topo_sorted = []
        while queue:
            cur = queue.popleft()
            topo_sorted.append(cur)
            for neigh in adj[cur]:
                indegre[neigh] -= 1

                if indegre[neigh] == 0:
                    queue.append(neigh)
        if len(indegre) == len(topo_sorted):
            return "".join(topo_sorted)
        return ""

    def canPermutePalindrome(self, s: str) -> bool:
        n = len(s)
        counts = Counter(s)
        is_odd = True if n % 2 == 1 else False
        for k, v in counts:
            if v % 2 != 0:
                if not is_odd:
                    return False
                else:
                    is_odd = False
        return True

    def canPermutePalindrome(self, s: str) -> bool:
        count = 0
        counts = Counter(s)
        for v in counts.values():
            count += v % 2
        return count

    def isValidPalindrome(self, s: str, k: int) -> bool:
        """
        O(n^3)
        """
        if len(s) <= k:
            return True
        odds = self.canPermutePalindrome(s)
        if odds - 1 > k:
            return False
        memo = {}

        def recurs(s):
            n = len(s)
            if n == 0:
                return 0
            if s in memo:
                return memo[s]
            count = 0
            left = 0
            right = n - 1
            while left < right:
                if s[left] != s[right]:
                    lres = recurs(s[left + 1 : right + 1])
                    rres = recurs(s[left:right])
                    count = min(lres, rres) + 1
                    memo[s] = count
                    return count
                left += 1
                right -= 1
            memo[s] = count
            return count

        return recurs(s) <= k

    def isValidPalindrome(self, s: str, k: int) -> bool:
        n = len(s)
        # Create a 2D DP array
        dp = [[0] * n for _ in range(n)]

        # Fill the DP array
        for length in range(2, n + 1):  # length of the substring
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1]
                else:
                    dp[i][j] = min(dp[i + 1][j], dp[i][j - 1]) + 1

        # The result for the whole string is in dp[0][n-1]
        return dp[0][n - 1] <= k

    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        adj = defaultdict(list)

        def build(node, parent):
            if parent:
                adj[node.val].append(parent.val)
                adj[parent.val].append(node.val)
            if node.left:
                build(node.left, node)
            if node.right:
                build(node.right, node)

        build(root, None)
        visited = {target.val}
        ans = []

        def dfs(node, dist):
            if dist == k:
                ans.append(node)
                return
            for neigh in adj[node]:
                if neigh not in visited:
                    visited.add(neigh)
                    dfs(neigh, dist + 1)

        dfs(target.val, 0)
        return ans

    def countSubstrings(self, s: str) -> int:
        """
        TLE
        """
        memo = {}

        def is_palindrome(word):
            if word in memo:
                return memo[word]
            left = 0
            right = len(word) - 1
            while left < right:
                if word[left] != word[right]:
                    memo[word] = False
                    return False
                left += 1
                right -= 1
            memo[word] = True
            return True

        adj = defaultdict(list)
        for i, w in enumerate(s):
            adj[w].append(i)
        ans = 0
        seen = set()

        for i, w in enumerate(s):
            for neigh in adj[w]:
                if neigh >= i and (i, neigh) not in seen:
                    if is_palindrome(s[i : neigh + 1]):
                        ans += 1
                    seen.add((i, neigh))
        return ans

    def minWindow(self, s: str, t: str) -> str:
        """_summary_
        a: [0,1,4]
        b: [2,5,7]
        c: [6,9]

        """
        required = Counter(t)
        coordinates = []

        for i, w in enumerate(s):
            if w in required:
                coordinates.append((i, w))
        queue = deque()
        good = 0
        min_range = float("inf")
        start = 0
        end = 0
        cur = Counter()
        for idx, w in coordinates:
            cur[w] += 1
            queue.append((idx, w))
            if cur[w] == required[w]:
                good += 1
            if good == len(required):
                while queue and cur[queue[0][1]] > required[queue[0][1]]:
                    sidx, nw = queue.popleft()
                    cur[nw] -= 1
                cur_range = idx - queue[0][0] + 1
                if cur_range < min_range:
                    min_range = cur_range
                    start = queue[0][0]
                    end = idx
        if min_range != float("inf"):
            return s[start : end + 1]
        return ""

    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        prefix = [0]
        for n in nums:
            prefix.append(prefix[-1] + n)

        ans = float("inf")
        left = 0
        right = 1
        while right < len(prefix):
            total = prefix[right] - prefix[left]
            if total >= target:
                length = right - left
                ans = min(length, ans)
                if left < right:
                    left += 1
                else:
                    right += 1
            else:
                right += 1
        if ans == float("inf"):
            return 0
        return ans

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:

        def calc_dist(point):
            x = point[0]
            y = point[1]
            return x**2 + y**2

        def quick(arr, k):
            if len(arr) <= k:
                return arr
            left = []
            right = []
            pivots = []
            idx = random.randint(0, len(arr) - 1)
            pivot = calc_dist(arr[idx])
            for point in arr:
                dist = calc_dist(point)
                if dist < pivot:
                    left.append(point)
                elif dist > pivot:
                    right.append(point)
                else:
                    pivots.append(point)
            if len(left) >= k:
                return quick(left, k)
            elif len(left) + len(pivots) < k:
                return left + pivots + quick(right, k - len(pivots) - len(left))
            elif len(left) < k:
                return left + quick(pivots + right, k - len(left))
            else:
                return left + pivots[: k - len(left)]

        return quick(points, k)

    # 227
    def calculate(self, s: str) -> int:
        """_summary_
        3 * 2 + 2 + 1
        1     -  6   -

        """
        signs = {"+", "-", "/", "*", "#"}
        cur_num = ""
        prev_num = ""
        prev_sign = "+"
        total = 0
        s += "#"
        for n in s:
            if n == " ":
                continue
            if n.isdigit():
                cur_num = str(cur_num) + n
            elif n in signs:
                if n == "+" or n == "-" or n == "#":
                    if prev_sign == "*":
                        total += int(prev_num) * int(cur_num)
                    elif prev_sign == "/":
                        total += int(int(prev_num) / int(cur_num))
                    elif prev_sign == "-":
                        total -= int(cur_num)
                    elif prev_sign == "+":
                        total += int(cur_num)
                    cur_num = ""
                    prev_sign = n
                elif n == "*" or n == "/":
                    if prev_sign == "*":
                        prev_num = int(prev_num) * int(cur_num)
                    elif prev_sign == "/":
                        prev_num = int(int(prev_num) / int(cur_num))
                    elif prev_sign == "-":
                        prev_num = -int(cur_num)
                    else:
                        prev_num = int(cur_num)
                    prev_sign = n
                    cur_num = ""
        return total

    def findChampion(self, n: int, edges: List[List[int]]) -> int:

        indegree = [0] * n
        for a, b in edges:
            indegree[b] += 1

        tops = []
        for i, n in enumerate(indegree):
            if n == 0:
                tops.append(i)
        if len(tops) > 1:
            return -1
        return tops[0]

    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        ans = []
        queue = deque([(root)])
        isleft = True
        while queue:
            length = len(queue)
            level = []
            for i in range(length):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right, not isleft)
            if not isleft:
                level.reverse()
            if level:
                ans.append(level)
            isleft = not isleft
        return ans

    def str2tree(self, s: str) -> Optional[TreeNode]:
        """
        "4(2(3)(1))(6(5))"
        4(2(3)(1))(6(5)(7))
        """
        stack = []
        num = ""
        root = None
        for w in s:
            if w == "(":
                if len(num) > 0:
                    node = TreeNode(int(num))
                    stack.append(node)
                    num = ""
                else:
                    child = stack.pop()
                    parent = stack.pop()
                    parent.right = child
            elif w == ")":
                if len(num) > 0:
                    node = stack.pop()
                    new_node = TreeNode(int(num))
                    if not node.left:
                        node.left = new_node
                    elif not node.right:
                        node.right = new_node
                    stack.append(node)
                    num = ""
                else:
                    if len(stack) == 1:
                        return stack[0]
                    child = stack.pop()
                    parent = stack.pop()
                    if not parent.left:
                        parent.left = child
                        stack.append(parent)
                    elif not parent.right:
                        parent.right = child
            else:
                num += w
        print(stack)
        return root

    def __init__(self):
        self.clones = {}

    def cloneGraph(self, node: Optional["Node"]) -> Optional["Node"]:
        if not node:
            return node
        if node not in self.clones:
            clone = Node(node.val)
            self.clones[node] = clone
        else:
            return self.clones[node]
        for neigh in node.neighbors:
            clone_n = self.cloneGraph(neigh)
            clone.neighbors.append(clone_n)
        return clone

    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        pass

    def islandPerimeter(self, grid: List[List[int]]) -> int:

        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        ROW = len(grid)
        COL = len(grid[0])
        ans = 0

        def dfs(r, c):
            if r >= ROW or r < 0:
                return 1
            if c >= COL or c < 0:
                return 1
            if grid[r][c] == 0:
                return 1
            if grid[r][c] == "*":
                return 0
            grid[r][c] = "*"
            borders = 0
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                borders += dfs(nr, nc)
            return borders

        for r in range(ROW):
            for c in range(COL):
                if grid[r][c] == 1:
                    ans = dfs(r, c)
                    break
        return ans

    def reorderList(self, head: Optional[ListNode]) -> None:
        """_summary_
        1 -> 2 -> 3 -> 4 -> 5
        first last-0 second last-1 third last-2 fourth last-3
        """
        arr = []
        cur = head
        while cur:
            arr.append(cur)
            cur = cur.next

        left = 0
        right = len(arr) - 1
        queue = deque()
        while left <= right:
            if left == right:
                queue.append(arr[left])
                break
            queue.append(arr[left])
            queue.append(arr[right])
            left += 1
            right -= 1

        prev = queue.popleft()
        while queue:
            cur = queue.popleft()
            prev.next = cur
            prev = cur
        prev.next = None
        return head


class RandomizedSet:

    def __init__(self):
        self.random = {}
        self.table = {}
        self.idx = 0

    def insert(self, val: int) -> bool:
        if val in self.table:
            return False
        self.idx += 1
        self.table[val] = self.idx
        self.random[self.idx] = val
        return True

    def remove(self, val: int) -> bool:
        """_summary_
        - IMPORTANT: careful when removing the last idx ele
        """
        if val not in self.table:
            return False
        del_idx = self.table.pop(val)
        if del_idx == self.idx:  # right here
            del self.random[del_idx]
            self.idx -= 1
            return True
        old_val = self.random.pop(self.idx)
        self.random[del_idx] = old_val
        self.table[old_val] = del_idx
        self.idx -= 1
        return True

    def getRandom(self) -> int:
        idx = random.randint(1, self.idx)
        return self.random[idx]


def test_solution():
    s = Solution()
    ss = "2*3+4"

    s.calculate(ss)


if __name__ == "__main__":
    test_solution()
