import heapq
from typing import Optional, List
import random
from collections import Counter, deque, defaultdict, OrderedDict
import math
import bisect
from math import inf
from functools import lru_cache


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

    def sum_num(self, arr):
        """_summary_
        arr = [7,8,[3,4]]
        7*1+8*2+3*3+4*3 = 44
        """

        def recursion(idx, i, cur, isArr):
            if i >= len(cur):
                return 0
            num = cur[i]
            total = 0
            if isArr == True:
                if isinstance(num, list):
                    total += recursion(idx, 0, num, True)
                else:
                    total += num + recursion(idx, i + 1, cur, True)
            else:
                if isinstance(num, list):
                    total += idx * recursion(idx, 0, num, True)
                else:
                    total += num * idx + recursion(idx + 1, i + 1, cur, False)
            return total

        return recursion(1, 0, arr, False)

    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        if n < 0:
            x = 1 / x
            n = abs(n)

        if n % 2 == 1:
            return x * self.myPow(x * x, n // 2)
        else:
            return self.myPow(x * x, n // 2)

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """ "
        - time
            - k = 4 (when n == avg len of each node)
                - [0,1,2], [3,4,5], [6,7,8], [9,10,11]
                - [0,1,2,3,4,5], [6,7,8,9,10,11]
                    - 2n + 2n
                - [0,1,2 ...]
                    - 4n
                - 2n + 2n + 4n = 8n
            - (when n == all # of nodes)
                - (1/k)*n*k + (2/k)*n*k/2 = 2n
        """

        def merge(list1, list2):
            cur1 = list1
            cur2 = list2
            res = ListNode(-1)
            head = res
            while cur1 and cur2:
                if cur1.val < cur2.val:
                    res.next = cur1
                    cur1 = cur1.next
                else:
                    res.next = cur2
                    cur2 = cur2.next
                res = res.next
            if cur1:
                res.next = cur1
            elif cur2:
                res.next = cur2
            return head.next

        queue = deque(lists)
        while len(queue) >= 2:
            first = queue.popleft()
            second = queue.popleft()
            queue.append(merge(first, second))
        return queue[0] if len(queue) > 0 else None

    def connect(self, root: "Optional[Node]") -> "Optional[Node]":
        if not root:
            return root
        queue = deque([root])
        while queue:
            llen = len(queue)
            prev = None
            for i in range(llen):
                cur = queue.popleft()
                if prev:
                    cur.next = prev
                prev = cur
                if cur.right:
                    queue.append(cur.right)
                if cur.left:
                    queue.append(cur.left)
        return root

    def longestWord(self, words: List[str]) -> str:
        diction = {}
        candidates = []
        for w in words:
            cur = diction
            for i, l in enumerate(w):
                if l not in cur:
                    cur[l] = {}
                cur = cur[l]
            cur["*"] = True

        words.sort(key=lambda x: len(x), reverse=True)
        biggest = -1
        for w in words:
            cur = diction
            good = True
            for l in w:
                cur = cur[l]
                if "*" not in cur:
                    good = False
                    break
            if good:
                if len(w) >= biggest:
                    biggest = len(w)
                    candidates.append(w)
                else:
                    break

        if len(candidates) == 0:
            return ""
        elif len(candidates) == 1:
            return candidates[0]
        else:
            first = candidates[0]
            options = []
            for c in candidates:
                if len(c) == len(first):
                    options.append(c)
                else:
                    break
            if len(options) == 1:
                return first
            options.sort()
            return options[0]

    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        """_summary_
        [5,0],[7,0],[6,1],[7,1],[5,2],[4,4]

        [5,0],[7,0],[5,2],[6,1],[4,4],[7,1]
        """
        adj = defaultdict(list)
        for i in range(len(people) - 1):
            hi, idx = people[i]
            for j in range(i + 1, len(people)):
                hj, jdx = people[j]

    def minCostToSupplyWater(
        self, n: int, wells: List[int], pipes: List[List[int]]
    ) -> int:
        """_summary_
        piples = [house1, house2, cost]
        wells = [cost1, cost2, cost3]
        n=houses
        """
        pipe_costs = {}
        degrees = [0] * n
        adj = defaultdict(list)
        for p in pipes:
            h1, h2, cost = p
            degrees[h2 - 1] += 1
            degrees[h1 - 1] += 1
            pipe_costs[(h1, h2)] = cost
            pipe_costs[(h2, h1)] = cost
            adj[h1].append(h2)
            adj[h2].append(h1)

        candidates = []
        total = 0
        temp_sum = 0
        for i, v in enumerate(degrees):
            if v == 0:
                total += wells[i]
            else:
                temp_sum += wells[i]
                candidates.append(i + 1)

        def dfs(node):
            total = 0
            for neigh in adj[node]:
                if neigh == node or neigh in piped_up:
                    continue
                if (node, neigh) not in visited:
                    cwell = wells[neigh - 1]
                    cpipe = pipe_costs[(node, neigh)]
                    if cpipe < cwell:
                        total += cpipe
                        piped_up.add(neigh)
                        if neigh in welled_up:
                            welled_up.remove(neigh)
                            total -= wells[neigh - 1]
                    else:
                        welled_up.add(neigh)
                        total += cwell
                    visited.add((node, neigh))
                    visited.add((neigh, node))
                    total += dfs(neigh)
            return total

        temp_sum = float("inf")

        for c in candidates:
            piped_up = set()
            welled_up = set()
            visited = set()
            costss = dfs(c) + wells[c - 1]
            for cc in candidates:
                if cc not in piped_up and c != cc:
                    costss += dfs(cc)
            temp_sum = min(temp_sum, costss)

        return temp_sum + total

    def __init__(self):
        self.pfound = False
        self.qfound = False

    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":

        def dfs(node):
            if not node:
                return node
            left = dfs(node.left)
            right = dfs(node.right)
            if node == p:
                self.pfound = True
                return node
            elif node == q:
                self.qfound = True
                return node
            if left and right:
                return node
            if left:
                return left
            else:
                return right

        lca = dfs(root)
        if self.pfound and self.qfound:
            return lca
        return None

    def checkIfExist(self, arr: List[int]) -> bool:
        """_summary_
        - i != j
        - arr[i] == 2 * arr[j]
        """
        nset = set()
        for n in arr:
            if n % 2 == 0:
                half = n / 2
                if half in nset:
                    return True
            if 2 * n in nset:
                return True
            nset.add(n)
        return False

    def longestPalindrome(self, s: str) -> str:
        """_summary_
        abcba
        abccba
        aaaba
        [b c b]
        [c c]
        """
        if len(s) <= 1:
            return s
        ans = s[0]
        ss = list(s)

        def is_palin(window):
            nonlocal ans
            left = window[0]
            right = window[-1]
            while left >= 0 and right < N:
                if s[left] != s[right]:
                    break
                left -= 1
                right += 1
            size = right - left - 1
            if size > len(ans):
                ans = ss[left + 1 : right]
            return right - 1

        N = len(s)
        window2 = deque()
        window3 = deque()
        visted = set()
        i = 0
        while i < N:
            cur = s[i]
            if i not in visted:
                if not window2:
                    window2.append(i)
                else:
                    if s[window2[-1]] != cur:
                        window2.pop()
                        window2.append(i)
                    else:
                        idx = i
                        while idx < N and cur == s[idx]:
                            window2.append(idx)
                            visted.add(idx)
                            idx += 1
                        is_palin(window2)
                        window2.clear()

            if not window3 or len(window3) < 2:
                window3.append(i)
            else:
                if s[window3[0]] != cur:
                    window3.popleft()
                    window3.append(i)
                else:
                    window3.append(i)
                    is_palin(window3)
                    window3.popleft()
            i += 1
        return "".join(ans)

    def isPrefixOfWord(self, sentence: str, searchWord: str) -> int:
        wlen = len(searchWord)
        arr = sentence.split(" ")
        for i, w in enumerate(arr):
            if len(w) < wlen:
                continue
            if w[:wlen] == searchWord:
                return i + 1
        return -1

    def sumOfUnique(self, nums: List[int]) -> int:
        counter = Counter(nums)
        total = 0
        for n, v in counter.items():
            if v == 1:
                total += n
        return total

    def pruneTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        self.no_parent = False

        def dfs(node, parent):
            if not node:
                return False
            left = dfs(node.left, node)
            right = dfs(node.right, node)
            if node.val == 0:
                if not left and not right:
                    if not parent:
                        self.no_parent = True
                        return False
                    if parent.left and parent.left == node:
                        parent.left = None
                    else:
                        parent.right = None
                    return False
                else:
                    return True
            else:
                return True

        dfs(root, None)
        if self.no_parent:
            return None
        return root

    def swimInWater(self, grid: List[List[int]]) -> int:
        # depth r c
        ROW = len(grid)
        COL = len(grid[0])
        directions = [(0, 1), (-1, 0), (1, 0), (0, -1)]
        time = grid[0][0]
        heap = [(time, 0, 0)]
        visited = set()
        visited.add((0, 0))
        while heap:
            t, r, c = heapq.heappop(heap)
            if r == ROW - 1 and c == COL - 1:
                return max(time, t)
            time = max(time, t)
            for dr, dc in directions:
                nr = dr + r
                nc = dc + c
                if 0 <= nr < ROW and 0 <= nc < COL and (nr, nc) not in visited:
                    depth = grid[nr][nc]
                    heapq.heappush(heap, (depth, nr, nc))
                    visited.add((nr, nc))
        return -1

    def isMonotonic(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        is_up = True
        is_down = True
        for i in range(1, len(nums)):
            prev = nums[i - 1]
            n = nums[i]
            if prev > n:
                is_up = False
            if prev < n:
                is_down = False
        return is_down or is_up

    def intervalIntersection(
        self, firstList: List[List[int]], secondList: List[List[int]]
    ) -> List[List[int]]:
        """_summary_
        [0, 100]
        [500,10]
        [9, 11]

        """
        if not firstList or not secondList:
            return []
        ans = []
        F = len(firstList)
        S = len(secondList)
        first = 0
        second = 0
        while first < F and second < S:
            fint = firstList[first]
            sint = secondList[second]
            if fint[1] < sint[0]:
                first += 1
                continue
            if sint[1] < fint[0]:
                second += 1
                continue

            start = max(sint[0], fint[0])
            end = min(sint[1], fint[1])
            ans.append([start, end])
            if sint[1] > fint[1]:
                first += 1
            elif fint[1] > sint[1]:
                second += 1
            else:
                first += 1
                second += 1
        return ans

    def numDecodings(self, s: str) -> int:
        N = len(s)
        sset = set()

        for i in range(N):
            n = int(s[i])
            if n == 0:
                if i == 0:
                    return 0
                prev = int(s[i - 1])
                if prev > 2 or prev == 0:
                    return 0

        def recursion(idx, arr):
            if idx >= len(s) - 1:
                sset.add(".".join(arr))
                return
            n = int(s[idx])
            nxt = int(s[idx + 1])
            if ((n == 2 and nxt < 8) or n == 1) and nxt != 0:
                arr.append(s[idx : idx + 2])
                recursion(idx + 2, arr)
                arr.pop()
            recursion(idx + 1, arr)

        recursion(0, [])

        return len(sset)

    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        """_summary_
        [name, email..]
        """
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            else:
                if parent[x] != x:
                    parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rootx = find(x)
            rooty = find(y)
            if rootx != rooty:
                parent[rootx] = parent[rooty]

        for acc in accounts:
            root = acc[1]
            for email in acc[2:]:
                union(root, email)

        table = defaultdict(set)
        names = {}
        ans = []
        for acc in accounts:
            name = acc[0]
            root = find(acc[1])
            key = name + "#" + root
            names[key] = root
            for email in acc[1:]:
                root = find(email)
                table[root].add(email)

        for key, root in names.items():
            keys = key.split("#")
            name = keys[0]
            root = keys[1]
            res = [name]
            emails = list(table[root])
            emails.sort()
            res.extend(emails)
            ans.append(res)
        return ans

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1

        N1, N2 = len(nums1), len(nums2)
        low, high = 0, N1

        while low <= high:
            partition1 = (low + high) // 2
            partition2 = (N1 + N2 + 1) // 2 - partition1

            maxLeft1 = float("-inf") if partition1 == 0 else nums1[partition1 - 1]
            minRight1 = float("inf") if partition1 == N1 else nums1[partition1]

            maxLeft2 = float("-inf") if partition2 == 0 else nums2[partition2 - 1]
            minRight2 = float("inf") if partition2 == N2 else nums2[partition2]

            if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:
                if (N1 + N2) % 2 == 0:
                    return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2
                else:
                    return max(maxLeft1, maxLeft2)
            elif maxLeft1 > minRight2:
                high = partition1 - 1
            else:
                low = partition1 + 1

        raise ValueError("Input arrays are not sorted.")

    def findAnagrams(self, s: str, p: str) -> List[int]:
        if len(s) < len(p):
            return []
        pcount = Counter(p)
        left = 0
        right = P - 1
        N = len(s)
        P = len(p)
        ans = []
        counts = Counter(s[:P])
        if counts == pcount:
            ans.append(left)
        while right < N:
            counts[s[left]] -= 1
            left += 1
            right += 1
            counts[s[right]] += 1
            if counts == pcount:
                ans.append(left)
        return ans

    def lcaDeepestLeaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        self.deepest = 0
        if not root:
            return None
        if not root.left and not root.right:
            return root

        def dfs(node, level):
            if not node:
                self.deepest = max(self.deepest, level - 1)
                return node, level - 1
            left, l_level = dfs(node.left, level + 1)
            right, r_level = dfs(node.right, level + 1)
            top = max(l_level, r_level)
            if left and right:
                if l_level == r_level and top >= self.deepest:
                    return node, top

            if not left and not right and top == level:
                return node, level

            outcome = (left, l_level) if l_level > r_level else (right, r_level)
            return outcome

        return dfs(root, 0)[0]

    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2 != 0:
            return False

        @lru_cache(maxsize=None)
        def recursion(sum1, pos):
            if len(nums) == pos:
                if sum1 == 0:
                    return True
                else:
                    return False
            option = nums[pos]
            result1 = recursion(sum1 - option, pos + 1)
            if result1:
                return True
            result2 = recursion(sum1, pos + 1)
            if result2:
                return True
            return False

        total //= 2
        return recursion(0, 0, 0, total)

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.top = 0

        def dfs(node, leng):
            if not node:
                return 0
            left = dfs(node.left, leng)
            right = dfs(node.right, leng)
            if left + right > self.top:
                self.top = left + right
            return max(left, right) + 1

        dfs(root, 0)
        return self.top

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: (x[0], x[1]))
        heap = []
        ans = 0
        for start, end in intervals:
            while heap and heap[0][0] <= start:
                heapq.heappop(heap)
            heapq.heappush(heap, (end, start))
            ans = max(len(heap), ans)
        return ans

    def fib(self, n: int) -> int:
        if n == 1:
            return 1
        if n <= 0:
            return 0
        return self.fib(n - 2) + self.fib(n - 1)

    def distributeCandies(self, candyType: List[int]) -> int:
        nset = set(candyType)
        n = len(candyType) // 2
        return min(len(nset), n)

    def trimBST(
        self, root: Optional[TreeNode], low: int, high: int
    ) -> Optional[TreeNode]:

        def dfs(node):
            if not node:
                return None
            left = dfs(node.left)
            right = dfs(node.right)

            cur = node.val
            if cur < low:
                if right:
                    if right.val >= low:
                        return right
                return None
            if cur > high:
                if left:
                    if left.val <= high:
                        return left
                return None
            node.left = left
            node.right = right
            return node

        return dfs(root)

    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:

        def reverse(node):
            cur = node
            prev = None
            while cur:
                nxt = cur.next
                cur.next = prev
                prev = cur
                cur = nxt
            return prev

        cur1 = reverse(l1)
        cur2 = reverse(l2)
        print(cur2)
        dummy = ListNode(-1)
        cur = dummy
        rem = 0
        while cur1 or cur2 or rem:
            val1 = cur1.val if cur1 else 0
            val2 = cur2.val if cur2 else 0
            new_val = val1 + val2 + rem
            rem = new_val // 10
            cur.next = ListNode(new_val % 10)
            if cur1:
                cur1 = cur1.next
            if cur2:
                cur2 = cur2.next

            cur = cur.next
        return reverse(dummy.next)

    def addSpaces(self, s: str, spaces: List[int]) -> str:
        stack = []
        idx = 0
        N = len(spaces)
        for i, n in enumerate(s):
            if idx < N and i == spaces[idx]:
                stack.append(" ")
                idx += 1
            stack.append(n)
        return "".join(stack)

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i1 = 0
        i2 = 0
        N1 = len(nums1)
        N2 = len(nums2)
        dq = deque(nums1[: N1 - N2])
        while dq and i2 < N2:
            n2 = nums2[i2]
            n1 = dq[0]
            if n2 < n1:
                nums1[i1] = n2
                i2 += 1
            else:
                nums1[i1] = dq.popleft()
            i1 += 1
        while dq:
            nums1[i1] = dq.popleft()
            i1 += 1
        while i2 < N2:
            nums1[i1] = nums2[i2]
            i1 += 1
            i2 += 1

    def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> Optional[TreeNode]:
        successor = None
        while root:
            if p.val < root.val:
                successor = root
                root = root.left
            else:
                root = root.right
        return successor

    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        def invert(row):
            left = 0
            right = ROW - 1
            while left <= right:
                l = row[left]
                r = row[right]
                row[left] = 1 if r == 0 else 0
                row[right] = 1 if l == 0 else 0
                left += 1
                right -= 1
            return row

        ROW = len(image)
        for i in range(ROW):
            image[i] = invert(image[i])
        return image

    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        self.leaf = deque()

        def dfs(node, is_first):
            if not node or (not is_first and not self.ans):
                return node
            left = dfs(node.left, is_first)
            right = dfs(node.right, is_first)
            if not left and not right:
                if is_first:
                    self.leaf.append(node.val)
                elif self.ans:
                    if self.leaf and self.leaf[0] == node.val:
                        self.leaf.popleft()
                    else:
                        self.ans = False
            return node

        self.ans = True
        dfs(root1, True)
        dfs(root2, False)
        if self.ans and not self.leaf:
            return True
        return False

    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        i = 0
        N = len(str1)
        if N < len(str2):
            return False
        for idx, s in enumerate(str2):
            found = False
            while i < N and not found:
                s1 = str1[i]
                c1 = chr((ord(s1) - ord("a") + 1) % 26 + ord("a"))
                if s1 == s or c1 == s:
                    found = True
                i += 1
            if not found or (i == N and idx < len(str2) - 1):
                return False
        return True

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        N = len(nums)

        def twoSum(target, start):
            table = set()
            ans = []
            for i in range(start, N):
                cur = nums[i]
                comp = target - cur
                if comp in table:
                    ans.append((cur, comp))
                table.add(cur)
            return ans

        visited = set()
        ans = set()
        for i, n in enumerate(nums[:-2]):
            if n not in visited:
                res = twoSum(-n, i + 1)
                if res and n not in visited:
                    for pairs in res:
                        ans.add(tuple(sorted((n, pairs[0], pairs[1]))))
                visited.add(n)
        return [list(x) for x in ans]

    def canChange(self, start: str, target: str) -> bool:
        if start.replace("_", "") != target.replace("_", ""):
            return False
        j = 0
        for i in range(len(start)):
            if start[i] == "_":
                continue
            while target[j] == "_":
                j += 1
            if start[i] != target[j]:
                return False
            if start[i] == "L" and i < j:
                return False
            if start[i] == "R" and i > j:
                return False
            j += 1

        return True

    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        nxt = node.next
        node.val = nxt.val
        node.next = nxt.next

    def hammingWeight(self, n: int) -> int:
        binary = bin(n)
        return binary.count("1")

    def rotateString(self, s: str, goal: str) -> bool:
        dq = deque(s)
        count = len(s) + 1
        while count > 0:
            while dq[0] == goal[0] and count > 0:
                dq.append(dq.popleft())
                count -= 1
            if "".join(dq) == goal:
                return True
            dq.append(dq.popleft())
            count -= 1
        return "".join(dq) == goal

    def majorityElement(self, nums: List[int]) -> List[int]:
        candidate1 = None
        candidate2 = None
        vote1 = 0
        vote2 = 0
        for n in nums:
            if n == candidate1:
                vote1 += 1
            elif n == candidate2:
                vote2 += 1
            elif vote1 == 0:
                candidate1 = n
                vote1 += 1
            elif vote2 == 0:
                vote2 += 1
                candidate2 = n
            else:
                vote1 -= 1
                vote2 -= 1
        limit = len(nums) // 3
        ans = []
        if nums.count(candidate1) > limit:
            ans.append(candidate1)
        if nums.count(candidate2) > limit:
            ans.append(candidate2)
        return ans

    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        0,1,2
        """
        buckets = [0] * 3
        for n in nums:
            buckets[n] += 1
        idx = 0
        for i, freq in enumerate(buckets):
            for j in range(freq):
                nums[idx] = i
                idx += 1

    def removeKdigits(self, num: str, k: int) -> str:
        if len(num) == k:
            return "0"
        stack = []
        pick = len(num) - k
        for i, n in enumerate(num):
            if not stack and n == "0":
                continue
            while stack and int(stack[-1]) > int(n) and k > 0:
                stack.pop()
                k -= 1
            stack.append(n)
        while k > 0:
            stack.pop()
            k -= 1
        if not stack:
            return "0"
        return "".join(stack)

    def dayOfTheWeek(self, day: int, month: int, year: int) -> str:

        months = {
            1: 31,
            2: 28,  # 29 in a leap year
            3: 31,
            4: 30,
            5: 31,
            6: 30,
            7: 31,
            8: 31,
            9: 30,
            10: 31,
            11: 30,
            12: 31,
        }
        week = {
            0: "Wednesday",
            1: "Thursday",
            2: "Friday",
            3: "Saturday",
            4: "Sunday",
            5: "Monday",
            6: "Tuesday",
        }

        def is_leap_year(year: int) -> bool:
            if year % 4 == 0:
                if year % 100 == 0:
                    if year % 400 == 0:
                        return True
                    else:
                        return False
                else:
                    return True
            else:
                return False

        def calc_days(year):
            total_days = 0
            for i in range(1971, year):
                if is_leap_year(i):
                    total_days += 366
                else:
                    total_days += 365
            return total_days

        cur_days = calc_days(2025)
        total_days = calc_days(year)
        cur_days += 1

        for m in range(1, month):
            total_days += months[m]

        if month > 2 and is_leap_year(year):
            total_days += 1
        total_days += day
        diff = (total_days - cur_days) % 7
        return week[diff]

    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        table = defaultdict(list)
        queue = deque([root, 0])
        start = float("inf")
        end = float("-inf")
        while queue:
            l = len(queue)
            tmap = defaultdict(list)
            for i in range(l):
                cur, level = queue.popleft()
                start = min(level, start)
                end = max(level, end)
                tmap[level].append(cur.val)
                if cur.left:
                    queue.append((cur.left, level - 1))
                if cur.right:
                    queue.append((cur.right, level + 1))
            for k, v in tmap.items():
                if len(v) > 1:
                    table[k].extend(sorted(v))
                else:
                    table[k].extend(v)
        ans = []
        for i in range(start, end + 1):
            ans.append(table[i])
        return ans

    def maxCount(self, banned: List[int], n: int, maxSum: int) -> int:
        """_summary_
        - 1 - n
        - return max # of integers can be chosen
        """
        banned = set(banned)
        total = 0
        i = 1
        count = 0
        while total <= maxSum and i <= n:
            if i not in banned:
                if total + i <= maxSum:
                    total += i
                    count += 1
                else:
                    break
            i += 1
        return count

    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:

        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(matrix)
        COL = len(matrix[0])
        memo = [[-1] * COL for _ in range(ROW)]

        def dfs(r, c):
            if memo[r][c] != -1:
                return memo[r][c]
            cur = matrix[r][c]
            count = 1
            top = 0
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < ROW and 0 <= nc < COL:
                    nxt = matrix[nr][nc]
                    if nxt < cur:
                        top = max(dfs(nr, nc), top)

            memo[r][c] = count + top
            return count + top

        ans = 0
        for r in range(ROW):
            for c in range(COL):
                total = dfs(r, c)
                ans = max(total, ans)
        return ans

    def isStrobogrammatic(self, num: str) -> bool:
        """_summary_
        - 1 8 0
        - 6 9
        """
        N = len(num)

        left = 0
        right = N - 1
        nset = set(["1", "8", "0"])

        if N == 1:
            if num[0] in ["6", "9"] or num[0] not in nset:
                return False
        while left <= right:
            ln = num[left]
            rn = num[right]
            if (ln == "6" and rn == "9") or (rn == "6" and ln == "9"):
                left += 1
                right -= 1
                continue
            if ln in nset and rn in nset:
                if ln != rn:
                    return False
            else:
                return False
            right -= 1
            left += 1
        return True

    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        row_1, col_1 = len(mat1), len(mat1[0])
        row_2, col_2 = len(mat2), len(mat2[0])

        result = [[0] * col_2 for _ in range(row_1)]
        for i in range(row_1):
            for j in range(col_2):
                for k in range(col_1):
                    result[i][j] += mat1[i][k] * mat2[k][j]
        return result

    def treeToDoublyList(self, root: "Optional[Node]") -> "Optional[Node]":
        if not root:
            return root
        self.arr = []

        def dfs(node):
            if node.left:
                dfs(node.left)
            self.arr.append(node)
            if node.right:
                dfs(node.right)

        dfs(root)
        head = Node(
            -1,
        )
        prev = self.arr[0]
        head.right = prev
        last = self.arr[-1]
        prev.left = last
        last.right = prev
        for node in self.arr[1:]:
            node.left = prev
            prev.right = node
            prev = node
        return head.right

    def shortestDistance(self, grid: List[List[int]]) -> int:
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(grid)
        COL = len(grid[0])
        dests = set()
        memo = [[0] * COL for _ in range(ROW)]
        table = {}

        def check_connection(r, c):
            for dest in dests:
                if (r, c) not in table[dest]:
                    return False
            return True

        for r in range(ROW):
            for c in range(COL):
                if grid[r][c] == 1:
                    dests.add((r, c))
                    memo[r][c] = inf
                elif grid[r][c] == 2:
                    memo[r][c] = inf

        def bfs(r, c):
            visited = set()
            queue = deque([(r, c, 0)])
            count = 1
            while queue:
                cr, cc, cost = queue.popleft()
                if (cr, cc) in visited:
                    continue
                visited.add((cr, cc))
                if memo[cr][cc] != inf:
                    memo[cr][cc] += cost
                for dr, dc in directions:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < ROW and 0 <= nc < COL:
                        nxt = grid[nr][nc]
                        if nxt == 0 and (nr, nc) not in visited:
                            queue.append((nr, nc, cost + 1))
                        elif (
                            (nr, nc) in dests
                            and (nr, nc) not in visited
                            and grid[cr][cc] != 1
                        ):
                            visited.add((nr, nc))
                            count += 1
            table[(r, c)] = visited
            return count

        builds = len(dests)
        for dest in dests:
            count = bfs(dest[0], dest[1])
            if count != builds:
                return -1
        ans = inf
        for r in range(ROW):
            for c in range(COL):
                cost = memo[r][c]
                if cost == 0 or cost == inf:
                    continue
                if check_connection(r, c):
                    ans = min(cost, ans)
        return ans if ans != inf else -1

    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        N = len(nums)
        FN = N + maxOperations

        def helper(min):
            res = 0
            for n in nums:
                if n > min:
                    rem = n % min
                    out = n // min
                    if rem > 0:
                        out += 1
                    res += out
                else:
                    res += 1
                if res > FN:
                    return False
            return True

        low = 1
        high = max(nums)
        found = -1
        while low <= high:
            mid = (low + high) // 2
            if helper(mid):
                high = mid - 1
                found = mid
            else:
                low = mid + 1
        return found

    def isPalindrome(self, x: int) -> bool:
        if x < 0:
            return False
        x = str(x)
        left = 0
        right = len(x) - 1
        while left <= right:
            if x[left] != x[right]:
                return False
            left += 1
            right -= 1
        return True

    def addStrings(self, num1: str, num2: str) -> str:
        table = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
        }
        carry = 0
        n1 = len(num1) - 1
        n2 = len(num2) - 1
        ans = deque()
        while n1 >= 0 or n2 >= 0 or carry:
            val1 = table[num1[n1]] if n1 >= 0 else 0
            val2 = table[num2[n2]] if n2 >= 0 else 0
            res = val1 + val2 + carry
            carry = res // 10
            ans.appendleft(str(res % 10))
            n1 -= 1
            n2 -= 1

        return "".join(ans)

    def defangIPaddr(self, address: str) -> str:
        arr = []
        for s in address:
            if s == ".":
                arr.append("[.]")
            else:
                arr.append(s)
        return "".join(arr)

    def minTotalDistance(self, grid: List[List[int]]) -> int:
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        ROW = len(grid)
        COL = len(grid[0])
        friends = set()

        for r in range(ROW):
            for c in range(COL):
                if grid[r][c] == 1:
                    grid[r][c] = 0
                    friends.add((r, c))
        if len(friends) == ROW * COL:
            return ROW * ROW + COL * COL

        def bfs(friend):
            visited = set()
            visited.add(friend)
            dq = deque([(friend[0], friend[1], 0)])
            while dq:
                r, c, dist = dq.popleft()
                grid[r][c] += dist
                for dr, dc in directions:
                    nr = dr + r
                    nc = dc + c
                    new_coord = (nr, nc)
                    if 0 <= nr < ROW and 0 <= nc < COL and new_coord not in visited:
                        visited.add(new_coord)
                        dq.append((nr, nc, dist + 1))

        for friend in friends:
            bfs(friend)

        ans = inf
        for r in range(ROW):
            for c in range(COL):
                ans = min(grid[r][c], ans)
        return ans


class MyQueue:

    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x: int) -> None:
        if not self.stack2:
            self.stack2.append(x)
        else:
            self.stack1.append(x)

    def pop(self) -> int:
        out = self.stack2.pop()
        if not self.stack2:
            while self.stack1:
                nxt = self.stack1.pop()
                self.stack2.append(nxt)
        return out

    def peek(self) -> int:
        return self.stack2[-1]

    def empty(self) -> bool:
        return len(self.stack2) == 0


def test_solution():
    s = Solution()
    # arr = [7, 8, [3, 4]]
    # print(s.sum_num(arr))
    # arr1 = [1, 2, 3, 4, 5, 6]
    # arr2 = [3, 4, 5, 6, 7, 8, 9]
    # s.findMedianSortedArrays(arr1, arr2)
    print(1 % 4)


if __name__ == "__main__":
    test_solution()
