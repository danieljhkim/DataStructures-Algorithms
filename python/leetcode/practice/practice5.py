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


def test_solution():
    s = Solution()
    # arr = [7, 8, [3, 4]]
    # print(s.sum_num(arr))
    print(ord("A"))
    print(chr(500))


if __name__ == "__main__":
    test_solution()
