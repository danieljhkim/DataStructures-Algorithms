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


def test_solution():
    s = Solution()


if __name__ == "__main__":
    test_solution()
