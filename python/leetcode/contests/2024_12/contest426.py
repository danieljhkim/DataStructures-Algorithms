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

    def smallestNumber(self, n: int) -> int:

        while True:
            binary = bin(n)[2:]
            if binary.count(0) > 0:
                n += 1
            else:
                return n

    def getLargestOutlier(self, nums: List[int]) -> int:
        """_summary_
        n - 2 are special nums
        rem 2:
            1: sum of the specials
            2: outlier
        return largest outlier

        n1 = total - n1 - outlier
        outlier = total - 2 * n1

        outlier

        [6,-31,50,-35,41,37,-42,13] = -35

        13

        """
        total = sum(nums)
        nset = set(nums)
        ans = float("-inf")
        counter = Counter(nums)

        for n in nums:
            outlier = total - 2 * n
            if outlier in nset:
                n1 = total - n - outlier
                if n1 in nset:
                    if n1 == outlier and counter[n1] == 1:
                        continue
                    ans = max(outlier, ans)
        return ans

        # def find(x, parent):
        #     if x not in parent:
        #         parent[x] = x
        #     else:
        #         if parent[x] != x:
        #             parent[x] = find(parent[x])
        #     return parent[x]

        # def union(x, y, parent):
        #     rootx = find(x, parent)
        #     rooty = find(y, parent)
        #     if rootx != rooty:
        #         parent[rootx] = parent[rooty]

    def maxTargetNodes(
        self, edges1: List[List[int]], edges2: List[List[int]], k: int
    ) -> List[int]:
        """_summary_
        - undirected tree: n and m nodes
        - distinct labels in range : [0, n - 1], [0, m - 1]
        - edges1 = n - 1
        = edges2 = m - 1 (u, v)
            u == target if # of edges on the path from u to v is <= k
                u is a target
        - k:
        return answer[i] == max possible num of nodes target to node i of edges1 if you have to connect one node from edges1 to edges2
        -

        (node, 0) = roots

        (0,0) = [(1,1), (2,2)]
        (1,0) = [(2,1)]
        """

        def build_adj(edges, adj):
            for s, e in edges:
                adj[s].append(e)
                adj[e].append(s)
            return adj

        def dfs(node, adj, steps, seen):
            if steps < 0:
                return 0
            if steps == 0:
                return 1
            count = 0
            for nei in adj[node]:
                if nei != node and nei not in seen:
                    count += dfs(nei, adj, steps - 1, seen)
            seen.add(node)
            return count

        def build_map(table, adj, k, rng):
            for i in range(rng[0], rng[1] + 1):
                if k > 0:
                    count = dfs(i, adj, k, set())
                elif k == 0:
                    count = 0
                else:
                    count = -1
                table[i] = count + 1
            return table

        adj1 = defaultdict(list)
        adj2 = defaultdict(list)
        table1 = defaultdict(int)
        table2 = defaultdict(int)
        adj1 = build_adj(edges1, adj1)
        adj2 = build_adj(edges2, adj2)
        min1 = min(adj1.keys())
        max1 = max(adj1.keys())
        min2 = min(adj2.keys())
        max2 = max(adj2.keys())
        table1 = build_map(table1, adj1, k, (min1, max1))
        table2 = build_map(table2, adj2, k - 1, (min2, max2))
        print(table2)
        top = -1
        top_i = 0
        for k, v in table2.items():
            if v > top:
                top = v
                top_i = k

        ans = []
        for i in range(min1, max1 + 1):
            n = table1[i]
            count = n + top
            ans.append(count)
        return ans


def test_solution():
    s = Solution()
    print(s.getLargestOutlier([6, -31, 50, -35, 41, 37, -42, 13]))


if __name__ == "__main__":
    test_solution()
