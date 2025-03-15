import heapq
from typing import *


class Solution:

    # 3485. Longest Common Prefix of K Strings After Removal
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        trie = {}
        heap = []
        ans = []

        for i, word in enumerate(words):
            cur = trie
            for j, w in enumerate(word):
                if w not in cur:
                    cur[w] = {"#": set(), "$": j + 1}
                cur = cur[w]
                cur["#"].add(i)

        def find_prefix(node):
            if not node or len(node.get("#", [])) < k:
                return -1
            heap.append((-node["$"], node["#"]))
            for key, cur in node.items():
                if key.isalpha():
                    if len(cur["#"]) >= k:
                        find_prefix(cur)

        for v in trie.values():
            find_prefix(v)

        heapq.heapify(heap)

        for i in range(len(words)):
            tmp = []
            while heap and (i in heap[0][1] and len(heap[0][1]) == k):
                out = heapq.heappop(heap)
                tmp.append(out)
            if heap:
                ans.append(-heap[0][0])
            else:
                ans.append(0)
            for v in tmp:
                heapq.heappush(heap, v)

        return ans
