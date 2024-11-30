from typing import Optional, List
from collections import Counter, deque, defaultdict, OrderedDict


class Solutions:

    def alienOrder(self, words: List[str]) -> str:
        """
        - words = ["wrt","wrf","er","ett","rftt"]
        """
        N = len(words)
        indegree = {}
        adj = defaultdict(set)
        for word in words:
            for w in word:
                indegree[w] = 0
        i = 0
        for i in range(N - 1):
            first = words[i]
            second = words[i + 1]
            for w1, w2 in zip(first, second):
                if w1 != w2:
                    if w2 not in adj[w1]:
                        indegree[w2] += 1
                        adj[w1].add(w2)
                        break
            else:  # no break
                if len(first) > len(second):
                    return ""

        queue = deque()
        ans = []
        for k, n in indegree.items():
            if n == 0:
                queue.append(k)
        while queue:
            w = queue.popleft()
            if indegree[w] == 0:
                ans.append(w)
            for neigh in adj[w]:
                indegree[neigh] -= 1
                if indegree[neigh] == 0:
                    queue.append(neigh)

        for n in indegree.values():
            if n > 0:
                return ""
        return "".join(ans)
