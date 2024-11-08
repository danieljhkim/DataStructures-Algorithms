from typing import Optional
from dataclasses import dataclass

"""
- prefix tree
"""


@dataclass
class SimpleTrie:

    def __init__(self):
        self.root = {}

    def insert(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur:
                cur[c] = {}
            cur = cur[c]
        cur["*"] = ""

    def search(self, word: str) -> bool:
        cur = self.root
        for c in word:
            if c not in cur:
                return False
            cur = cur[c]
        return "*" in cur

    def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for c in prefix:
            if c not in cur:
                return False
            cur = cur[c]
        return True


@dataclass
class TrieNode:

    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.prefix_of = set()  # stores group id


@dataclass
class GroupTrie:
    def __init__(self):
        self.root = TrieNode()
        self.ids = set()

    def insert(self, word: str, id: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.is_end_of_word = True
        cur.prefix_of.add(id)

    def search(self, word: str, id: str) -> bool:
        cur = self.root
        for c in word:
            if c not in cur.children or id not in cur.prefix_of:
                return False
            cur = cur.children[c]
        return cur.is_end_of_word

    def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True

    def longest_common_prefix_within_group(self, ids: Optional[list]) -> int:
        """_summary_
        within a set of id's, returns the longest common prefix length
        Args:
            ids (list): group ids
        """
        cur = self.root
        if not ids:
            ids = self.ids
        return self._find_longest_common_prefix(cur, 0, ids)

    def _find_longest_common_prefix(self, cur: TrieNode, level: int, ids: list) -> int:
        top_outcome = level
        if cur.prefix_of.issuperset(ids):
            for v in cur.children.values():
                outcome = self._find_longest_common_prefix(v, level + 1, ids)
                top_outcome = max(top_outcome, outcome)
        return top_outcome
