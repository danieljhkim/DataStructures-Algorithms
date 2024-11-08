from __future__ import annotations


class SuffixTreeNode:

    def __init__(
        self,
        children: dict | None = None,
        is_end: bool = False,
        indexes: list | None = None,
    ):
        self.children = children or {}
        self.is_end = is_end
        self.indexes = indexes or []


class SuffixTree:

    def __init__(self, text):
        self.root = SuffixTreeNode()
        self.text = text
        self.build_suffix_tree()

    def insert_suffix(self, suffix, index):
        node = self.root
        for char in suffix:
            # If the character isn't in the current node, create a new child node
            if char not in node.children:
                node.children[char] = SuffixTreeNode()
            node = node.children[char]
            node.indexes.append(index)  # Store the starting index of this suffix

    def build_suffix_tree(self):
        # Insert all suffixes into the suffix tree
        for i in range(len(self.text)):
            suffix = self.text[i:]
            self.insert_suffix(suffix, i)

    def search(self, pattern):
        node = self.root
        for char in pattern:
            # If character isn't in the current node's children, pattern not found
            if char not in node.children:
                return []
            node = node.children[char]
        return node.indexes


text = "banana"
suffix_tree = SuffixTree(text)

# Searching for patterns in the suffix tree
print("Suffix Tree created for:", text)
print("Indexes of 'ana':", suffix_tree.search("a"))
