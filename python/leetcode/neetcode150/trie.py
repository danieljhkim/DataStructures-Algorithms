# 211. Design Add and Search Words Data Structure


class WordDictionary:

    def __init__(self):
        self.root = {}

    def addWord(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur:
                cur[c] = {}
            cur = cur[c]
        cur["*"] = True

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return "*" in node
            if word[idx] == ".":
                for ch in node.values():
                    if dfs(ch, idx + 1):
                        return True
            else:
                if word[idx] not in node:
                    return False
                else:
                    return dfs(node[word[idx]], idx + 1)

        cur = self.root
        return dfs(cur, 0)
