from ast import List
from typing import Optional


class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:

    # 230. Kth Smallest Element in a BST
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        values = []

        def inorder(root):
            if not root:
                return
            inorder(root.left)
            values.append(root.val)
            inorder(root.right)

        inorder(root)
        return values[k - 1]

    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        ans = None
        found = False

        def inorder(root, count):
            nonlocal ans, found
            if not root or found:
                return count
            count = inorder(root.left, count)
            if found:
                return count
            if count == k:
                ans = root.val
                found = True
                return count
            count += 1
            count = inorder(root.right, count)
            return count

        inorder(root, 1)
        return ans

    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        while root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val
            root = root.right

    # 98. Validate Binary Search Tree
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        stack = []
        prev_val = float("-inf")
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if root.val <= prev_val:
                return False
            prev_val = root.val
            root = root.right
        return True

    # 114. Flatten Binary Tree to Linked List
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        pre-order traversal
        """
        queue = []

        def pre_order(root):
            if not root:
                return
            queue.append(root)
            pre_order(root.left)
            pre_order(root.right)

        pre_order(root)
        if queue:
            curr = queue.pop(0)
        while queue:
            curr.left = None
            curr.right = queue.pop(0)
            curr = curr.right

    # 102. Binary Tree Level Order Traversal
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        def bfs(root):
            if not root:
                return
            level = 1
            queue = [(root, level)]
            levels = []
            while queue:
                node, level = queue.pop(0)
                if len(levels) < level:
                    n = level - len(levels)
                    for _ in range(n):
                        levels.append([])
                levels[level - 1].append(node.val)
                if node.left:
                    queue.append((node.left, level + 1))
                if node.right:
                    queue.append((node.right, level + 1))
            return levels

        return bfs(root)

    """
            0
        1       2
      2  3    4  5
    
    """

    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        queu = [(root, 0)]
        levels = {}
        while queu:
            curr, level = queu.pop(0)
            if level not in levels:
                levels[level] = []
            levels[level].append(curr)
            if curr.right:
                queu.append((curr.right, level + 1))
            if curr.left:
                queu.append((curr.left, level + 1))

        for level in levels:
            if level % 2 == 1:
                nodes = levels[level]
                vals = [node.val for node in nodes]
                j = 0
                for i in range(len(vals) - 1, -1, -1):
                    nodes[i].val = vals[j]
                    j += 1
        return root

    # 199. Binary Tree Right Side View
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        levels = []
        ans = []

        def bfs(root):
            if not root:
                return
            queue = [(root, 0)]
            while queue:
                node, level = queue.pop(0)
                if len(levels) < level + 1:
                    levels.append([])
                levels[level].append(node.val)
                if (len(queue) > 0 and queue[0][1] > level) or not queue:
                    ans.append(node.val)
                if node.left:
                    queue.append((node.left, level + 1))
                if node.right:
                    queue.append((node.right, level + 1))

        bfs(root)
        return ans

    # 236. Lowest Common Ancestor of a Binary Tree
    def lowestCommonAncestor(
        self, root: "TreeNode", p: "TreeNode", q: "TreeNode"
    ) -> "TreeNode":
        queue = [(root, [root])]
        first = None
        second = None
        while queue:
            node, anc = queue.pop(0)
            anc = anc.copy()
            anc.append(node)
            if node.val == p.val or node.val == q.val:
                if not first:
                    first = (node, anc)
                else:
                    second = (node, anc)
                    break
            if node.left:
                queue.append((node.left, anc))
            if node.right:
                queue.append((node.right, anc))
        for i in range(len(first[1]) - 1, -1, -1):
            for j in range(len(second[1]) - 1, -1, -1):
                if first[1][i].val == second[1][j].val:
                    return first[1][i]

    # 1367. Linked List in Binary Tree
    def isSubPath(self, head: Optional[ListNode], root: Optional[TreeNode]) -> bool:
        curr_node = head
        ans = False

        def inorder(root, curr_node, prev):
            nonlocal ans
            if not curr_node or ans:
                ans = True
                return
            if not root:
                curr_node = head if not prev else prev
                return
            if curr_node.val != root.val:
                curr_node = head if not prev else prev
            else:
                prev = curr_node
                curr_node = curr_node.next
            inorder(root.left, curr_node, prev)
            inorder(root.right, curr_node, prev)

        inorder(root, curr_node, curr_node)
        return ans or False

    # 450. Delete Node in a BST
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:

        def find_max_node(node):
            cur = node
            while cur.right:
                cur = cur.right
            return cur

        if not root:
            return root
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            if not root.left:
                return root.right
            elif not root.right:
                return root.left
            max_left_node = find_max_node(root.left)
            root.val = max_left_node.val
            root.left = self.deleteNode(root.left, root.val)
        return root

    def kthLargestPerfectSubtree(self, root: Optional[TreeNode], k: int) -> int:
        freq = []

        def dfs(node):
            if not node:
                return 0
            l = dfs(node.left)
            r = dfs(node.right)
            if l == r and l != -1:
                freq.append(l + r + 1)
                return l + r + 1
            return -1

        dfs(root)
        freq.sort(reverse=True)
        return -1 if len(freq) < k else freq[k - 1]

    # 1650
    def lowestCommonAncestor(self, p: "Node", q: "Node") -> "Node":
        def find_depth(node):
            level = 0
            while node:
                node = node.parent
                level += 1
            return level

        p_level = find_depth(p)
        q_level = find_depth(q)

        for _ in range(p_level - q_level):
            p = p.parent
        for _ in range(q_level - p_level):
            q = q.parent

        while p != q:
            p = p.parent
            q = q.parent
        return p
