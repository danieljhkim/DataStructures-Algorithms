

# Depth First Search

### Preorder

```python
def preorderTraversal(root):
    # used to create a copy/mirror of the tree
    # used to get prefix expressions of an expression tree
    # 1 2 4 5 3 6 7 
    if root:
        print(root.val)
        preorderTraversal(root.left)
        preorderTraversal(root.right)
```


##### Use Cases for Preorder Traversal
1. **Creating a Copy of the Tree**: Preorder traversal can be used to create a copy of the tree. By visiting the root node first, you can create a new node and then recursively copy the left and right subtrees.
2. **Prefix Expressions**: In expression trees (binary trees used to represent arithmetic expressions), preorder traversal is used to generate prefix expressions (also known as Polish notation). This is useful in certain types of compilers and calculators.
3. **Serialization/Deserialization**: Preorder traversal is often used in algorithms to serialize (convert to a string) and deserialize (reconstruct from a string) binary trees.
4. **Hierarchical Data**: Preorder traversal is useful for hierarchical data structures where you need to process the parent node before its children, such as in file systems or organizational charts.

##### Example
Given the binary tree:
```
         1
       /   \
      2     3
     / \   / \
    4   5 6   7
```
The preorder traversal would visit the nodes in the following order: [`1, 2, 4, 5, 3, 6, 7`]


---

# Breadth First Search

Breadth-First Search (BFS) is an algorithm for traversing or searching tree or graph data structures. It starts at the root node (or an arbitrary node in the case of a graph) and explores all of the neighbor nodes at the present depth prior to moving on to nodes at the next depth level.

#### concepts
- Queue: BFS uses a queue data structure to keep track of nodes to be explored. This ensures that nodes are explored in the order they are discovered.
- Level-by-Level Traversal: BFS explores nodes level by level. First, it visits all nodes at the current depth level before moving on to nodes at the next depth level.

```python
def BFS(root):
    if root is None:
        return

    queue = [root]
    while queue is:
        node = queue.pop(0)
        process(node)
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)
```