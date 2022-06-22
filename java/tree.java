
public class TreeNode {
	int val;
	TreeNode left;
	TreeNode right;
	TreeNode() {}
	TreeNode(int val) { this.val = val; }
	TreeNode(int val, TreeNode left, TreeNode right) {
		this.val = val;
		this.left = left;
		this.right = right;
	}
 }

class Solutions {

	public int deepestLeavesSum(TreeNode root) {
		int sum = 0;
		int maxDepth = 0;
		dfs(root, 1);
		return sum;
	}

	private void dfs(TreeNode root, int depth) {
		if(root == null) return;
		if(depth > maxDepth) {
			maxDepth = depth;
			sum = root.val;
		} else if(depth == maxDepth) {
			sum += root.val;
		}
		dfs(root.left, depth+1);
		dfs(root.right, depth+1);
	}

}