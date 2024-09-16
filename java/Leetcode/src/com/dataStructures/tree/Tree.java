package leetcode.src.com.dataStructures.tree;


public class Tree {

	public int deepestLeavesSum(TreeNode root) {
		int sum = 0;
		int maxDepth = 0;
		dfs(root, 1);
		return sum;
	}

	private void dfs(TreeNode root, int depth) {

	}

}
