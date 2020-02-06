//Second Minimum Node In a Binary Tree
/**
 * Definition of TreeNode:
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left, right;
 *     public TreeNode(int val) {
 *         this.val = val;
 *         this.left = this.right = null;
 *     }
 * }
 */
//671

public class Solution {
    /**
     * @param root: the root,非负值二叉树，
     * @return: the second minimum value in the set made of all the nodes' value in the whole tree
     */
    long min = Long.Max_VALUE;//Capitalize
    long minSecond = Long.Max_VALUE;
    
    public int findSecondMinimumValue(TreeNode root) {
        // Write your code here
    	dfs(root);
    	return min2 == Long.Max_VALUE ? -1 : (int) min2;
    }

    public void dfs(TreeNode root) {
    	if (root == null) {
    		return;
    	}

    	if (min > root.val) {
    		min = root.val;
    	}

    	if (min < root.val && root.val < minSecond) {
    		minSecond = root.val;
    	}

    	dfs(root.left);
    	dfs(root.right);
    }
}