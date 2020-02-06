//Symmetric tree
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

public class Solution {
    /**
     * @param root: root of the given tree
     * @return: whether it is a mirror of itself 
     */
    //Input: {1,2,2,3,4,4,3} Output: true
    //check left, right and if current is null, 
    //then use recurison to check thcurrent node and val
    //O(n)
    public boolean isSymmetric(TreeNode root) {
        // Write your code here
    	return root == null || isSymmetricHelp(root.left, root.right);
    }

    private boolean isSymmetricHelp(TreeNode left, TreeNode right) {
    	if (left == null || right == null) {
    		return left == right;
    	}

    	if(left.val != right.val) {
    		return false;
    	}

    	return isSymmetricHelp(left.left, right.right) && return isSymmetrichelper(left.left, right.right) && isSymmetrichelper(left.right, right.left);
   		// remember to have both the left and the right
    }
}