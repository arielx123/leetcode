//Plus One Linked List
/**
 * Definition for ListNode
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
/*
*输入: 1 -> 2 -> 3 -> null
输出: 1 -> 2 -> 4 -> null
解释:
123 + 1 = 124
*/
/*
两个pointer，slow是最后一个不是9的数字，fast不断向后travle，直到最后一个node
如果slow是9，意味着list是这种形式 9->9->9->null
如果不是9，则slow = slow + 1， 并且更新slow之后的所有node为0
O(N) time
O(1) space
*/

public class Solution {
    /**
     * @param head: the first Node
     * @return: the answer after plus one
     */
    public ListNode plusOne(ListNode head) {
        // Write your code here
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode left = dummy;
        ListNode right = head;

        while (right != null) {
        	if (right.val != 9) {
        		left = right;
        	}

        	right = right.next;
        }
        // currently, right is the most right
        // left is the last one that is not 9
        right = left.next;// since right is 9; 
        while (right != null) {
        	right.val = 0;
        	right = right.next; // make all of them to 0
        }

        left.val++;

        return left != dummy ? head : left;

    }
}