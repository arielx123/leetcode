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
*/

public class Solution {
    /**
     * @param l1: ListNode l1 is the head of the sorted linked list
     * @param l2: ListNode l2 is the head of the sorted linked list
     * @return: ListNode head of linked list (ascending)
     * // time: O(min(m,n)),space: O(1)
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        // write your code here
        /* add dummy node, 如果我们需要返回头部，一般情况可以创建一个虚拟节点,
        * dummy.next = head, 就算head 变化了以后,只要返回dummynode.next就能轻松得到新头部
        */ 
        ListNode dummy = new ListNode(0);
        ListNode head = dummy; // add a reference 

        while (l1 != null && l2 != null) {
        	if (l1.val < l2.val){
        		head.next = l1;
        		l1 = l1.next;
        	} else {
        		head.next = l2;
        		l2 = l2.next;
        	}
        	head = head.next;

        }

        //check if it is null 
        if (l1 != null) {
        	head.next = l1;
        } else {
        	head.next = l2;
        }

        return dummy.next;


    }
}