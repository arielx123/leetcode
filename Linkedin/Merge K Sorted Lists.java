Merge K Sorted Lists
/**
 * Definition for ListNode.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int val) {
 *         this.val = val;
 *         this.next = null;
 *     }
 * }
 */ 
//有序数组合并，使用两个指针向后移动，每次比较，小的一个数字取出来，并将指针后移一位。
//K个有序链表可以每次合并一个链表进结果链表中。合并k次
public class Solution {
    /**
     * @param lists: a list of ListNode
     * @return: The head of one sorted list.
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0 || lists ==null) {
            return null;
        }
        return mergeHelper(lists, 0, lists.length - 1);
    }

    private ListNode mergeHelper(ListNode[] lists, int start, int end) {
        if (start == end) {
            return lists[start];
        }

        int mid = start + (end - start) / 2;
        ListNode left = mergeHelper(lists, start, mid);
        ListNode right = mergeHelper(lists, mid + 1, end);
        return mergeTwoLists(left, right);
    }

    private ListNode mergeTwoLists(ListNode list1, ListNode list2) {
      
        if (list1 == null) return list2;
        if (list2 == null) return list1;
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        while(list1 != null && list2 != null){
            if(list1.val<list2.val){
                current.next = new ListNode(list1.val);
                list1 = list1.next;
                
            } else {
                current.next= new ListNode(list2.val);
                list2=list2.next;

            }
        current = current.next;
        }
        while (list1!=null){
            current.next= new ListNode(list1.val);
            current = current.next;
            list1 = list1.next;
            
        }
        
        while (list2!=null){
            current.next= new ListNode(list2.val);
            current = current.next;
            list2 = list2.next;
            
        }
        
        return dummy.next;
        
    }
}