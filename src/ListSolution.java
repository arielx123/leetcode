package list;

import java.util.*;


/**
 * Definition for singly-linked list.
 */
class ListNode { // 去掉public
	int val;
	ListNode next;

	ListNode(int x) {
		val = x;
		next = null;
	}
}

public class ListSolution {

	/**206. Reverse Linked List 
	 * @param head
	 * @return
	 * !!!!!pre一定要设为null，不能从head开始。
		否则当pre在head那时，2nd.next->pre, 但pre.next->2nd 依然没变，所以会循环！
		
	 */
	public ListNode reverseList(ListNode head) {
		ListNode pre = null;		//记得是null
		
		while (head != null) {
			ListNode next = head.next;
			head.next = pre;
			pre = next;	
			head = next;
		}
		return pre;
	}
	
	//用recursion做
	public ListNode reverseListRec(ListNode head) {
		// 整个list head为空，或者只有1个  || 最后退出是后面没node了
        if (head == null || head.next == null)
            return head;
        
        ListNode newHead = reverseListRec(head.next);     //把后面的reverse
        head.next.next = head;		// 1->2->3, 变成1->2<-3, 那么1的next还是2，需要变
        head.next = null;
        
        return newHead ;
    }
	

	/** 92. Reverse Linked List II
	 * Reverse a linked list from position m to n
	 * 
	 * 记得先用dummy node, 最后也是返回它。
	 * 找oldPre时，用pre往前走，而不是cur
	 * 
	 * 需要记录oldPre和oldCur, 找到以后reverse，最后连起来
	 * oldPre.next = pre; // 老的连到旧的
	 * oldCur.next = cur;
	 */
	public ListNode reverseBetween(ListNode head, int m, int n) {
		if (m == n) {
			return head;
		}
		ListNode dummy = new ListNode(0); // dummy不动，指向head，即使删了head也没所谓
		dummy.next = head;
		ListNode pre = dummy; // cur从dummy开始

		// find the pre point of m
		for (int i = 1; i < m; i++) { 
			pre = pre.next;
		}
		ListNode oldPre = pre; // note down original pre & cur
		ListNode cur = pre.next;
		ListNode oldCur = cur;			//也要记得old cur，到时要连后面

		// start reverse
		for (int i = m; i <= n; i++) { 
			ListNode next = cur.next; // 形成一个loop
			cur.next = pre; // 第一次会提前指回去，但之后能改变指针
			pre = cur;
			cur = next;
		}
		oldPre.next = pre; // 老的连到旧的
		oldCur.next = cur;
		
		return dummy.next; // 记得是dummy.next ，而不是head
	}
	
	
	
	/** 234. Palindrome Linked List
	 * @param head
	 * @return
	 * 先找中点mid，然后reverse后半段，再跟前半段比较
	 */
	public boolean isPalindrome(ListNode head) {
  //      if (head == null)   return true;
        ListNode mid = head;
        ListNode fast = head;		//也可以=head.next
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            mid = mid.next;
        }
        if (fast != null) {     // odd, 右边比较短shorter
            mid = mid.next;
        }
        mid = reverse(mid);
        fast = head;
        
        while (mid != null) {       //因为右边短，所以以right长度为准
            if (fast.val != mid.val)
                return false;
            fast = fast.next;
            mid = mid.next;
        }
        return true;
    }
	
	

	/** 148. Sort List - merge sort方法
	 * Sort a linked list in O(n log n) time using constant space complexity.
	 * 看到O(nlog n)想到要用quick或merge sort
	 * 但是quick的话链表无法随机访问任意节点来挑选pivot值比较，如果挑head会比得不均匀 所以用merge sort. 
	 * 怎样都是O(nlogn) 
	 * @param head
	 * @return dummy.next
	 * 记得找到mid以后，先right，然后断开mid, mid.next=null，再sort左边
	 */
	public ListNode sortList(ListNode head) {
		if (head == null || head.next == null) {
			return head;		//有rec所以要有break条件
		}

		ListNode mid = findMiddle(head);
		ListNode secondHead = mid.next;
		mid.next = null;					// 记得断开 让left半段的尾巴为null，否则一直指到right段会死循环

		ListNode left = sortList(head);
		ListNode right = sortList(secondHead);

		return merge(left, right);
	}

	// 找中点，快慢指针. 这个mid 偏左.. 这样之后 right半段可以直接 mid.next表示
	private ListNode findMiddle(ListNode head) {
		ListNode slow = head;
		ListNode fast = head.next;		//是head.next
		
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
		}
		return slow;
	}
		
	
	/**
	 * 21. Merge Two Sorted Lists - easy
	 */
	public ListNode merge(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode head = dummy;
        
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                head.next = l1;
                l1 = l1.next;
            } else {
                head.next = l2;
                l2 = l2.next;
            }
            head = head.next;
        }
        if (l1 != null)		head.next = l1;
        else				head.next = l2;
        
        return dummy.next;
    }


    // Recursion 写法
	public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		// exception check
		if(l1 == null){
			return l2;
		}
		if(l2 == null){
			return l1;
		}

		if(l1.val < l2.val){
			l1.next = mergeTwoLists(l1.next,l2);
			return l1;
		}else{
			l2.next = mergeTwoLists(l2.next,l1);
			return l2;
		}
	}


	// quick sort
	public ListNode quickSortList(ListNode head) {
		if (head == null || head.next == null)
			return head;

		ListNode leftHead = new ListNode(0), left = leftHead;
		ListNode rightHead = new ListNode(0), right = rightHead;
		ListNode midHead = new ListNode(0), mid = midHead;      // middle that equals to pivot

		int pivot = head.val;

		// partition
		while (head != null) {
			// 需要有mid 来equals pivot隔断，否则如果h.v <= p都放left的话，会死循环，分不出
			if (head.val == pivot) {
				mid.next = head;
				mid = mid.next;
			} else if (head.val < pivot) {
				left.next = head;
				left = left.next;
			} else {
				right.next = head;
				right = right.next;
			}

			head = head.next;
		}

		// 断开
		left.next = null;
		mid.next = null;
		right.next = null;

		left = quickSortList(leftHead.next);     // sorted left head
		right = quickSortList(rightHead.next);   // sorted right head

		return concat(left, midHead.next, mid, right);
	}

	private ListNode concat(ListNode left, ListNode midHead, ListNode midTail, ListNode right) {
		ListNode leftTail = getTail(left);
		midTail.next = right;

		// 如果 head是最小的，left一直不会动，那就是 left.next = null，所以记得判断！！！
		if (leftTail != null) {
			leftTail.next = midHead;
			return left;
		} else {
			return midHead;
		}
	}

	private ListNode getTail(ListNode head){
		while(head != null && head.next != null) {
			head = head.next;
		}
		return head;
	}



	/** 86. Partition List  --- Quick Sort的一个步骤（partition）
	 * 给你x 和一串list，< x的放左边，>= right放右边
	 * Given 1->4->3->2->5->2 and x = 3,  return 1->2->2->4->3->5.
	 * @param head
	 * @param x
	 * @return
	 * 用dummynode。
	 *
	 * !!!!!!!! 记得也需要rightDummy来建新的big part，否则如果只用原先的会TLE 死循环，因为没有断掉 big -> small 的link
	 */
	public ListNode partition(ListNode head, int x) {
		ListNode leftDummy = new ListNode(0);
		ListNode rightDummy = new ListNode(0);
		ListNode left = leftDummy;      //left跟dummy一样指向新建的node(0), 它们都只是地址
		ListNode right = rightDummy;

		while (head != null) {
			if (head.val < x) {
				left.next = head;
				left = head;
			} else {
				right.next = head;
				right = right.next; // or = head
			}
			head = head.next;
		}
		right.next = null;
		left.next = rightDummy.next;

		return leftDummy.next;
	}



	/** 23. Merge k Sorted Lists 
	 * Merge k sorted linked lists and return it as one sorted list.
	 *
	 * complexity: O(nlogk), n means total numbers, heap's size is k
	 * PriorityQueue的enque,deque是O(log n)
	 * 先把k个list的head放到heap里，塞进去时就自动排序，这是heap的头是min. 把heap的头取出来让ListNode指向它。
	 * 之后把head后面的node塞进heap里
	 * @param lists
	 * @return
	 */
	// lists store all of the heads, cause the data type is ListNode, k个头
	public ListNode mergeKLists(ArrayList<ListNode> lists) {
		if (lists == null || lists.size() == 0) {
			return null;
		}
		// 建立heap，size为K。调用Comparator，每次往heap里塞东西都会塞到排好序的位置上
		Queue<ListNode> heap = new PriorityQueue<ListNode>(lists.size(), (a, b) -> a.val - b.val);
		for (int i = 0; i < lists.size(); i++) { // lists.size()=k
			if (lists.get(i) != null) { 		// 有可能某个头为null
				heap.add(lists.get(i));
			}
		}

		ListNode dummy = new ListNode(0);
		ListNode cur = dummy;
		while (!heap.isEmpty()) {
			ListNode node = heap.poll(); 
			cur.next = node;
			cur = cur.next; 					// 记得move to next !!!!!!

			if (node.next != null) { 		// !!!! 可能后面为null. 记得判断
				heap.add(node.next); 		// add next node(new head) into heap
			}
		}
		return dummy.next;
	}

	
	/** Merge K sorted lists
	 * Divide & Conquerer 分治   Merge Sort -  recursion
	 * 将K个list分为(0, k/2) 和 (k/2 + 1, n-1) 的区间
	 * O(nlogK)
	 * @param lists
	 * @return
	 */
	public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        
        return mergeHelper(lists, 0, lists.length - 1);
    }
    
    public ListNode mergeHelper(ListNode[] lists, int start, int end) {
        if (start == end) {
            return lists[start];	//list[s] !!
        }
        
        int mid = start + (end - start) / 2;
        ListNode left = mergeHelper(lists, start, mid);
        ListNode right = mergeHelper(lists, mid + 1, end);
        
        return merge(left, right);
    }
    
    
    /** Merge K sorted List - iterative
     * 两两合并。因为合并后要再两两合并，要用到2层循环。且要改变参数成arrayList. 而不能数组
     * @param lists
     * @return
     */
    public ListNode mergeKLists(List<ListNode> lists) {
        if (lists == null || lists.size() == 0) {
            return null;
        }
        
        // merge 2-by-2
        while (lists.size() > 1) {
            List<ListNode> newList = new ArrayList<>();
            for (int i = 0; i < lists.size() - 1; i += 2) {
                ListNode merged = merge(lists.get(i), lists.get(i + 1));
                newList.add(merged);
            }

            if (lists.size() % 2 == 1) {    // if odd, add last one 因为前面会忽略掉最后的奇数
                newList.add(lists.get(lists.size()-1));
            }
            
            lists = newList;     // so as to iterative the combination 
        }
        return lists.get(0);
    }


	/** Insertion Sort List
	 * Sort a linked list using insertion sort.
	 *
	 * @param head
	 * @return 每次都插入head.. 慢！！
	 */
	public ListNode insertionSortList(ListNode head) {
		ListNode dummy = new ListNode(0);		// !!! 注意不能 dummy.next = head. 否则死循环
		ListNode node;

		while (head != null) {
			node = dummy;
			while (node.next != null && node.next.val < head.val) {
				node = node.next;
			}
			ListNode next = head.next;
			head.next = node.next;
			node.next = head;
			head = next;
		}
		return dummy.next;
	}


	/**
	 * 用preInsert 和 toInsert来记录下，这样稍微快一点，而不用每次都从头来过
	 * @param head
	 * @return
	 */
	public ListNode insertionSortListBetter(ListNode head) {
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode preInsert = dummy;
		ListNode toInsert;

		while (head != null && head.next != null) {
			if (head.val <= head.next.val) {
				head = head.next;
			} else {
				toInsert = head.next;			// 要insert的是next.. 而不是head自己

				// find preInsert to insert the node
				if (preInsert.next.val > toInsert.val) {	// 更快点，也可以不用这个判断，每次都=dummy, 但这样就稍微慢点
					preInsert = dummy;
				}
				while (preInsert.next.val < toInsert.val) {
					preInsert = preInsert.next;
				}
				head.next = toInsert.next;
				toInsert.next = preInsert.next;
				preInsert.next = toInsert;
			}
		}
		return dummy.next;
	}


	/** Insertion Sort List
	 * 分三种情况，这样比较复杂，但是比上面快很多
	 * @param head
	 * @return
	 */
	public ListNode insertionSortList2(ListNode head) {
		if(head == null || head.next == null){
			return head;
		}

		ListNode preNode = head;			// node before insertNode
		ListNode insertNode = head.next;	// node need to be inserted

		while(insertNode != null){
			//store next insert node
			ListNode nextInsert = insertNode.next;
			//insert before head
			if(insertNode.val <= head.val){
				preNode.next = nextInsert;
				insertNode.next = head;
				head = insertNode;
			}
			else if(insertNode.val >= preNode.val){    //insert after tail
				preNode = preNode.next;
			}
			else{                                      //insert between head and tail
				ListNode compareNode = head;
				//start from the node after head, find insert position
				while(compareNode.next.val < insertNode.val) {
					compareNode = compareNode.next;
				}
				//insert
				preNode.next = nextInsert;
				insertNode.next = compareNode.next;
				compareNode.next = insertNode;
			}
			//get next insert node
			insertNode = nextInsert;
		}
		return head;
	}
    

	
	/** 143. Reorder List 
	 * Given a singly linked list L: L0→L1→…→Ln-1→Ln, reorder it
	 * to: L0→Ln→L1→Ln-1→L2→Ln-2→… 
	 * For example,Given {1,2,3,4}, reorder it to {1,4,2,3}. 
	 * 先分成2段找mid, 把后半段mid~end反转reverse, 再merge到前半段
	 */
	public void reorderList(ListNode head) {
		if (head == null || head.next == null) {
			return;
		}
		ListNode mid = findMiddle(head);
		ListNode revHead = reverse(mid.next); // reverse right half
		mid.next = null; // !!!!!!记得断开。所以find mid要在mid前一位，前半段才能断开
		merge2(head, revHead);
	}

	private ListNode reverse(ListNode head) {
		ListNode pre = null;
		ListNode next = null;
		while (head != null) {
			next = head.next;
			head.next = pre;
			pre = head;
			head = next;
		}
		return pre;
	}

	private void merge2(ListNode head1, ListNode head2) {
		ListNode newTail = new ListNode(0);
	//	int index = 0; // count the index, start from 0 算奇偶
		boolean odd = true;
		while (head1 != null && head2 != null) {
			if (odd) {
				newTail.next = head1;
				head1 = head1.next; // ！！！！记得h1也往后
			} else {
				newTail.next = head2; // head2 is odd
				head2 = head2.next;
			}
			newTail = newTail.next;
			odd = !odd;
		}

		if (head1 != null) { // head2 is null, only head1 left
			newTail.next = head1;
		} else {
			newTail.next = head2;
		}
	}
	

	/** 19. Remove Nth Node From End of List
	 * Given linked list: 1->2->3->4->5, and n = 2.
		After removing the second node from the end, the linked list becomes 1->2->3->5.
	 * @param head
	 * @param n
	 * @return
	 */
	public ListNode removeNthFromEnd(ListNode head, int n) {
		if (head == null || (head.next == null && n == 1))
			return null;

		ListNode tmp = new ListNode(0);
		tmp.next = head;
		ListNode node = tmp;

		//先走N步
		for (int i = 0; i < n; i++) {	
			head = head.next;
		}

		// head继续走剩下的len-n步，node也一起走len-n步
		while (head != null) { // otherwise, use a preDummy(next=head).preDummy.next!=null
			head = head.next;
			node = node.next;
		}
		node.next = node.next.next;
		return tmp.next;
	}

	
	
	/** 141. Linked List Cycle
	 * Linked List Cycle I Given a linked list, determine if it has a cycle in
	 * it. fast走2步，slow走1步。有环的话肯定能相遇
	 * @param head
	 * @return Boolean
	 */
	public boolean hasCycle(ListNode head) {
		ListNode fast = head;
		ListNode slow = head;

		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
			if (slow == fast) {
				return true;
			}
		}
		return false;
	}


	/** 142. Linked List Cycle II 
	 * Given a linked list, return the node where the cycle
	 * begins. If there is no cycle, return null. 
	 * fast走2步，slow走1步。有环的话相遇,然后slow=head, slow和fast都走一步直到相遇 
	 * @param head
	 * @return Boolean
	 *
	 * 也有最naive的就是用hashset.. 第一个contains的那个就是
	 */
	public ListNode detectCycleII(ListNode head) {
		ListNode fast = head;
		ListNode slow = head;

		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
			if (fast == slow) {
				break;
			}
		}

		if (fast == null || fast.next == null) { 	//也可以用过Boolean hasCycle来表示
			return null;
		}

		// 找到交接点后，slow和fast一步步走，直到相遇
		slow = head;
		while (fast != slow) {
			fast = fast.next;
			slow = slow.next;
		}
		return slow;
	}


	/** 369. Plus One Linked List
     * 1->2->3 加1 变成 1->2->4  return head
     * @param head
     * @return
     * 找到第一个非9的node. 然后n.val++, 后面全变0.
     * n在dummy，也cover了9999..+1 后有一个carry的情况
     */
    public ListNode plusOne(ListNode head) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode n = dummy;
        ListNode last = dummy;  //last non-9
        
        while (n.next != null) {
            n = n.next;
            if (n.val != 9) {
                last = n;
            }
        }
        
        last.val++;
        last = last.next;
        while (last != null) {
            last.val = 0;
            last = last.next;
        }
        
        if (dummy.val == 0) {
            return dummy.next;
        }
        
        return dummy;   //last at dummy, and it ++ after 1st while
    }
    
    // 复杂。 先reverse，再+1，再reverse
    public ListNode plusOneWorse(ListNode head) {
        ListNode newHead = reverse(head);
        ListNode tmp = newHead;
        int carry = 1;
        while (tmp != null) {
            tmp.val += carry;
            if (tmp.val < 10) {
                carry = 0;
                break;
            }
            tmp.val %= 10;
            tmp = tmp.next;
        }
       
        head = reverse(newHead);
        if (carry == 1) {
            newHead = new ListNode(1);
            newHead.next = head;
            return newHead;
        }
        return head;
    }
    
	/**Rotate List 
	 * Given a list, rotate the list to the right by k places, where
	 * k is non-negative. Given 1->2->3->4->5->NULL and k = 2, return
	 * 4->5->1->2->3->NULL.
	 *
	 * 需要有n > list.size 要循环。 先找倒数第k个：fast先走n, 然后和slow一起走
	 * fast走到end就能point to head slow就走到倒数第k前一个，要记录newHead和newEnd
	 */
	public ListNode rotateRight(ListNode head, int n) {
		if (n == 0 || head == null || head.next == null) {
			return head;
		}

		ListNode fast = head, slow = head, index = head;
		for (int i = 0; i < n; i++) {
			fast = fast.next; // fast move first
			if (fast == null) { // need to loop
				fast = head;
			}
		}

		// 寻找倒数第k点
		while (fast.next != null) {
			fast = fast.next;
			slow = slow.next; // find prev of 倒数第k点
		}
		// 形成一个loop
		fast.next = head; // point end to head
		head = slow.next;
		slow.next = null;
		return head;
	}

	// mine, naive
	public ListNode rotateRightMine(ListNode head, int k) {
        if (head == null || k == 0 ) {
            return head;
        }
        ListNode tmp = head;
        int len = 1;
        while(tmp.next != null) {
            tmp = tmp.next;
            len++;
        }
        tmp.next = head;    //connect tail->head
        
        // find node before k
        ListNode tail = head;
        for (int i = 1; i < (len - k % len); i++) {
            tail = tail.next;
        }
        
        ListNode newHead = tail.next;
        tail.next = null;
        return newHead;
    }
	
	
	 /** 138. Copy List with Random Pointer 
	  * return deep copy.. 用HashMap。
	  * 因为刚建node时，next和random所指的node还没建起来，所以用map存old & new，分两次loop
	  * 
	  * 后面有个 走一次 即可的
	 * @param head
	 * @return
	 */
	public RandomListNode copyRandomListHashMap(RandomListNode head) {
		 if (head == null) {
	            return null;
	        }
	        //因为可以用map取，所以不需要dummy node!!!!
	        // copy all nodes with val
	        RandomListNode old = head;
	        Map<RandomListNode, RandomListNode> map = new HashMap<>();
	        while (old != null) {
	            map.put(old, new RandomListNode(old.label));
	            old = old.next;
	        }
	        
	        // copy nodes with next & random
	        old = head;
	        RandomListNode cur;
	        while (old != null) {
	            cur = map.get(old);
	            cur.next = map.get(old.next);
	            cur.random = map.get(old.random);
	            old = old.next;
	        }
	        
	        return map.get(head);
	 }
	
	// recursion
	private RandomListNode copy(RandomListNode node, Map<RandomListNode, RandomListNode> map) {
        if (node == null)
           return null;
       
       if (map.containsKey(node))
           return map.get(node);
       
       RandomListNode clone = new RandomListNode(node.label);
       map.put(node, clone);       // put this before recursion  记得放前面，否则放最后会overflow
       
       clone.next = copy(node.next, map);
       clone.random = copy(node.random, map);
       
       return clone;
   }
		
	/**
	 * 138. Copy List with Random Pointer 
	  * return deep copy.. 用HashMap。
	  * 
	  * 这个跟上面的recursion很像，是它的iteration版本，而且只iterate一次（而非两次）
	  * 
	 * @param head
	 * @return
	 */
	public RandomListNode copyRandomList2(RandomListNode head) {
        if (head == null)
            return null;
        
        Map<RandomListNode, RandomListNode> map = new HashMap<>();
        RandomListNode node = head;
        
        RandomListNode clone = new RandomListNode(node.label);
        map.put(node, clone);
        
        while (node != null) {
            
            clone.next = getClonedNode(node.next, map);
            clone.random = getClonedNode(node.random, map);
            
            node = node.next;
            clone = clone.next;     // 记得clone要往后走，因为已经有了next
        }
        
        return map.get(head);
    }
    
    private RandomListNode getClonedNode(RandomListNode node, Map<RandomListNode, RandomListNode> map) {
        if (node != null) {
            if (!map.containsKey(node)) {
                map.put(node, new RandomListNode(node.label));
            }
            return map.get(node);
        }
        return null;
    }

	/**Copy List with Random Pointer - 不用map的方法
	 * 
	 * 开始数组是1->2->3->4 变成 1->1`->2->2`->3->3`->4->4`
	 * 1. 先通过next来赋值链表：第一遍扫描顺序复制next指针
	 * ，把ABCD的next分别指向A’B’C’D’，将A’的next指针指向B，B’的next指针指向C
	 * 2. 然后复制random指针：A’->random = A->random->next(后面这next是指向n')
	 * 3. 最后分开原链表和复制的list：A->next = A’->next(A'->B); A’->next =
	 * A’->next->next(A'->B');
	 */
	public RandomListNode copyRandomList(RandomListNode head) {
		if (head == null) {
			return null;
		}
		
		RandomListNode node = head;
		
		// 1st round, copy node, make orig -> copy A-A'-B-B'
        while (node != null) {
        	RandomListNode clone = new RandomListNode(node.label);
            clone.next = node.next;
            node.next = clone;
            
            node = node.next.next;        
        }
        
        // 2nd round, copy random
        node = head;
        while (node != null) {
            if (node.random != null) {				// 记得检查null，否则下面那句话是null会出错
                node.next.random = node.random.next;
            }
            node = node.next.next;
        }
        
        // 3rd round, extract the cloned list
        RandomListNode newHead = head.next;
        RandomListNode newNode = newHead;
        node = head;
        while (node != null) {
            node.next = newNode.next;
            if (newNode.next != null) {			//!!!记得check null 有可能 B是空
                newNode.next = newNode.next.next;
            }
            // move
            node = node.next;
            newNode = newNode.next;
        }
        
        return newHead;
	}
	
	
	class RandomListNode {
		int label;
		RandomListNode next, random;

		RandomListNode(int x) {
			this.label = x;
		}
	};
	

	/**
	 * 24. Swap Nodes in Pairs
	 * 两两互换。 Given 1->2->3->4, you should return the list as
	 * 2->1->4->3.
	 * @param head
	 * @return
	 */
	public ListNode swapPairs(ListNode head) {
		ListNode dummy = new ListNode(-1);
		dummy.next = head;
		ListNode pre = dummy;
        ListNode next = null;

        while (head != null && head.next != null) {
            next = head.next;   //2
            head.next = head.next.next; //1->3
            pre.next = next;    //0->2
            next.next = head;   //2->1

			// 往前走
            pre = head;
            head = head.next;
        }
		return dummy.next;
	}
	
	public ListNode swapPairsRec(ListNode head) {
		if (head == null || head.next == null) {
            return head;
        }
        ListNode n = head.next;
        head.next = swapPairs(head.next.next);
        n.next = head;
        
        return n;
	
	}

	/**
	 * 25. Reverse Nodes in k-Group
	 * 
	 * @param head
	 * @param k
	 * @return
	 */
	public ListNode reverseKGroup(ListNode head, int k) {
		if (head == null || head.next == null)
			return head;

		ListNode cur = head;
		int count = 0;
		while (cur != null && count < k) {
			cur = cur.next;
			count++;
		}

		if (count < k)
			return head;        // not enough

		ListNode pre = null;
		cur = head;

		// reverse current k group, and link to reversedHead
		while (count-- > 0) {
			ListNode next = cur.next;
			cur.next = pre;
			pre = cur;
			cur = next;
		}

		// link head (tail after reverse)
		head.next = reverseKGroup(cur, k);

		return pre;
	}


	// Iterative
	public ListNode reverseKGroupIterative(ListNode head, int k) {
		if (head == null || k == 1) {
			return head;
		}
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode pre = dummy;
		int i = 0;
		while (head != null) {
			i++;
			if (i % k == 0) {
				pre = reverse(pre, head.next); // pre & head.next are exclusive
				head = pre.next; // link pre->head
			} else {
				head = head.next;
			}
		}
		return dummy.next;
	}

	// pre & next are exclusive. only middle reverse
	// 可看成 exclusive的start，end
	private ListNode reverse(ListNode pre, ListNode next) {
		ListNode last = pre.next;// where first will be doomed "last"
		ListNode cur = last.next;
		// generate a loop
		while (cur != next) { // change 2 nodes and link pre, next
			last.next = cur.next; // point to 3rd.
			cur.next = pre.next; // reverse direction 1 <- 2
			pre.next = cur; // pre link the finished node(2)
			cur = last.next; // move to next step
		}
		return last;
	}

	// end是inclusive. 把后面的一个个放最前
	private ListNode reverse2(ListNode pre, ListNode end) {
		ListNode cur = pre.next;
		while (pre.next != end) {
			ListNode tmp = cur.next;
			cur.next = tmp.next;
			tmp.next = pre.next;
			pre.next = tmp;
		}
		return cur; // cur not change
	}

    
    public ListNode reverse3(ListNode head, ListNode end) {
        ListNode pre = null;
        while (head != end) {
            ListNode next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        
        return pre;
    }
    
    
    
    /** 382. Linked List Random Node
     * 随机找一个数
     * @param head
     * @return
     * naive方法是1.算出长度 2.random生成一个数 3. 再循环一下找到那个数 
     * 
     * 水塘抽样法.. 见Solution.java
     */
    public int getRandom(ListNode head) {
    	Random random = new Random();
        int count = 0;
        int result = 0;
        
        ListNode cur = head;
        while (cur != null) {
            if (random.nextInt(++count) == 0) {
                result = cur.val;       //现在cur是目前最后一个数，在count位置
            }
            cur = cur.next;
        }
        return result;
    }
	
    
    
    /** 160. Intersection of Two Linked Lists  简单
     * @param headA
     * @param headB
     * @return
     * 最naive方法是算出各自的长，然后走到一样长的地方开始比
     * 
     * 这个是当a提前走完，就跳到headB这么走。b也是同样。其实走的路是一样长的
	 *
	 * say A length = a + c, B length = b + c,
	 * after switching pointer, pointer A will move another b + c steps,
	 * pointer B will move a + c more steps,
	 *
	 * since a + c + b + c = b + c + a + c, it does not matter what value c is.
	 * Pointer A and B must meet after a + c + b (b + c + a) steps. If c == 0, they meet at NULL.
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        //boundary check
        if(headA == null || headB == null) return null;
        
        ListNode a = headA;
        ListNode b = headB;
        
        //if a & b have different len, then we will stop the loop after second iteration
        while( a != b){
        	//for the end of first iteration, we just reset the pointer to the head of another linkedlist
            a = a == null? headB : a.next;
            b = b == null? headA : b.next;    
        }
        
        return a;
    }

    // 正常的naive方法
	public ListNode getIntersectionNodeNaive(ListNode headA, ListNode headB) {
		int lenA = getLength(headA);
		int lenB = getLength(headB);

		// move forward to clear the diff
		while (lenA > lenB) {
			headA = headA.next;
			lenA--;
		}
		while (lenB > lenA) {
			headB = headB.next;
			lenB--;
		}

		// find the intersection
		while (headA != headB) {
			headA = headA.next;
			headB = headB.next;
		}
		return headA;
	}


	/**
	 * Intersection of two linked list - 多种情况
	 * follow up:
	 * 两个没有环，不想交
	 * 两个没有环，相交
	 * 两个有环，不想交
	 * 两个有环，相交
	 */
	public ListNode intersection(ListNode head1, ListNode head2) {
		if (head1 == null || head2 == null) {
			return null;
		}
		ListNode cycle1 = detectCycleII(head1);
		ListNode cycle2 = detectCycleII(head2);

		// 一个有环，另一个没，那就不相交 one has cycle but another one does not, then no intersection
		if ((cycle1 == null && cycle2 != null) || (cycle1 != null && cycle2 == null)) {
			return null;
		}


		// if both no cirlce, then cycle is null and we check the length to the end
		// else we check the length before circle start point
		int len1 = checkLengthBeforeNode(head1, cycle1);
		int len2 = checkLengthBeforeNode(head2, cycle2);
		// check if before circle start point there is intersection
		while (len1 > len2) {
			head1 = head1.next;
			len1--;
		}
		while (len2 > len1) {
			head2 = head2.next;
			len2--;
		}
		for (int i = 0; i < len1; i++) {
			if (head1 == head2) {			// 在cycle之前intersect了
				return head1;
			}
			head1 = head1.next;
			head2 = head2.next;
		}
		// if both no circle, then no intersection for now
		if (cycle1 == null && cycle2 == null) {
			return null;
		}
		// check if the intersection is in the circle
		if (cycle1 == cycle2) {
			return cycle1;
		}

		ListNode mover = cycle1.next;
		while (mover != cycle1) {
			if (mover == cycle2) {
				return mover;
			}
			mover = mover.next;
		}

		return null;
	}

	private int checkLengthBeforeNode(ListNode head, ListNode end) {
		int len = 0;
		while (head != end) {
			head = head.next;
			len++;
		}
		return len;
	}

	private int getLength(ListNode head) {
		int length = 0;
		while (head != null) {
			head = head.next;
			length++;
		}
		return length;
	}


	/**
	 * 708. Insert into a Cyclic Sorted List
	 * @param head
	 * @param insertVal
	 * @return
	 *
	 * 关键判断3种case
	 * 1. 正常中间点  pre < x < cur
	 * 2 & 3. min, max的点，pre > cur... 这就要看 究竟大还是小，跟pre还是cur比
	 *
	 * 记得找到就break，这样最后还能包括  只有head一个node的情况，或者 nodes全部相等，找不到插入的点的情况，这样插哪都一样
	 */
	public ListNode insert(ListNode head, int insertVal) {
		ListNode newNode = new ListNode(insertVal);

		if (head == null)
			return newNode;

		ListNode prev = head;
		ListNode node = head.next;

		while (node != head) {
			int pre = prev.val;
			int cur = node.val;
			if ((pre <= insertVal && insertVal <= cur) ||
				(cur < pre && pre <= insertVal) ||
				(cur < pre && cur >= insertVal)) {

				break;
			}
			prev = prev.next;
			node = node.next;
		}

		prev.next = newNode;
		newNode.next = node;

		return head;
	}
    
    
	
	public static void main(String[] args) {
		ListSolution sol = new ListSolution();
		ListNode n1 = new ListNode(1);
		ListNode n2 = new ListNode(2);
		ListNode n3 = new ListNode(3);
		n1.next = n3;
		n3.next = n2;
		
		sol.sortList(n1);
	}
}
