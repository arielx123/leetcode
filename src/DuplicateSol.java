import java.util.*;

public class DuplicateSol {
	// L3-L28 Remove Duplicates from Sorted List 

		// Definition for singly-linked list.
  public class ListNode {
      int val;
      ListNode next;
      ListNode(int x) {
          val = x;
          next = null;
      }
  }
		 
	    /** 83. Remove Duplicates from Sorted List
	     * 去掉重复多余的，只留一次
	     * Given 1->1->2->3->3, return 1->2->3
	     * @param head
	     * @return
	     */
	    public ListNode deleteDuplicates(ListNode head) {
	    	ListNode cur = head;
	        while (cur != null && cur.next != null) {
	            if (cur.val == cur.next.val) {
	                cur.next = cur.next.next;
	            } else {
	                cur = cur.next;
	            }
	        }
	        return head;
	    }
	    
	 
	    /** 82. Remove Duplicates from Sorted List II
	     * delete all nodes that have duplicate numbers, 只留distinct
	     * Given 1->1->1->2->3, return 2->3
	     * @param head
	     * @return
	     */
	    public ListNode delDupDistinct (ListNode head) {
	    	if (head == null || head.next == null) {
	    		return head;
	    	}
	    	ListNode dummy = new ListNode(0);
	    	dummy.next = head;
	    	head = dummy;				//要删的话需要知道前面的位置，so往前挪
	    	
	    	while (head.next != null && head.next.next != null) {
	    		if (head.next.val == head.next.next.val) {		//下一个开始是否重复
	    			int val = head.next.val;					//先存在val里
	    			while (head.next != null && head.next.val == val) { 	//下一个值==val
	    				head.next = head.next.next;
	    			}
	    		} else {
	    			head = head.next;
	    		}
	    	}
	    	return dummy.next;
	    }
	    
	    // others
	    public ListNode deleteDuplicatesII(ListNode head) {
	        ListNode dummy = new ListNode(-1);
	        dummy.next = head;
	        
	        ListNode pre = dummy;
	        ListNode cur = head;
	        while (cur != null) {
	            if (cur.next != null && cur.val == cur.next.val) {
	                int val = cur.val;
	                while(cur != null && cur.val == val) {
	                    cur = cur.next;
	                }
	                pre.next = cur;
	            } else {
	                pre = cur;
	                cur = cur.next;
	            }
	        }
	        return dummy.next;
	    }
	    
	    
	    /** 27. Remove Element 
	     * 给一个数组和val, 要求删掉数组里的所有val，返回剩下的长度
	     * @param nums
	     * @param val
	     * @return
	     * 需要in place..可以换order
	     * 双指针。一找到 不同的 就调换
	     */
	    public int removeElement(int[] nums, int val) {
	        int i = 0;
	        for (int j = 0; j < nums.length; j++) {
	            if (nums[j] != val) {
	                nums[i] = nums[j];
	                i++;
	            }
	        }
	        return i;
	    }
	    
	    // 适合需要remove的val元素少的情况，这个是 val那i 跟最后的元素调换，这样快点.. 这种会换order，且
	    public int removeElementBetter(int[] nums, int val) {
	        int len = nums.length;
	        int i = 0;
	        while (i < len) {
	            if (nums[i] == val) {
	                nums[i] = nums[len - 1];
	                len--;
	            } else {
	                i++;
	            }
	        }
	        return len;
	    }

	    
	    /**
	     * 26. Remove Duplicates from Sorted Array I - easy
	     * 需要in place，order不能改
	     * @param nums
	     * @return
	     */
	    public int removeDuplicatesFromSortedArray(int[] nums) {
	        if (nums == null || nums.length == 0)
	            return 0;
	        
	        int k = 0;
	        for (int i = 1; i < nums.length; i++) {
	            if (nums[k] != nums[i]) {
	                k++;
	                nums[k] = nums[i];
	            }
	        }
	        return k + 1;
	    }
	    
	    
	    /** 80. Remove Duplicates from Sorted Array II
	     * 排好序的，最多允许2次重复  或k次
	     * @param nums
	     * @return
	     * 后面有更简短的
	     */
	    public int removeDuplicatesII(int[] nums) {
	        if (nums == null || nums.length == 0) {
	            return 0;
	        }
	        
	        int index = 0;
	        int dup = 1;
	        for (int i = 1; i < nums.length; i++) {
	            if (nums[index] == nums[i]) {
	                if (dup < 2) {					//也可以为K次
	                    index++;
	                	nums[index] = nums[i];		// 记得赋值，否则没有copy正确的值
	                }
	                dup++;
	            } else {
	                dup = 1;
	                index++;
                	nums[index] = nums[i];
	            }
	        }
	        return index + 1;
	    }	    
	    
	    // 更简短！也可以扩展为 最多重复 K次
	    // 也可以idx和i都从2开始。 注意是 nums[idx - k] 与nums[i]比!!
	    // 允许两次的话就从2开始，n != nums[idx-2]时就换
	    public int removeDuplicatesBetter(int[] nums, int k) {
	        int i = 0;
	        for (int num : nums) {
	            if (i < k || nums[i - k] < num) {	//记得是 arr[i - k], 不是n-k!!
	                nums[i++] = num;
	            }
	        }
	        return i;
	    }
	    
	    
	    /** 219. Contains Duplicate II
	     * UNsorted Array，找到nums[i] == nums[j] 且 j-i <= k
	     * @param nums
	     * @param k
	     * @return
	     * 用set维护一个大小为k的sliding window
	     * hashset的大小 <= k。一旦大于k，就删掉最老的那个。
	     */
	    public boolean containsNearbyDuplicate(int[] nums, int k) {
	        HashSet<Integer> set = new HashSet<>();
	        for (int i = 0; i < nums.length; i++) {
	            if (set.contains(nums[i])) {
	                return true;
	            } else {
	                set.add(nums[i]);
	                if (set.size() > k) {
	                    set.remove(nums[i-k]);
	                }
	            }
	        }
	        
	        
	        // =========或者用HashMap存index================
	        Map<Integer, Integer> map = new HashMap<>();
	        
	        for (int i = 0; i < nums.length; i++) {
	            if (map.containsKey(nums[i])) {
	                if (i - map.get(nums[i]) <= k) {
	                    return true;
	                }
	            }
	            map.put(nums[i], i);
	        }
	        
	        return false;
	    }
	    
	    
	    
	    /** 220. Contains Duplicate III
	     * abs(nums[i]- - nums[j]) <= t. && abs(i - j) <= k
	     * 这又是一道sliding window的题，要维护k的窗口
	     * @param nums
	     * @param k
	     * @param t
	     * @return
	     * 这里这个naive方法是 O(n * min(k,n))
	     * j从 i-k 开始（或者0），j < i. 这样就能维护 k 的sliding window.  
	     * 
	     * 上面那题为何需要set？因为上面只是duplicate的，所以用set。
	     * 但这题是一个范围，所以不用set, j直接在(i-k, i)的范围内比就行。
	     */
	    public boolean containsNearbyAlmostDuplicateNaive(int[] nums, int k, int t) {
	        for (int i = 0; i < nums.length; i++) {
	            for (int j = Math.max(0, i - k); j < i; j++) {      //sliding window
	                if (Math.abs(nums[i] - nums[j]) <= t)
	                    return true;
	            }
	        }
	        return false;
	    }
	    
	    class Pair {
	        int val;
	        int index;
	        Pair(int v, int i) {
	            this.val = v;
	            this.index = i;
	        }
	    }
	    
	    // 下面这个就先按照值val来sort，再看index是否符合
	    public boolean containsNearbyAlmostDuplicateSort(int[] nums, int k, int t) {
	        if (nums == null || nums.length < 2 || t < 0 || k < 1) {
	            return false;
	        }
	        int len = nums.length;
	        Pair[] pair = new Pair[len];
	        for(int i = 0; i < len; i++) {
	            pair[i] = new Pair(nums[i], i);
	        }
	        
	        Arrays.sort(pair, new Comparator<Pair> () {
	          public int compare(Pair p1, Pair p2) {
	              return p1.val - p2.val;
	          } 
	        });
	        
	        for(int i = 0; i < len; i++) {   //因为对排好序了，n[i]后面的n[j]会更大，所以看差值 <=t, 并且看距离<=k就行
	            for(int j = i + 1; j < len && Math.abs((long)pair[j].val - (long)pair[i].val) <= (long)t; j++){
	                int indexDiff = Math.abs(pair[i].index - pair[j].index);
	                if (indexDiff <= k) {
	                    return true;
	                }
	            }
	        }
	        return false;
	    }
	    
	    
	    /**  220. Contains Duplicate III - BST做法
	     * @param nums
	     * 因为是找范围，其实只要找到一个就行TRUE，那这个就是 最小上限- nums[i] <= t
	     * 最小上限 就是 找大于n[i]的最小值。如果连第一个比n[i]大的数都超过t..那后面更大的数更没可能
	     * 
	     * 所以求出最大下限和最小上限，那就是java里的TreeMap二叉搜索树BST.因为左孩子floor就是最大下限，右孩子就是最小上限
	     * 用一个大小为k的滑动窗口，将窗口内的元素组织成一个BST，每次向前滑动一步，添加一个新元素，同时删除一个最老的元素，不断更新BST。
	     * 若k < n，那么复杂度就是 nlogk , BST里的增，删，查都是logk。然后要遍历n个元素。BST大小为k
	     */
	    public boolean containsNearbyAlmostDuplicateBST(int[] nums, int k, int t) {
	        TreeSet<Integer> bst = new TreeSet<>();
	        for (int i = 0; i < nums.length; i++) {
	            Integer ceil = bst.ceiling(nums[i]);   // smallest ele that >= n[i]。就是sorted以后第一个比n[i]大的
	            if (ceil != null && ceil - nums[i] <= t)    return true;
	            
	            Integer floor = bst.floor(nums[i]);		// 最大下限
	            if (floor != null && nums[i] - floor <= t)      return true;
	            
	            bst.add(nums[i]);
	            if (i >= k)						//或者bst.size > k
	                bst.remove(nums[i-k]);
	        }
	        return false;
	    }
	    
	    
	    /** 220. Contains Duplicate III - bucket sort + HashMap做法
	     * abs(nums[i] - nums[j]) <= t. && abs(i - j) <= k
	     * @param nums
	     * A. 利用bucket sort的其中一步，分成几个大小为size(t+1)的buckets。
	     * 	  这样能保证range在 <= (t)diff的范围内，所有数都在同一个或者左右相邻的bucket里
	     * B. k是sliding window, 用hashmap来记录目前的buckets，map的size其实就是坐标i~j的范围。
	     * 	  如果map的大小超过k个就删掉oldest，保持abs(i-j) <= k
	     * 	  如果map里有 bucketID, 和当前n[i]的bckID一样或者相隔，那就说明找到了
	     * 
	     * 由于这题t可能为0，所以我们size = t+1, 防止 /0的错误发生
	     * 同时要注意!!!除了找当前bucket，还要找左右相邻的，有可能在相邻的边界 
	     * 		比如t=3, 9和12符合条件，但(除以大小t)放bucket时，分别在3,4号bucket，所以就要找相邻的bucket
	     * 
	     * 另外，nums[i]有可能为负数，in java -3/5=0, 这就跟整数情况 3/5=0 一样了，所以要 -1 负数时(n + 1) / size - 1;
	     * O(n)
	     */
	    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
	        if (t < 0)  return false;   
	        Map<Long, Long> bucketMap = new HashMap<>();    //size is k
	        
	        for (int i = 0; i < nums.length; i++) {
	            long bucketId = getBucketID(nums[i], t);      //get bucket's id，大概就是 n[i]/t 看落在哪个bucket里
	            
	            if (bucketMap.containsKey(bucketId))		//已经有数在同一个bucket, 那肯定 abs(n[i]-n[j]) <= t, 就不用再比较
	                return true;						// 并且map大小为k, 所以index也符合
	            if (bucketMap.containsKey(bucketId - 1) && nums[i] - bucketMap.get(bucketId-1) <= t)
	                return true;
	            if (bucketMap.containsKey(bucketId + 1) && bucketMap.get(bucketId+1) - nums[i] <= t)
	                return true;
	            
	            bucketMap.put(bucketId, (long)nums[i]);		// 直接更新用最新的n[i]，保证 idx <= k
	            
	            if (i >= k)     // window size exceed k, need to remove oldest
	                bucketMap.remove(getBucketID(nums[i - k], t)); 
	        }
	        return false;
	    }
	    
	    private long getBucketID(long n, int t) {
	        long size = (long) t + 1;
	        if (n < 0) {    //in java -3/5=0, but we want -1
	            return (n + 1) / size - 1;
	        }
	        return n / size;
	        
	    //  return n < 0 ? (n+1)/w - 1 : n/w;  或者直接一句
	    }
	    
	    
	    
	    /** 316. Remove Duplicate Letters - increasing stack
	     * 删掉duplicate数，那些重复的只能保留一个，并且返回顺序是之前最小字母顺序
	     * Given "bcabc"， Return "abc"
	     * Given "cbacdcbc"， Return "acdb".. !!注意这里，b在cd后面
	     * @param s
	     * @return
	     * 要扫一遍知道duplicate，然后把符合条件的放sb / stack里，让sb里尽量保持递增
	     * 
	     * 1. 扫一遍存每个letter的出现次数
	     * 2. 再扫一遍，把结果放sb / stack里, 且次数-1
	     * 	  2.1 freq[c - 'a']--;		//记得先--，如果只出现过1次的，减完就是0，那么下面while不能删。同样也能track重复的剩多少
	     * 	  2.2 如果stack里有的，可以不用管着先（有重复） --> 这需要visit[] 来记录stack里是否有
	     *    2.3 如果新的数，不在栈里，那就要跟栈顶peek比较了 
	     *        while  小于 < peek && peek重复，那么 就可以stack/sb里的duplicate数，放 最小的新的数
	     * 		  去掉stack里的duplicate时记得把 used[peek] -> false, 这样后面遇到可以再加进来
	     * 	  2.4 把这个char加进去.. 
	     * 
	       所以在判断之前，先把重复的都找出来，这样知道后面能不能删掉
	       用used[] 来存 是否在stack里，这样后面重复的可以删掉前面stack里已经有的
	       
	     */
	    public String removeDuplicateLetters(String s) {
	    	int[] freq = new int[26];
	        char[] ch = s.toCharArray();
	        
	        for (char c : ch) {
	            freq[c - 'a']++;
	        }
	        
	        StringBuilder sb = new StringBuilder();
	        boolean[] used = new boolean[26];
	        
	        for (char c : ch) {
	            freq[c - 'a']--;		//记得--，如果只出现过1次的，减完就是0，那么下面while不能删。同样也能track重复的剩多少
	            if (used[c - 'a'])  continue;
	            
	            // 如果当前char小于之前sb里的字母，并且前面的有重复，那就可以替换掉
	            while (sb.length() > 0 && freq[sb.charAt(sb.length() - 1) - 'a'] > 0 && sb.charAt(sb.length() - 1) > c) {
	                used[sb.charAt(sb.length() - 1) - 'a'] = false;
	                sb.deleteCharAt(sb.length() - 1);
	            }
	            
	            sb.append(c);
	            used[c - 'a'] = true;
	            
	            /*相当于
	            while(!st.isEmpty() && s<st.peek() && res[st.peek()-'a'] > 0){ 
		            visited[st.pop()-'a']=false;
		        }
		        最后还要pop到sb里
		        */
	        }
	            
	        return sb.toString();
	    }
	    
	    
	    
	    /** 287. Find the Duplicate Number - O(logn)
	     * 一个没有sort的nums数组里有 n + 1个数，从 1 ~ n。有一个duplicate的数，但它可能出现多次
	     * 不能额外空间(hashset)，不能改变顺序(sort)
	     * @param nums
	     * @return
	     * 这里二分查找。
	     * 因为整体都是连续的，那个分大小两半的话应该都是5. 如果其中一半部分的个数 > 5，那就是有重复的
	     * 比如给10个数，n=10, 那么mid=5.
	     * 那么 <= 5的个数有5个， >5的也是5个。如果 <= 5的数 总共有6个，那就是小部分这半，就要搜[1, 5]
	     */
	    public int findDuplicate(int[] nums) {
	        int low = 1, high = nums.length - 1;		// 个数 分一半，而不是里面的数
	        while (low < high) {
	            int mid = low + (high - low) / 2;
	            int count = 0;
	            for (int n : nums) {		// O(n) 扫所有
	                if (n <= mid) {
	                    count++;			
	                }
	            }
	            if (count > mid) {			// 说明小的数太多, O(logn)来去掉一半
	                high = mid;
	            } else {
	                low = mid + 1;
	            }
	        }
	        return low;
	    }
	    
	    
	    
	    /** 287. Find the Duplicate Number - O(n)
	     * @param nums
	     * @return
	     * 跟linked list cycle很像，快慢指针
	     * https://discuss.leetcode.com/topic/25685/java-o-n-time-and-o-1-space-solution-similar-to-find-loop-in-linkedlist
	     * e.g. nums = [1, 2, 3, 4] => 0 -> 1, 1 -> 2, 2 -> 3, 4 -> 5
	     * 
	     * However, since the array has one duplicate number, we will see at 
	     * least two indexes pointing to the the same value. 
	     * e.g. nums = [1, 2, 1, 3] => 0 -> 1, 1 -> 2, 2 -> 1, 3 -> 3 
	     * => 0 -> 1 -> 2 -> 1 -> 2 -> 1 ...
	     * In this case, a cycle will be generated.
	     * 
	     * To detect the start of the cycle, we can use two pointers to traverse 
	     * the 'circular array'.
	     */
	    public int findDuplicateCycle(int[] nums) {
	        int slow = nums[0];
	        int fast = nums[nums[0]];       //走2步
	        while (slow != fast) {
	            slow = nums[slow];
	            fast = nums[nums[fast]];
	        }
	        
	        // 开始找环的起点，也就是那个重复的数
	        fast = 0;
	        while (slow != fast) {
	            slow = nums[slow];
	            fast = nums[fast];
	        } 
	        return fast;
	    }
	    
	    
	    
	    /** 442. Find All Duplicates in an Array
	     * 一个数组  1 ≤ a[i] ≤ n (n = size of array),有的数出现1或2次. 找出出现2次的
	     * @param nums
	     * @return
	     * 出现一次是负数，第二次出现如果发现是负数，那就找到了
	     * 
	     * 另一种方法..调换顺序成1，2，3，4 跟上面一样
	     */
	    public List<Integer> findDuplicates(int[] nums) {
	        List<Integer> list = new ArrayList<>();
	        for (int i = 0; i < nums.length; i++) {
	            int index = Math.abs(nums[i]) - 1;
	            if (nums[index] < 0) {      //if already appear once, then is negative, we found it
	                list.add(index + 1);
	            }
	            nums[index] = -nums[index];
	        }
	        return list;
	    }
	    
	    
	    public static void main(String[] args) {
	    	DuplicateSol sol = new DuplicateSol();
//	    	System.out.println(sol.removeDuplicates(new int[]{1,1,2}));
//	    	System.out.println("sdfsdfsdf);

	    	
	    }
}
