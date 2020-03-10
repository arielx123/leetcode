package twoPointer;

import java.util.*;


class ListNode {		//去掉public
    int val;
    ListNode next;
    ListNode(int x) {
        val = x;
        next = null;
    }
}

public class TwoPointSol {
	/*
	986. Interval List Intersections
	/*有A和B两个闭区间数组，它们分别都是排过序的，且每个数组里的区间之间互不相交。请求出两个数组中闭区间的交。Java two pointers O(m + n)
        */
	*/
		

    public int[][] intervalIntersection(int[][] A, int[][] B) {
        /*有A和B两个闭区间数组，它们分别都是排过序的，且每个数组里的区间之间互不相交。请求出两个数组中闭区间的交。Java two pointers O(m + n)
        */
   
        if(A == null || A.length == 0 || B == null || B.length == 0)
            return new int[][]{};
        List<int[]> res = new ArrayList<>();

        int i = 0, j = 0;
        int startMax, endMin;
        while(i < A.length && j < B.length){
            startMax = Math.max(A[i][0], B[j][0]);
            endMin = Math.min(A[i][1], B[j][1]);

            if(endMin >= startMax)
                res.add(new int[]{startMax, endMin});

            // Remove the interval with the smallest endpoint
            if(A[i][1] == endMin)
                i++;
            if(B[j][1] == endMin) 
                j++;
        }

        return res.toArray(new int[res.size()][2]);
    }

	/** 27. Remove Element
	 *  remove all instances of that value in place and return the new length.
	 *  顺序可以改变
	 *  
	 *  两个指针i, j从头开始。不是val时就一起++，当遇到val时只有i++，直到不同时j才往前走
	 * @param nums
	 * @param val
	 * @return
	 */
	public int removeElement(int[] nums, int val) {
        int len = nums.length;
        int i = 0;
        int j = 0;
        while (i < len) {
            if (nums[i] != val) {
                nums[j] = nums[i];	//碰到相同时，i++. j不动，直到不同时把后面的n[i]值给n[j]
                j++;
            }
            i++;
        }
        return j;
    }
	
	// 这个头尾跑.. 如果要remove的元素很少的话，这样更好..不过这样顺序就打乱了
	public int removeElement1(int[] nums, int val) {
        if (nums == null || nums.length == 0)   return 0;
        
        int i = 0, n = nums.length;
        while (i < n) {
            if (nums[i] == val) {           //不用另外考虑n[j]==val，直接调转后让n[i]来弄
                nums[i] = nums[n - 1];
                n--;
            } else {
                i++;
            }
        }
        return n;
    }
	
	
	
	/** Move Zeroes
	 * move all 0's to the end of it while maintaining the relative order of the non-zero elements.
     * nums = [0, 1, 0, 3, 12] --> [1, 3, 12, 0, 0].
     * @param nums
     * 双指针，一快一慢.. i快..跳过0，等非0时再跟慢的j来swap
     */
	public void moveZeroes(int[] nums) {
        if (nums == null || nums.length <= 1)   return;
        
        int j = 0;
        for(int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {         // n[i]为0时就i++, j不动
                int tmp = nums[i];          //swap
                nums[i] = nums[j];
                nums[j] = tmp;
                j++;
            }
        }
    }
	
	
	/** Two Sum
	 * Input: numbers={2, 7, 11, 15}, target=9
		Output: index1=0, index2=1. 下标从0开始
	 *
	 * 可以用hashmap. 算余数
	 */
	public int[] twoSum(int[] nums, int target) {
		Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int diff = target - nums[i];
            if (map.containsKey(diff)) {
                return new int[]{map.get(diff), i};
            } else {
                map.put(nums[i], i);
            }
        }
        return new int[2];
    }
	
	
	/** 167. Two Sum II - Input array is sorted 
	 * 两个指针，头尾移
	 * @param nums
	 * @param target
	 * @return
	 */
	public int[] twoSumSorted(int[] nums, int target) {
        int[] result = new int[2];
        int i = 0;
        int j = nums.length - 1;
        while (i < j) {
            int sum = nums[i] + nums[j];
            if (sum == target) {
            	return new int[]{i + 1, j + 1};
            } else if (sum > target) {
                j--;   
            } else {
                i++;
            }
        }
        return result;
    }
	
	
	/** Three Sum
	 * Elements in a triplet (a,b,c) must be in non-descending order. (ie, a ≤ b ≤ c)
	   The solution set must not contain duplicate triplets.
	   不能有duplicate的solution
	 *  a+b+c=0。 找n个solution
	 */
	public ArrayList<List<Integer>> threeSum(int[] nums) {
        ArrayList<List<Integer>> result = new ArrayList<>();
        //可以用HashSet存多个solution。避免重复... 但其实可以不用set，多余的空间
        // 最后 result.addAll(set);			//所有set里的结果放到result
        Arrays.sort(nums);
        int len = nums.length;
        for (int i = 0; i < len - 2; i++) {
            if (i > 0 && nums[i] == nums[i-1]) {
                continue;
            }
            int j = i + 1;
            int k = len - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum > 0) {
                    k--; 
                } else if (sum < 0){
                    j++;
                } else {	// == 0
                    result.add(Arrays.asList(nums[i], nums[j], nums[k]));
                    
                    do {j++;} while (j < k && nums[j] == nums[j - 1]);
                    do {k--;} while (j < k && nums[k] == nums[k + 1]);
                    
//                    j++;		//同上。但是记得要先加j++, k--. 否则直接while会跳过一步
//                    k--;		//如果要先while, 再j++,k--的话。while那需要跟[j+1], [k-1]比（跟后面的数比才不会错过当前这个）
//                    while(j < k && nums[j] == nums[j - 1]) j++;  // Skip same results
//                    while(j < k && nums[k] == nums[k + 1]) k--;  // Skip same results
                }
            }
        }
        return result;
    }
	
	// 跟上面一样.. 只是把判断 j, k重复的放到前面..
	public ArrayList<List<Integer>> threeSumTarget(int[] nums, int target) {
        ArrayList<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        int len = nums.length;
        for (int i = 0; i < len - 2; i++) {
            if (i > 0 && nums[i] == nums[i-1]) {
                continue;
            }
            int j = i + 1;
            int k = len - 1;
            while (j < k) {
            	if (j > i + 1 && nums[j] == nums[j-1]) {
            		j++;
            		continue;
            	}
            	if (k < len - 1 && nums[k] == nums[k+1]) {
            		k--;
            		continue;
            	}
            	
                int sum = nums[i] + nums[j] + nums[k];
                if (sum > target) {
                    k--; 
                } else if (sum < target){
                    j++;
                } else {	// == 0
                    result.add(Arrays.asList(nums[i], nums[j], nums[k]));
                    j++;
                    k--;		//记得往前走
                }
            }
        }
        return result;
    }
	
	// 这个用HashMap，但很慢 TLE. 且占空间
	public List<List<Integer>> threeSumHashMap(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        Set<List<Integer>> set = new HashSet<>();
        
        for (int i = 0; i < nums.length - 1; i++) {
            map.clear();		//记得一定要clear，否则map会有上次nums[i-1]的值，就不对
            
            for (int j = i + 1; j < nums.length; j++) {
                if (map.containsKey(-nums[j])) {
                	List<Integer> list = Arrays.asList(nums[i], nums[map.get(-nums[j])], nums[j]);
                    Collections.sort(list);		//如果不允许重复
                    set.add(list);
                } else {
                    map.put(nums[i] + nums[j], j);
                }
            }
        }
        return new ArrayList<>(set);
    }
	
	
	
	/** 16. 3Sum Closest
	 * Return the sum of the three integers that their sum closest to target
	 * @param nums
	 * @param target
	 * @return
	 */
	public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int len = nums.length;
        int result = nums[0] + nums[1] + nums[len - 1];
        for (int i = 0; i < len - 2; i++) {
            int j = i + 1;
            int k = len - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == target) {
                    return sum;
                } else if (sum > target) {
                    k--;
                } else {
                    j++;
                }
                if (Math.abs(target - sum) < Math.abs(target - result)) {
                    result = sum;
                }
            }
        }
        return result;
    }
	

	/** 259. 3Sum Smaller - 假设都是distinct数，或者允许结果有重复
	 * 3 sum < target, return how many triplets
	 * @param nums
	 * @param target
	 * 一旦找到sum < target ，那count += k - j; 因为k--更小，都符合
	 */
	public int threeSumSmaller(int[] nums, int target) {
        if (nums.length < 3)    return 0;
        
        Arrays.sort(nums);
        int count = 0;
        
        for (int i = 0; i < nums.length; i++) {
            int j = i + 1;
            int k = nums.length - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum < target) {
                    count += k - j;        // k往前肯定更小，所以有k-j个
                    j++;				//j向前走
                } else {
                    k--;
                }
            }
        }
        return count;
    }
	
	
	/** 259. 3Sum Smaller  - nums[]里有duplicate怎么办
	 * 3 sum < target, return how many triplets
	 * @param nums
	 * @param target
	 * @return
	 */
	public int threeSumSmaller2(int[] nums, int target) {
        int count = 0;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
        	if (i > 0 && nums[i] == nums[i-1]) {		//跳过duplicate
        		continue;
        	}
            int j = i + 1;
            int k = nums.length - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum < target) {
                    count += k - j;     //since k is the largest, all <= k should be < target
                    while (j < k && k < nums.length-1 && nums[k] == nums[k+1]) {
                    	k--;			//if duplicate, need change
                    	count--;
                    }
                    j++;	// go to next round of while loop
                    while (j < k && nums[j] == nums[j-1])	j++; 
                } else {
                    k--;
                }
            }
        }
        return count;
    }
	
	
	/** 3Sum Smaller - google
	 * Print out all the result..  followup, 有duplicate怎么办
	 * @param nums
	 * @param target
	 * @return
	 */
	public List<int[]> threeSumSmallerList(int[] nums, int target) {
        Arrays.sort(nums);
        List<int[]> result = new ArrayList<>();
        for (int i = 0; i < nums.length - 2; i++) {
        	if (i > 0 && nums[i] == nums[i-1]) {
        		continue;
        	}
            int j = i + 1;
            int k = nums.length - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum < target) {
                	int tmp = k;	//要用tmp存，因为每次加result，而不是上面那样算个数
                	
                	// 开始打印(j, k]之间的结果
                	while (j < tmp) {
                    	result.add(new int[]{nums[i], nums[j], nums[tmp--]});
                    	while (j < tmp && nums[tmp] == nums[tmp+1])	tmp--;	//avoid duplicate
                    }
                	
                    j++;
                    
                    while (j < k && nums[j] == nums[j-1])	j++;	//avoid duplicate
                } else {
                    k--;
                }
            }
        }
        return result;
    }
    
    
    /** 4 sum
	 * a + b + c + d = target
	 * 比3sum多一层for循环. 2个头尾，2个从中间。这样还是只有start, end动，方便管理
	 * @param num
	 * @param target
	 * @return
	 * i在头，中间m, n, 然后j在尾
	 * 跳过duplicate的情况：do {m++;} while (m < n && nums[m] == nums[m-1]);
	 * 提前break减少useless运算：if nums[i] * 4 > target 或 nums[j] * 4 < target
	 * 剪枝!!!!
	 */
	public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        
        for (int i = 0; i < nums.length - 3; i++) {
            if (i > 0 && nums[i] == nums[i-1]) {
                continue;
            }
            if (nums[i] * 4 > target) {     //no need to compute further提前结束
                return result;  
            }
            
            for (int j = nums.length - 1; j > i + 2; j--) {
                if (nums[j] * 4 < target) {		//如果j差不多到中间了，然而n[j]*4已经最大但<target, 那么i++要变大
                    break;
                }
                if (j < nums.length - 1 && nums[j] == nums[j+1]) {
                    continue;
                }
                int m = i + 1, n = j - 1;
                while (m < n) {
                    int sum = nums[i] + nums[j] + nums[m] + nums[n];
                    if (sum < target) {
                        m++;
                    } else if (sum > target) {
                        n--;
                    } else {		//Arrays.asList返回List而不是AL，所以要cast
                        result.add(Arrays.asList(nums[i], nums[j], nums[m], nums[n]));
                        while (m < n && nums[++m] == nums[m-1]);
                        while (m < n && nums[--n] == nums[n+1]);
                        // do {m++;} while (m < n && nums[m] == nums[m-1]);
//                        do {n--;} while (m < n && nums[n] == nums[n+1]);
                    } 
                }
            }
        }
        return result;
        
        /* 也可以3个在头，i,j,m..只有n在尾
        for (int j = i + 1; j < nums.length - 2; j++) {		//也可以，但没有下面那种可以提前break
            if (j > (i + 1) && nums[j] == nums[j-1]) {
                continue;
            }								
            int m = j + 1;
            int n = nums.length - 1;
        */
    }
	
	
	
	/** 454. 4Sum II
	 * 给4个数组，在每个数组里找一个数，使4个数和为0。返回这种tuples的个数
	 * A[i] + B[j] + C[k] + D[l] is zero.
	 * @return
	 * 用HashMap两两存.. 先把A, B数组用HashMap存他们的sum.. 然后C, D数组看有没存在
	 * O(n^2).
	 */
	public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        Map<Integer, Integer> map = new HashMap<>();
    
        for(int i = 0; i < C.length; i++) {
            for(int j = 0; j < D.length; j++) {
                int sum = C[i] + D[j];
                if (!map.containsKey(sum)) {
                    map.put(sum, 1);
                } else {
                    map.put(sum, map.get(sum) + 1);
                }
            }
        }
        
        int count = 0;
        for(int i = 0; i < A.length; i++) {
            for(int j = 0; j < B.length; j++) {
                int sum = A[i] + B[j];
                if (map.containsKey(-sum)) {
                    count += map.get(-sum);
                }
            }
        }
        return count;
    }
	
	
    /** K Sum  k个数总和为target - general
     * @param nums
     * @param target
     * @return
     * 记录start和length作为头尾，k=2时算pair。其他时候就是for循环<len-k+1
     * 复杂度是O(n^(k-1)). k=4时就是n^3复杂度
     */
    public List<List<Integer>> kSumGeneral(int[] nums, int target, int k) {
        Arrays.sort(nums);
        return kSum(nums, target, k, 0);
    }
    
    public List<List<Integer>> kSum(int[] nums, int target, int k, int start) {
        List<List<Integer>> result = new ArrayList<>();
        int len = nums.length;
        if (start >= len) {
            return result;
        }
        
        if (k == 2) {
            int i = start, j = len - 1;
            while (i < j) {
                if (nums[i] + nums[j] == target) {
                	// Arrays.asList返回fixed-size的list,这样后面l.add(0, nums[i])想再加东西就不行。
                	//所以要再new一个ArrayList，这样size可以变。或者把下面这句用普通的new, 再一个个加
                    result.add(new ArrayList<Integer>(Arrays.asList(nums[i], nums[j])));
                    while (i < j && nums[++i] == nums[i-1]);
                    while (i < j && nums[--j] == nums[j+1]);
                } else if (nums[i] + nums[j] > target) {
                    j--;
                } else {
                    i++;
                }
            }
        } else {
            for (int i = start; i < len - k + 1; i++) {
            	if (i > start && nums[i-1] == nums[i]) 
                    continue;			//跳过重复的
            	
            	if (nums[i] * k > target) {		//太大就不用算后面的了
                    return result;
                }
            	
                List<List<Integer>> list = kSum(nums, target - nums[i], k - 1, i + 1);
                if (list != null && list.size() > 0) {
                    for (List<Integer> l : list) {
                    	l.add(0, nums[i]);		//因为得到最里面k=2的pair，要在外面加nums[i]
                    }
                    result.addAll(list);
                }
            }
        }
        return result;
    }
    
    
    /** K Sum - number of solutions  - 但要求nums都为正数 > 0 ，且target > 0
     * 给数组A，找出其中k个数的和为target 的solution数目
     * @param A
     * @param k
     * @param target
     * @return
     * 上面那题是输出solution的结果，这个只是个数，所以很容易想到用DP 
     * f[i][j][t]：前i(n)个数里选j(k)个数的和为t(target) 的方案数
     */
    public int kSumDP(int A[], int k, int target) {
        int n = A.length;
        int[][][] f = new int[n+1][k+1][target+1];   //前n个数里选k个数的和为target 的方案数
        
        //initialize
        for (int i = 0; i <= n; i++) {
            f[i][0][0] = 1;      // 不管多少数，选0个数和为0的 都是1种方案
        }
        
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= k && j <= i; j++) {     //记得 j <= k && j<= i
                for (int t = 1; t <= target; t++) {
                	f[i][j][t] = f[i-1][j][t];      // 不取当前值的话
                	
                    	//下标所以是A[i-1], 当前值
                    if (t >= A[i-1]) {           // 若t大于当前值
                        f[i][j][t] += f[i-1][j-1][t - A[i-1]];    //取当前值
                    }
                    
                }
            }
        }
        return f[n][k][target];
     }
	
    
    
    /**
     * 611. Valid Triangle Number   -  Triangle Count
     * 给一个数组，找出3条边使得能组成三角形。return最多有多少对
     * 跟3sum smaller很像
     * @param T
     * @return
     * 那就是两两相加要 > 第三边
	 * 因为n[i] < n[j] < n[k]. 所以只用判断前面2个小的 n[i]+n[j] > n[k]。 因为另外2种情况都肯定 > ，不用考虑了
	 * 所以里面用2sum larger
     * 让开始的left+right > end, 而且count += r - l 因为 left++ 与不变的right 的和 肯定也大于end
     */
    public int triangleCount(int T[]) {
    	Arrays.sort(T);
    	int count = 0;
    	
    	int left = 0, right = T.length - 1;
    	for (int end = 0; end < T.length; end++) {
    		left = 0;
    		right = end - 1;
    		while (left < right) {
    			if (T[left] + T[right] > T[end]) {
    				count += right - left;			//跟3sum smaller类似

                    // 如果要返回所有result
//                    for (int k = left; k < right; k++) {         // 注意是改left
//                        List<Integer> tri = Arrays.asList(input[k], input[right], input[end]);
//                        result.add(tri);
//                    }

    				right--;
    			} else {
    				left++;
    			}
    		}
    	}
    	
    	return count;
    }

    // 看能否组成三角形.. 降低复杂度.. 一个for 循环就行
    public boolean canFormTriangle(int[] input) {
        Arrays.sort(input);

        for (int i = input.length - 1; i >= 2; i--) {
            int end = input[i];
            int right = input[i - 1];
            int left = input[i - 2];

            if (left + right > end) {
                return true;
            }
        }
        return false;
    }



    /**
     * 1010. Pairs of Songs With Total Durations Divisible by 60
     * 找加起来能被60整除的song pair.. 排列组合 所以要 乘 *
     * @param time
     * @return
     *
     * 用map存 time % 60.. 之后  pairs += count[i] * count[60 - i];
     *
     * 注意30和60的情况
     */
    public int numPairsDivisibleBy60(int[] time) {
        int[] count = new int[61];

        for (int t : time) {
            count[t % 60]++;
        }

        int pairs = 0;
        for (int i = 0 ; i <= 30; i++) {
            if (i == 0 || i == 30) {           // special case
                pairs += count[i] * (count[i] - 1) / 2;
            } else {                         // 算的是排列组合
                pairs += count[i] * count[60 - i];
            }
        }

        return pairs;
    }
	
    
    
    /** 11. Container With Most Water
     * Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). 
     * n vertical lines are drawn such that the two end points of line i is at (i, ai) and (i, 0). 
     * Find two lines, which together with x-axis forms a container, such that the container contains the most water
     * height[] 就是Ai高。找两条线的面积，但是短板效应
     * 
     * 双指针分别指头尾。算面积。保留高的不动，短的移
     * @param height
     * @return
     */
    public int maxArea(int[] height) {
        int max = 0;
        int i = 0;
        int j = height.length - 1;
        
        while (i < j) {
            int area = (j - i) * Math.min(height[i], height[j]);	//长是min那个
            max = Math.max(max, area);
            if (height[i] < height[j]) {        // keep the taller one
                i++;                            // shorter one move to find taller
            } else {
                j--;
            }
        }
   // ===============END===========================
        
        // 下面这个快一点。 主要在i++或j--时，找到比之前h[i]大的就停下来
        while (i < j) {
            int area = Math.min(height[i], height[j]) * (j - i);
            max = Math.max(max, area);
            int lh = height[i];
            int rh = height[j];
            if (height[i] < height[j]) {
                while(lh > height[++i]);
            } else {
                while(rh > height[--j]);
            }
        }
        
        return max;
    }
    
    

    
    /** 42. Trapping Rain Water
     * Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6. 能灌多少水。每个bar为1的宽
     * @param height
     * @return
     * Two pointer
     * 左右向中间靠拢. 当前n[i]小的话，就要加水/更新leftMax 往前走
     * 同时 leftMax是左边最高的，left每往前走一格，如果比leftMax矮，那就res += leftMax - n[left]
     * 每次一格格地走，所以宽都是1.. leftMax也会一直更新
     * right同理
     */
    public int trap(int[] height) {
        if (height == null || height.length <= 1)   return 0;
        
        int i = 0;
        int j = height.length - 1;
        int leftMax = 0, rightMax = 0;
        int result = 0;
        
        while (i < j) {
            if (height[i] < height[j]) {
                if (leftMax < height[i])    leftMax = height[i];    //遇到高的就更新leftMax
                else       result += leftMax - height[i]; //否则当前h[i]矮，那就加水  
                i++;        
            } else {
                if (rightMax < height[j])   rightMax = height[j];
                else        result += rightMax - height[j];
                j--;
            }
        }
        return result;
    }
    
    
    
    
    /**The smallest Difference
     * 最小的 abs(A[i] - B[j]). 直接sort，再双指针移动
     * @param A
     * @param B
     * @return
     */
    public int smallestDifference(int[] A, int[] B) {
        Arrays.sort(A);
        Arrays.sort(B);
        
        int min = Integer.MAX_VALUE;
        int i = 0, j = 0;
        while (i < A.length && j < B.length) {
            if (min > Math.abs(A[i] - B[j])) {
                min = Math.abs(A[i] - B[j]);
            }
            if (A[i] > B[j]) {
                j++;
            } else if (A[i] < B[j]) {
                i++;
            } else {        // equal
                return 0;
            }
        }
        return min;
    }
    
    
    

    /** 75. Sort Colors 
     * sort 0, 1, and 2 数组nums有三种颜色，让所有颜色分在一起，变成 红红(0)..白白(1)..蓝蓝(2)..
     * @param A
     */
    public void sortColors(int[] nums) {
        int left = 0;					//left表示前面排完0
        int right = nums.length - 1;		//后面排完2.. 两边往中间靠
        int i = 0;							// i向后走
        
        while (i <= right) {
            if (nums[i] == 0 && left < i) {		//不一样就swap, 只移left 
                swap(nums, i, left++);
            } else if (nums[i] == 2 && i < right) {
                swap(nums, i, right--);
            } else {
                i++;    // i == 1  || i=0 & i=left
            }
        }
    }
    
    
    /** Sort Colors II - 假设有k个color 1,2...,k
     * @param colors
     * @param k
     * 用counting sort..two pass  (上面那题也能这么做，但是慢一点)
     * 1. 用count[k]记下每种颜色出现几次
     * 2. 依次把count放进原数组里
     * 
     * 时间复杂度 O(n), 空间是O(k)
     */
    public void sortColorsII(int[] colors, int k) {
        int[] count = new int[k];		//多少种颜色
        for (int color : colors) {
            count[color - 1]++;
        }
        
        int idx = 0;
        for (int i = 0; i < k; i++) {
            while (count[i] > 0) {
                colors[idx] = i + 1;        //因为从1开始，不是0
                count[i]--;
                idx++;
            }
        }
    }
    
    
    /** Sort Colors II - k种颜色
     * @param colors
     * @param k
     * 跟sort colors的双指针很像..
     * 重点是..先把min和max颜色排在最前面和最后面.. 然后中间的再for循环继续排
     * 所以每次找这个区间的min, max把它们排在头尾.. 排完后 count+=2，因为排好了2个颜色
     */
    public void sortColorsII2(int[] colors, int k) {
        int count = 0;
        int start = 0, end = colors.length - 1;
        
        //count是color的个数，每次找最大和最小的color来分到左右两边
        while (count < k) {         
            int min = k + 1;
            int max = -1;
            for (int m = start; m <= end; m++) {
                min = Math.min(min, colors[m]);
                max = Math.max(max, colors[m]);
            }

        // ******开始 3-way-partition模板
            int i = start;
            int j = end;
            int cur = start;
            while (cur <= j) {
                if (colors[cur] == min) {
                    swap(colors, i, cur);
                    i++;
                    cur++;
                } else if (colors[cur] == max) {
                    swap(colors, j, cur);
                    j--;
                } else {
                    cur++;
                }
            }
        // *************模板结束

            start = i;      //更新 start和end
            end = j;
            count += 2;     //每次排完2种color
        }
    }
    
    
    private void swap(int[] A, int a, int b) {
        int tmp = A[a];
        A[a] = A[b];
        A[b] = tmp;
    }
    
    
    
    /** Partition Array
     * 分区，使得所有 < k的数都在左边， >= k的都在右边，return第一个 >=k的index
     * @param nums
     * @param k
     * @return
     */
    public int partitionArray(int[] nums, int k) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            while (left <= right && nums[left] < k) {
                left++;
            }
            while (left <= right && nums[right] >= k) {
                right--;
            }
            if (left <= right) {
                swap(nums, left, right);
                left++;
                right--;
            }
        }
        return left;
    }
    
    
    /** 280. Wiggle Sort
     * nums[0] <= nums[1] >= nums[2] <= nums[3]...
     * @param nums
     * 由于邻居可以相等，所以可以直接n[i]跟[i+1]换
     * 
     * 当i为奇数时，nums[i] >= nums[i + 1]
     * 当i为偶数时，nums[i] <= nums[i + 1]
     * so, 当不满足上面，就要swap
     */
    public void wiggleSort(int[] nums) {
        for (int i = 0; i < nums.length - 1; i++) {
        	// 或直接 if ((i % 2 == 0) == (nums[i] > nums[i+1])) {swap}. 包含了都为false的情况
            if (((i % 2 == 0) && nums[i] > nums[i+1]) || ((i % 2 != 0) && nums[i] < nums[i+1])) {
                swap(nums, i, i+1);
            }
        }
        
        //**** 先sort再前后换.. 慢
//        Arrays.sort(nums);
//        for (int i = 1; i < nums.length - 1; i += 2) {		// i += 2
//            swap(nums, i, i + 1);
//        }
    }
    
    
    /** 324. Wiggle Sort II
     * reorder it such that nums[0] < nums[1] > nums[2] < nums[3]. 
     * 可能有相等duplicate的数，所以邻居不能相等 (不能跟 I 一样直接i跟i+1换)
     * @param nums
     * 先sort，找mid, 分成前后2半段.
     * 然后先从 前半段取最后的(也是最大)放nums[]里，然后再从 后半段的末尾取来放[]里
     * 这样偶数都是 < 奇数，因为后半段都大于
     * 
     * Q: 为何不直接 start和end向中间靠拢？
     * A: 因为这样的话最后中间的值可能都一样，无法区分..
     * 比如1,2,3,3,3,5. 如果两边往中间靠，那就是1,5,2,3,3,3..就错了
     * 所以要从中间mid分开，交错放，这样不会错
     */
    public void wiggleSortII(int[] nums) {
        int n = nums.length;
        
        int[] sorted = Arrays.copyOf(nums, n);		// copy一个数组让他sorted
        Arrays.sort(sorted);
        
        int mid = n % 2 == 0 ? n/2 - 1 : n/2;		//mid靠前放，之后从mid开始放
        int end = n - 1;
        for (int i = 0; i < n; i++) {	//偶数时放前半段小的，odd放大的
            nums[i] = i % 2 == 0 ? sorted[mid--] : sorted[end--];
        }
    }
    
    
    /** O(n). 先用findKthElement找median，再跟上面差不多，奇偶这么放
     * @param nums
     * https://leetcode.com/problems/wiggle-sort-ii/discuss/77684/summary-of-the-various-solutions-to-wiggle-sort-for-your-reference
     */
    public void wiggleSortII2(int[] nums) {
        int n = nums.length;
        int median = findKthElement(nums, n / 2);
        int odd = 1;
        int even = n % 2 == 0 ? n-2 : n-1;		//为何
        int[] tmp = new int[n];

        for (int i = 0; i < n; i++) {
            if (nums[i] > median) {
                tmp[odd] = nums[i];
                odd += 2;
            } else if (nums[i] < median) {
                tmp[even] = nums[i];
                even -= 2;
            }
        }
        while (odd < n) {    //只能while
            tmp[odd] = median;
            odd += 2;
        }
        while (even >= 0) {   //要while
            tmp[even] = median;
            even -= 2;
        }

        // 最后返回input的nums里
        for (int i = 0; i < n; i++) {
            nums[i] = tmp[i];
        }        
    }
    
    
    /** 用mapping来表示新的index。之后再用sort colors的3-way-partition做
     * @param nums
     */
    public void wiggleSortII3(int[] nums) {
        int n = nums.length;
        int median = findKthElement(nums, n / 2);
        
        int left = 0;
        int i = 0;
        int right = n - 1;
        while (i <= right) {
            if (nums[newIndex(i, n)] > median) {                    // 大于 median的 放左边  -> 奇数
                swap(nums, newIndex(left, n), newIndex(i, n));
                left++;
                i++;                                                // 中间是 median
            } else if (nums[newIndex(i, n)] < median) {
                swap(nums, newIndex(right, n), newIndex(i, n));     // 小于 median 放右边   -> 偶数
                right--;
            } else {
                i++;
            }
        }
    }
    
    /**
     * Create an index mapping.
     * [0, 1, 2, 3, 4, 5] -> [1, 3, 5, 0, 2, 4]
     * [0, 1, 2, 3, 4, 5, 6] -> [1, 3, 5, 0, 2, 4, 6]
     */
    private int newIndex(int i, int n) {
        return (2 * i + 1) % (n | 1);  
        // (n | 1) calculates the nearest odd >= n.  | 表示bit manipulation里的 OR.  0 | 1 = 1 ; 1 | 1 = 1 ;
        // 所以 n | 1 就是 跟1 or.. 可以找到最近 大于n的odd，也可以是自己
    }
    
    public int findKthElement(int[] nums, int k) {
        int start = 0, end = nums.length - 1;
        while (true) {
            int pos = partition(nums, start, end);
            if (pos == k) {
                return nums[pos];
            } else if (pos > k) {
                end = pos - 1;
            } else {
                start = pos + 1;
            }
        }
    }
    
    public int partition(int[] nums, int i, int j) {
        int start = i;
        int pivot = nums[start];
        while (i <= j) {
            while (i <= j && nums[i] <= pivot) {
                i++;
            }
            while (i <= j && nums[j] > pivot) {
                j--;
            }
            if (i <= j) {
                swap(nums, i, j);
            }
        }
        // j stops when n <= pivot, and j-1, so j should at the last smaller one. need to swap with start
        swap(nums, start, j);   // j is the smallest of right part
        return j;
    }
    
    
    

    /**
     * 209. Minimum Size Subarray Sum  - 都是正数
     * 找subarry sum >= s 的最小长度。给[2,3,1,2,4,3] and s = 7,结果是[4,3]返回2.  这里没有负数
     * @param s
     * @param nums
     * @return
     * 因为算sum >= s, 不能用HashMap，所以只能 Two Pointer (i & j)
     * 
     * 1. j在前面走，当sum >= s时，加到ans里
     * 2. 去掉前面的i看看结果是否>=s。因为可以n[j]够大，i可以++ 
     */
    public int minSubArrayLen(int s, int[] nums) {
        if (nums == null || nums.length == 0)   return 0;
        
        int ans = Integer.MAX_VALUE;
        int sum = 0;
        int i = 0;
        
        //************** 模板*************************			
        for (int j = 0; j < nums.length; j++) {	//也可以while{ sum += n[j++]
            sum += nums[j];
            while (sum >= s) {
                ans = Math.min(ans, j - i + 1);
                sum -= nums[i];
                i++;
            }
        }
        
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }


    /**
     * 209. Minimum Size Subarray Sum
     *
     * binary search 方法.. 慢
     *
     * 主要用 prefix sum的做法.. sums[j] - sums[i] >= target的话就OK
     *
     * 因为都是正数，所以sums[]是递增的, sorted. 可以用binary search搜索
     * 如果找到sums[i] + target的地方，就证明找到 end (j) 的位置
     */
    public int minSubArrayLenBinarySearch(int target, int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;

        int len = nums.length;
        int[] sums = new int[len + 1];

        for (int i = 1; i <= len; i++) {
            sums[i] = sums[i - 1] + nums[i - 1];
        }

        int min = len + 1;

        for (int i = 0; i <= len; i++) {
            int end = binarySearch(i + 1, len, sums[i] + target, sums);
            if (end == len + 1)
                break;
            if (end - i < min) {
                min = end - i;
            }
        }
        return min == len + 1 ? 0 : min;
    }

    private int binarySearch(int lo, int hi, int target, int[] sums) {
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (sums[mid] >= target) {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }


    /**
     * 862. Shortest Subarray with Sum at Least K - 有可能负数
     * 找 sum >= K 的最短subarray.. 跟上题一样，只是有可能negative负数
     * @param A
     * @param target
     * @return
     * 要考虑prefix sum的做法.. sums[j] - sums[i] >= target的话就OK
     *
     * 需要单调递增increasing的Dequeu来保存index.
     *
     * PS: 239. Sliding Window Maximum  跟这题有点像，也是用单调递减deque
     */
    public int shortestSubarray(int[] A, int target) {
        if (A == null || A.length == 0)
            return 0;

        int len = A.length;
        int min = len + 1;
        int[] sums = new int[len + 1];

        for (int i = 1; i <= len; i++) {
            sums[i] = sums[i - 1] + A[i - 1];
        }

        Deque<Integer> dq = new ArrayDeque<>();     // put index

        for (int i = 0; i <= len; i++) {
            // 找到了
            while (dq.size() > 0 && sums[i] - sums[dq.getFirst()] >= target) {
                min = Math.min(min, i - dq.pollFirst());
            }
            while (dq.size() > 0 && sums[i] <= sums[dq.getLast()]) {
                dq.pollLast();          // keep deque increasing 否则decrease了不会是答案
            }

            // put index in deque
            dq.addLast(i);
        }
        return min == len + 1 ? -1 : min;
    }



    /** 76. Minimum Window Substring 
     * Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).
     * S = "ADOBECODEBANC". T = "ABC". 结果是"BANC"
     * @param s
     * @param t
     * @return
     * 只用一个HashMap（数组），第一次for T时 m[k]++, 之后扫描S时，再-- 
     * while(count == tlen)时, 前面的i++往后移  
     */
    public String minWindow(String s, String t) {
        String res = "";
        int[] map = new int[128];   //store t's hashmap
        int tlen = t.length();
        
        for (int k = 0; k < tlen; k++) {		//放t的freq
            map[t.charAt(k)]++;
        }
        
        int count = 0;
        int minLen = s.length() + 1;
        int i = 0;          // start
        for (int j = 0; j < s.length(); j++) {   // j < s.len, run faster, 是end指针
            if (map[s.charAt(j)] > 0) {                //j exist in T.
                count++;
            }
            map[s.charAt(j)]--;
            
            while (count == tlen) {			//在这因为无法判断是否map.contains，所以用while来循环让i++
                if (minLen > j - i + 1) {
                    minLen = j - i + 1;				//也可以只用result (result == "" || result.length() > j - i + 1) {
                    res = s.substring(i, j + 1);
                }

                map[s.charAt(i)]++;     //decrese freq in source，if char not exists那就是-1变成0
                if (map[s.charAt(i)] > 0)    //t需要, count--。t不需要的顶多变0，所以记得不是>=0
                    count--;
                i++;
            }
            
        }
        return res;
    }
    
    
    /** 76. Minimum Window Substring
     *
     * 用一个hashMap存T(toFind)，另一个存S 里匹配的(found).  --- 用数组存更快
     * 当找到符合所有T时，就开始看i能否减小window / 加结果
     * 若不valid(count变小), 就j++往前直到valid，接着再i++直到不valid
     * 
     * 1. 存T用到char, freq的map
     * 2. 双指针i, j 扫描S. 都是j 走得快
     * 3. 找到匹配的char就更新found[]，且count++ 直到count==T长度
     * 4. 符合条件后就要判断start i 能否缩小了
     * 	 4.1 while 不包含 OR 过了freq，start就++往后走来缩小。（若超过freq记得更新--）
     * 	 4.2 看长度，加到result里
     * 
     * can use 1 hashmap.. 看后面
     */
    public String minWindowHashMap(String s, String t) {
        if (s == null || s.length() == 0 || t == null || t.length() == 0 || s.length() < t.length())   return "";
        
        String res = "";
        int[] map = new int[128];   //store t's hashmap
        int[] found = new int[128];
        int tlen = t.length();
        
        for (int k = 0; k < tlen; k++) {
            map[t.charAt(k)]++;
        }
        
        int count = 0;
        for (int i = 0, j = 0; j < s.length(); j++) {   // j < s.len, run faster
            if (map[s.charAt(j)] == 0)  //not in t
                continue;
            
            // map包含, so found[]++ 更新freq, then compare.
            if (++found[s.charAt(j)] <= map[s.charAt(j)])
                count++;        //only when this char enough count++, but the found[] could have more freq
            
            // 找到1st result后下次会再循环，而且cnt都=Tlen 能直接进来比较
            if (count == tlen) {     
                // 这里都是看start i， 跟e没什么关系了
            	// try to see if start can ++ to minimize window, when i not needed OR i has more freq than require
                while (found[s.charAt(i)] == 0 || found[s.charAt(i)] > map[s.charAt(i)]) {
                    if (found[s.charAt(i)] > map[s.charAt(i)]) {
                        found[s.charAt(i)]--;
                    }
                    i++;
                }
                if (res == "" || res.length() > j - i + 1) {
                    res = s.substring(i, j + 1);
                }
            }
        }
        return res;
    }


    /**
     * 727. Minimum Window Subsequence - two pointer sliding window
     * subsequence的顺序要一样.. 不像 minimum window substring 可以不同顺序
     * S = "abcdebdde", T = "bde"
     * Output: "bcde"
     *
     * 一旦match了，i, j开始往前缩短window，看是否能找到最小的minLen
     */
    public String minWindowSubsequence(String S, String T) {
        int sLen = S.length();
        int tLen = T.length();
        int start = -1;
        int minLen = Integer.MAX_VALUE;

        for (int i = 0, j = 0; i < sLen; i++) {
            if (S.charAt(i) == T.charAt(j)) {
                j++;
            }

            // match了
            if (j == tLen) {
                int end = i + 1;

                // 开始往前缩短window
                while (--j >= 0) {
                    while (S.charAt(i) != T.charAt(j)) {    // 不同就i--
                        i--;
                    }
                    i--;        // 正常是 i & j 都 -- 往前
                }

                i++;        // 减过头了加回来
                j++;

                if (end - i < minLen) {
                    minLen = end - i;
                    start = i;
                }
            }
        }
        return start == -1 ? "" : S.substring(start, start + minLen);
    }
    
    
    /** 3. Longest Substring Without Repeating Characters
	 * Given a string, find the length of the longest substring without repeating characters. 
	 * For example, the longest substring without repeating letters for "abcabcbb" is "abc", which the length is 3. 
	 * For "bbbbb" the longest substring is "b", with the length of 1.
	 * @param s
	 * @return
	 * 每次就先map[s(j)]++
	 * 如果 == 1 不重复，就算maxLen
	 * 	   > 1 有重复，那就把左边的i往后移，直到没有重复的m[s(j)]为止
	 * 
	 * 比较像模板
	 */
	public int lengthOfLongestSubstring(String s) {
		int[] map = new int[128];
        int maxLen = 0;
        int i = 0;
        for (int j = 0; j < s.length(); j++) {
            map[s.charAt(j)]++;					// ************模板*********************
            
            if (map[s.charAt(j)] == 1) {
                maxLen = Math.max(maxLen, j - i + 1);
            }
            while (map[s.charAt(j)] > 1) {      //重复了, while直到现在不重复
                map[s.charAt(i)]--;
                i++;
            }
        }
        return maxLen;
    }
	
	
	/** 3. Longest Substring Without Repeating Characters  - Better solution	
	 * int[26] for Letters 'a' - 'z' or 'A' - 'Z'
		int[128] for ASCII
		int[256] for Extended ASCII	
	 * @param s
	 * @return
	 * 这个比上面更快，不需要i一步步往后，而是直接看max(i, arr[s.charAt(j)])。
	 * 这样就知道之前这个s[j]出现在什么地方了，那么跳过它，后面的数开始就不会重复，可以算j-i+1的maxLen
	 * 
	 * arr[s.charAt(j)]放index位置，如果为0，说明没出现过，否则 >0就重复，所以i就直接跳到重复以后的地方
	 */
	public int lengthOfLongestSubstringBetter(String s) {
		
		Map<Character, Integer> map = new HashMap<Character, Integer>();	
		
        int max = 0;
        // extend the range [i, j)
        for (int i = 0, j = 0; j < s.length(); j++) {
            if (map.containsKey(s.charAt(j))) {
            	i = Math.max(map.get(s.charAt(j)), i);	//判断i最大在哪,记得用max。 如果是"tmmzuxt" 最后的t就跳过了，不能else
            } 
            max = Math.max(max, j - i + 1);
            map.put(s.charAt(j), j + 1);		//这次j是重复，所以要放j+1就不会重复                       
        }
        
        //=========用数组代替HashMap=========
        // 注意取i时，不能用arr[s.charAt(j)] + 1， 因为arr[s.charAt(j)]初始都是0，包括不存在的，+1就选它会错
        int[] arr = new int[128];
        for (int i = 0, j = 0; j < s.length(); j++) {
            i = Math.max(i, arr[s.charAt(j)]);			//若取arr[s[j]], 说明有重复。
            max = Math.max(max, j - i + 1);
            arr[s.charAt(j)] = j + 1;			//放的时候记得 + 1，这样跳过重复的   
        }					//比如abcabc,第一个a的位置arr[a]=1,表示到b。这样到第二个a时，j=3,结果3-1+1=3
        
        return max;
	}
	
	

    /** 340. Longest Substring with At Most K Distinct Characters
     * Given s = “eceba” and k = 2, return "ece" which its length is 3.
     * @param s
     * @param k
     * @return
     * 双指针 sliding window
     * 1. 直接把freq[s.charAt(j)]++;
	 * 2. 如果 count > k，需要start i往后挪直到能去掉map的一个size(count--)
	 * 3. 每次 j 挪后求result (不要在while n > k里面加result，否则会错过一些count<k 但是j到len结束的情况)
     */
    public int lengthOfLongestSubstringKDistinctBest(String s, int k) {
        int max = 0;
        int i = 0;
        int[] freq = new int[128];
        int count = 0;    //用num来代替HashMap.size
        
        for (int j = 0; j < s.length(); j++) {
        	freq[s.charAt(j)]++;
            if (freq[s.charAt(j)] == 1) {
                count++;          //when new ch
            }
            
            // 这里算count > k，而不是 ==k, 否则会算得短
            while (count > k) {   // need to remove start i to let num = k, 所以要用while
                freq[s.charAt(i)]--;
                if (freq[s.charAt(i)] == 0) {		//如果还 >0,证明没删完多出来的distinct,所以继续while
                    count--;
                }
                i++;
            }
            max = Math.max(max, j - i + 1);
        }
        return max;
        
        
        //===================hashmap版本================
        /*
        HashMap<Character, Integer> map = new HashMap<>();
        
        for (; j < s.length(); j++) {
            c = s.charAt(j);
            map.put(c, map.getOrDefault(c, 0) + 1);
            
            while (map.size() > k) {
                // try to remove i
                c = s.charAt(i);
                if (map.containsKey(c)) {   //记得检查，否则会NullPointerEx
                    map.put(c, map.get(c) - 1);
                    if (map.get(c) == 0) {
                        map.remove(c);
                    }
                }
                i++;
            }
            max = Math.max(max, j - i + 1);
        }
        
        return max;
        */
    }
        
    
    

    /** 438. Find All Anagrams in a String
     * 找string s 里面跟p一样anagram的index
     * s: "cbaebabacd" p: "abc"，返回[0, 6]. 因为"cba"和"bac"
     * @param s
     * @param p
     * @return
     * 也是跟min window差不多，用sliding window来做。
     * 这里的window是固定plen, 所以每次都要判断j - i + 1== plen时,前面i的count要--
     * 
     * 1. 先arr[j]--, 看是否match, 是就count++
     * 2. if count==plen, 符合条件，就加到list里
     * 3. left i 要往后走，Reset
     * 		记得当j - i + 1== plen时，left i就要往前移，并且arr[i]++要Reset
     */
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> list = new ArrayList<>();
        int[] arr = new int[128];			//也可以26, 然后arr[c-'a']这样
        for (char c : p.toCharArray()) {
            arr[c]++;
        }
        
        int plen = p.length();
        int i = 0, count = 0;
        for (int j = 0; j < s.length(); j++) {
            arr[s.charAt(j)]--;
            if (arr[s.charAt(j)] >= 0) { 		//match p
                count++;
            }
            
            if (count == plen) {        //found
                list.add(i);
            }
            
          //sliding window,要减掉left i这样才能保证固定的window大小
          // so if出现了多余的单词(p没有的), 也会到这里判断，因为窗口大小达到plen, 所以要往右移 
            if (j - i + 1 == plen) {        
            	arr[s.charAt(i)]++;     //加回来reset
                if (arr[s.charAt(i)] > 0) {    //char在p里才需要count--
                    count--;
                }
                i++;                    //left右移
            }
        }
        return list;
    }
    

    
    /** 30. Substring with Concatenation of All Words 
     * You are given a string, S, and a list of words, L, that are all of the same length.
     * Find all starting indices of substring(s) in S that is a concatenation of each word in L exactly once and without any intervening characters.
     * S: "barfoothefoobarman" L: ["foo", "bar"] 返回[0,9] 。 
     * 字典的单词长度一样
     *
     * 最简单的做法
     * i表示起点，j表示dict的第几个单词（因为单词长度一样可以算）
     */
    public ArrayList<Integer> findSubstring(String S, String[] dict) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        HashMap<String, Integer> map = new HashMap<String, Integer>();
        HashMap<String, Integer> found = new HashMap<String, Integer>();
        
        for (String str : dict) {
            if (map.containsKey(str))
            	map.put(str, map.get(str) + 1);
            else
            	map.put(str, 1);
        }
        
        int wlen = dict[0].length();
        int dictNum = dict.length;	// i <= ...比如"a" 字典里是a
        for (int i = 0; i <= S.length() - wlen * dictNum; i++) {
        	found.clear();
        	int j;
        	for (j = 0; j < dictNum; j ++) {	//每次j只扫dict[]的长度
        		int k = i + j * wlen;		// j表示第几个单词，i表示起始位置
        		String sub = S.substring(k, k + wlen);
        		if(!map.containsKey(sub))
        			break;
        		
        		if (found.containsKey(sub)) 
        			found.put(sub, found.get(sub) + 1);
        		else
        			found.put(sub, 1);
        		
        		if (found.get(sub) > map.get(sub))
        			break;						// 超出就不算
        	} 
        	if (j == dictNum) {
        		result.add(i);
        	}
        }
        return result;
    }

    
    /** Substring with Concatenation of All Words - 双指针  更快 O(wl * s.len)
     * 跟上面的双指针差不多。
     * 不一定第一个字母就可以符合，可能出现下面的情况
     * {"abbababba", ["ab", "ab", "ba", "bb"]} 从"ab", "ba"开始 and meet "ba" again。但之后就不行了
     * 所以从第二个char 'b'开始，就是"bb", "ab", "ab", "ba". result should return 1. 。
     * 所以最外层需要 for i循环 < wlen
     */
    public ArrayList<Integer> findSubstring2(String s, String[] words) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        HashMap<String, Integer> map = new HashMap<>();
        HashMap<String, Integer> found = new HashMap<>();
        
        for (String w : words) {
      //      map.put(w, map.getOrDefault(w, 0) + 1);		//新版java可以这样
        	if (map.containsKey(w))
            	map.put(w, map.get(w) + 1);
            else
            	map.put(w, 1);
        }
        
        int wl = words[0].length();
        int n = words.length;
        int count = 0;
        String tmp = "";
        int left = 0;
     // maybe the 1st char is not part of the word, so just check word length times
       //或者有可能都符合..比如aaaaaaaa, 字典里有3个"aa". 如果外层i不一个个++，直接j+wl,那就错过了
        for (int i = 0; i < wl; i++) {
            left = i;
            count = 0;
            found.clear();
            for (int j = i; j <= s.length() - wl; j += wl) {
                String str = s.substring(j, j + wl);
                if (map.containsKey(str)) {
               //     found.put(str, found.getOrDefault(str, 0) + 1);
                    if (found.containsKey(str))		found.put(str, found.get(str) + 1);
                    else			            	found.put(str, 1);

                    int freq = map.get(str);
                    if (found.get(str) <= freq) {
                        count++;
                    } else {		
                        while (found.get(str) > freq) {     //当freq 大于需要的，left就要 循环 往后移缩小window，直到最外层的str的freq正常
                            tmp = s.substring(left, left + wl);
                            found.put(tmp, found.get(tmp) - 1);
                            if (found.get(tmp) < map.get(tmp)) {
                                count--;
                            }
                            left += wl;
                        }
                    }
                    // check result
                    if (count == n) {
                        result.add(left);
                        tmp = s.substring(left, left + wl);
                        found.put(tmp, found.get(tmp) - 1);     //left 往前移一个Word
                        left += wl;
                        count--;
                    }
                } else {            // if map not contain word
                    count = 0;
                    found.clear();
                    left = j + wl;		//因为j会+wl, 所以i也要
                }
            }
        }
        return result;
    }


    /**
     * 845. Longest Mountain in Array
     * mountain是 先递增，再递减.. 找到longest的
     * @param A
     * @return
     *
     * 简单的用two pointer, start & end..
     * 用一个peak 来确定 peak < j 保证后面有down的情况 才算valid可以算max
     */
    public int longestMountain(int[] A) {
        int len = A.length;
        int max = 0;
        int j = 1;      // end

        while (j < len) {
            if (A[j - 1] < A[j]) {
                int i = j - 1;          // start

                // up
                while (j < len && A[j - 1] < A[j]) {
                    j++;
                }
                // down
                int peak = j;
                while (j < len && A[j - 1] > A[j]) {
                    j++;
                }
                if (peak < j) {                         // 找到才会算max
                    max = Math.max(max, j - i);
                }
            } else {        // is not valid mountain (up), keep going to find the up
                j++;
            }
        }
        return max;
    }
    
    
    
    public static void main(String[] args) {
    	TwoPointSol sol = new TwoPointSol();
    	List<List<Integer>> list = sol.kSumGeneral(new int[]{1, 0, -1, 0, -2, 2}, 0, 4);
    	System.out.println(list.size());
    	for (List<Integer> l : list) {
    		System.out.println(l.toString());
    	}
    	
    	List<List<Integer>> list1 = sol.kSumGeneral(new int[]{1, 2, 3, 4,5,6,7}, 7, 3);
    	System.out.println(list1.size());
    	System.out.println(sol.kSumDP(new int[]{1, 2, 3, 4,5,6,7}, 3, 7));
    }
}



