import java.util.*;



public class BinarySearch {

	/** first position of target in sorted array (has duplicats)
     * @param nums: The integer array.
     * @param target: Target to find.
     * @return: The first position of target. Position starts from 0.
     */
    public int binarySearch(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        
        int start = 0, end = nums.length - 1;
        while (start < end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] == target) {
                end = mid;					    //如果找last就是 start = mid
            } else if (nums[mid] < target) {
                start = mid;
                                                // or start = mid + 1
            } else {
                end = mid;				// 可以跟上面的==合并
                                                // or end = mid - 1
            }
        }
        return start;
    }
    
    
    /** 35. Search Insert Position
     * 找到target就返回index，否则返回需要插入的位置
     * [1,3,5,6], 2 → 1
     * 跟Arrays.binarySearch一样
	 */
    public int searchInsert(int[] nums, int target) {
        int lo = 0, hi = nums.length - 1;

        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return lo;      // 不能hi
    }


    public int searchInsert1(int[] nums, int target) {
        int low = 0;
        int high = nums.length;     // len !!!  not len - 1

        while (low < high) {
            int mid = low + (high - low) / 2;
            if(nums[mid] < target)
                low = mid + 1;
            else
                high = mid;
        }
        return low;     // 也可用high
    }


    /**
     * 374. Guess Number Higher or Lower
     * -1, 你的lower,  0 对了， 1 你的太高了.. 返回最终猜对的数
     * @param n
     * @return
     */
    public int guessNumber(int n) {
        int start = 1, end = n;

        while (start <= end) {
            int mid = start + (end - start) / 2;
            if (guess(mid) == 0) {
                return mid;
            } else if (guess(mid) < 0) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
        return -1;
    }

    private int guess(int num) {
        return 0;
    }
    
    
    /** 278. First Bad Version
     * 一个bad的话，后面全bad。要找第一个bad的地方。。算是重复的找first出现
     * @param n
     * @return
     */
    public int firstBadVersion(int n) {
        int start = 1, end = n;
        
        while (start < end) {
            int mid = start + (end - start) / 2;
            if (isBadVersion(mid)) {
                end = mid;
            } else {					
                start = mid + 1;
            }
        }
        return start;
    }
    
    public boolean isBadVersion(int key) {
    	return true;
    }


    /**
     * 744. Find Smallest Letter Greater Than Target - easy
     * 找到第一个大于target的char
     * Letters also wrap around.
     * For example, if the target is target = 'z' and letters = ['a', 'b'], the answer is 'a'.
     *
     * 所以最后result  lo % n
     *
     * Naive O(n) 方法就是
     *          for (char c: letters)
     *             if (c > target) return c;
     *         return letters[0];
     */
    public char nextGreatestLetter(char[] letters, char target) {
        int start = 0, end = letters.length;

        while (start < end) {
            int mid = start + (end - start) / 2;
            if (letters[mid] > target) {
                end = mid;
            } else {
                start = mid + 1;
            }
        }
        return letters[start % letters.length];     // 也可以是end. 一样的
    }
    
    /** 34. Search for a Range	- better binary search两次 - Find First and Last Position of Element in Sorted Array
	 * [5, 7, 7, 8, 8, 10] and target value 8. return [3, 4]
     * @param nums
     * @param target
     * @return
     * 其实是2道题.. 找first position + 找last position
     * 记住 模板！！！
     * 
     * 这里不用另外的index，因为如果没找到，就要返回-1.。而非index。所以最后要比较n[start] == target
     * 
     * !!!记得 找last position时，mid要+1往右取，否则死循环，因为要找最右的位置
     */
    public int[] searchRange(int[] nums, int target) {
        int[] arr = {-1, -1};
        int s = 0, e = nums.length - 1;
        
        // 找first position
        while (s < e) {							 // s < e
            int mid = s + (e - s) / 2;
            if (nums[mid] >= target) {
                e = mid;
            } else {
                s = mid + 1;
            }
        }
        if (nums[s] != target) {     // 可以是 end 一样
            return arr;                 // not found, faster
        } else {
            arr[0] = s;
        }
        
        // 找最后的last position
        e = nums.length - 1;
        while (s < e) {
            int mid = s + (e - s) / 2 + 1;      //记得mid要+1， 往右取，否则死循环，因为要找最右的位置
            if (nums[mid] <= target) {
                s = mid;
            } else {
                e = mid - 1;
            }
        }
        arr[1] = e;     // not need to see if == 这里可以start, 或者end
        return arr;
    }


    /**
     * 34. Search for a Range - Find First and Last Position of Element in Sorted Array
     * 用同一个search方法
     * @param nums
     * @param target
     * @return
     */
    public int[] searchRange1(int[] nums, int target) {
        int[] result = {-1, -1};

        int leftIdx = extremeSearch(nums, target, true);
        if (leftIdx == nums.length || nums[leftIdx] != target)
            return result;

        result[0] = leftIdx;
        result[1] = extremeSearch(nums, target, false) - 1;     // !!! -1

        return result;
    }

    private int extremeSearch(int[] nums, int target, boolean left) {
        int lo = 0;
        int hi = nums.length;   // 是len, 因为总体往后挪了一位

        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (nums[mid] > target || (left && target == nums[mid])) {
                hi = mid;
            }
            else {
                lo = mid+1;
            }
        }

        return lo;
    }
    
    
    // solution里有描述
    public int hIndexIIBS(int[] citations) {
        int n = citations.length;
        if (n == 0)     return 0;
        int low = 0, hi = n - 1;
        while (low <= hi) {
            int mid = low + (hi - low) / 2;
            if (citations[mid] == n - mid) {
                return citations[mid];
            } else if (citations[mid] < n - mid) {
                low = mid + 1;          // ok,但有更多被引用的
            } else {
                hi = mid - 1;
            }
        }
        return n - low;
    }
   

 
    /** 153. Find Minimum in Rotated Sorted Array 
     * 中间某个数可能是最小的，如4 5 6 7 0 1 2
     * @param nums
     * @return
     * 只用看n[mid]和后面的数比较
     */
    public int findMin(int[] nums) {
        int start = 0, end = nums.length - 1;
        
        while (start < end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] < nums[end]) {          //这段是递增，那就往左找
                end = mid;						//需要end = mid, 有可能mid就是最小
            } else {
                start = mid + 1;
            }
        }
        return nums[start];         // 也可以是 n[end]
    }
	
	
	/** 154. Find Minimum in Rotated Sorted Array II - 可能contains duplicate numbers
	 * @param nums
	 * @return
	 * 这里的区别是，n[mid]==n[e]时要e--, 且之后n[mid]要跟n[e]比较
	 */
	public int findMinRotatedII(int[] nums) {
        int s = 0, e = nums.length - 1;
        
        while (s < e) {
            int mid = s + (e - s) / 2;
            if (nums[mid] < nums[e]) {
                e = mid;
            } else if (nums[mid] > nums[e]) {
                s = mid + 1;
            } else {    // when n[mid] == n[e], e--跳过重复的
                e--;
            }
        }
        
        return nums[s];
    }
	
	
	/**
     * 33. & 81. Search in Rotated Sorted Array I & II
	 * 4 5 6 7 0 1 2
	 *
	 * II 当允许duplicate时，返回Boolean看是否能找到
	 * else if (A[start]==A[mid]) {start++. } 往后移忽略重复的
	 * 或者如果是A[mid]==A[end], 那就end--   
	 */
	public int search(int[] nums, int target) {
        int start = 0;
        int end = nums.length - 1;

     // 也可以 while(s + 1 < e) { s=mid; e=mid}  最后判断nums[s] == target 后还要判断 n[end]==target

        while (start <= end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] == target) {
                return mid;
            }

            // II 如果有 duplicate的话， == 单独拿出来
//            if (nums[start] == nums[mid]) {         // n[start] == n[mid] duplicate
//                start++;
            if (nums[start] <= nums[mid]) {     // 记住 <=  left monotonically increasing, first bigger part
                if (nums[start] <= target && target < nums[mid]) {
                    end = mid - 1;
                } else {
                    start = mid + 1;
                }
            } else {            // right monotonically increasing, right smaller
                if (nums[mid] < target && target <= nums[end]) {
                    start = mid + 1;
                } else {
                    end = mid - 1;
                }
            }
        }
        
        return -1;
        
    }
	
	
	
	
	/** Search a 2D Matrix 
	 * 数组升序排列，每行从小到大，并比前一行所有都大
	 * @param matrix
	 * @param target
	 * @return
	 * 可以拉成一位数组，用二分法
	 */
	public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        int row = matrix.length;
        int cols = matrix[0].length;
        int start = 0;
        int end = row * cols - 1;				// 变一维数组
        int mid;
        
        while (start <= end) {
            mid = start + (end - start) / 2;
            int ele = matrix[mid / cols][mid % cols];
            if (ele == target) {
                return true;
            } else if (ele < target) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        return false;
    }
	
	
	
	/** Search a 2D Matrix II
	 * 每行 或 每列都是升序的。但不保证后一行大于前一行所有元素
	 * @param matrix
	 * @param target
	 * 比较不同列（竖）：如果ele > target，则在列的左边。假设row一样，后>前
	 * .......行（横）：    ele > target，则在行的下面。 因为每列上<下
	 * 所以从 右上角 OR 左下角 开始比。一头一尾才比得出
	 */
	public boolean searchMatrixIIa(int[][] matrix, int target) {
		int row = 0;
		int col = matrix[0].length - 1;				// 右上角 开始
		while (row < matrix.length && col >= 0) {
			if (matrix[row][col] == target) {
				return true;
			} else if (matrix[row][col] > target) {
				col--;				// 从最后一列开始，所以只用往左移 缩小范围
			} else {
				row++;
			}
		}
		return false;
	}
	
	
	/** 378. Kth Smallest Element in a Sorted Matrix
     * 用二分法，找第K个元素<=mid
     * @param matrix
     * @param k
     * @return
     * 因为每行每列都有序，那么for(每行), 看有多少count个数是 <= mid的.  那么就while(j列的matrix[i][j] > mid) j--来找
     * 算完所有行所有列以后，这次二分以mid来找的 count就是 前count个smallest数
     * 然后count 和 k 比较，如果不到所需的k个smallest，那就low=mid+1，增加查找的范围
     * 
     * 时间复杂度 m * log(hi-lo)
     */
    public int kthSmallestBS(int[][] matrix, int k) {
    	int m = matrix.length;
        int n = matrix[0].length;
        
        int low = matrix[0][0];
        int high = matrix[m-1][n-1];
        while (low < high) {
            int mid = (low + high) / 2;
            int count = 0;
            int j = n - 1;
            for (int i = 0; i < m; i++) {       //对于每行，找到 <= mid的个数
                while (j >= 0 && matrix[i][j] > mid)
                    j--;
                count += j + 1;         //这行有多少个
            }
            
            // 这轮binary search的mid结束后
            if (count < k)  low = mid + 1;			//注意这里，即使count==k也不能返回mid,因为mid很可能不存在
            else    high = mid;
        }
        return low;
    }
	
	
	/** 69. Sqrt(x)  用二分法求平方根
	 */
	public int mySqrt(int x) {
        int lo = 1, hi = x;
        
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (mid == x / mid ) {
                return mid;
            } else if (mid < x / mid) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return hi;
    }


    /**
     * 658. Find K Closest Elements
     * 最近的k个element，一样近的话取最小的
     * @return
     *
     * 刚开始binary search找离target最近的点，然后缩小window 当right - left + 1 > k就缩小范围
     */
    public List<Integer> findClosestElements(int[] arr, int k, int target) {
        int len = arr.length;

        // use binary search to find x or closet

        int lo = 0, hi = len - 1;
        while (lo < hi) {                           //  或者 last position的话
            int mid = lo + (hi - lo) / 2;           //  mid = lo + (hi - lo) / 2 + 1   记得 + 1 靠右 否则死循环
            if (arr[mid] < target) {                //  if (arr[mid] <= target) {
                lo = mid + 1;                       //      lo = mid
            } else {
                hi = mid;                           //      hi = mid - 1
            }
        }

        // 也可用自带的
        // int lo = Arrays.binarySearch(arr, target);
        // if (lo < 0) {
        //     lo = -lo - 1;
        // }

        // 考虑 越界的可能性
        int left = Math.max(0, lo - k - 1);
        int right = Math.min(len - 1, lo + k - 1);

        while (right - left + 1 > k) {
            if (target - arr[left] > arr[right] - target) {
                left++;
            } else {
                right--;
            }
        }

        LinkedList<Integer> list = new LinkedList<>();
        for (int i = left; i <= right; i++) {
            list.add(arr[i]);
        }

        return list;
    }

    /**
     * 658. Find K Closest Elements
     * 最近的k个element，一样近的话取最小的
     * @return
     *
     * 主要是binary search找到 start的点，之后就是 [start, start + k] 的sublist
     *
     * 最开始的想法是，s = 0, e = len - 1
     *
     *          跟上题类似
     *         while (e - s + 1 > k) {
     *             if (Math.abs(arr[s]-x) > Math.abs(arr[e]-x))     // 当两边任意的一个abs dist大于另外一边时，舍弃大的
     *                 s++;
     *             else
     *                 e--;
     *         }
     * 但其实我们可以通过binary search来减少搜索range，所以用 a[mid] 和 a[mid + k] 来看move left or right
     */
    public List<Integer> findClosestElements1(int[] arr, int k, int target) {
        int lo = 0;
        int hi = arr.length - k;        //  !!!! len - k

        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (target - arr[mid] > arr[mid + k] - target) {        // 不是绝对值!!!
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        List<Integer> list = new ArrayList<>();
        for (int i = lo; i < lo + k; i++) {         // 从 lo 开始， k个
            list.add(arr[i]);
        }

        return list;
        // return Arrays.stream(arr, lo, lo + k).boxed().collect(Collectors.toList());
    }
	
	
	/** Wood Cut
	 * Given n pieces of wood with length L[i] . 
	 * 切小块使 块数 >= k pieces with the same length.求最长的length
     *@param L: Given n pieces of wood with length L[i]
     *@param k: An integer
     *return: The maximum length of the small pieces.
     * 找到L里最大的数，这是最大的length作为end。然后二分法来算每个length能否符合要求
     */
    public int woodCut(int[] L, int k) {
        int max = 0;
        for (int l : L) {
            max = Math.max(max, l);  //find max to set the largest length as end
        }
        
        int s = 1, e = max;     // these are the length
        while (s + 1 < e) {
            int mid = s + (e - s) / 2;
            if (count(L, mid) >= k) {
                s = mid;
            } else {
                e = mid;
            }
        }
        
        if (count(L, e) >= k) {
            return e;
        }
        if (count(L, s) >= k) {
            return s;
        }
        
        return 0;
        
    }
    
    // given length for every chunk, get the number of total pieces
    public int count(int[] L, int length) {
        int sum = 0;
        for (int l : L) {
            sum += l / length;
        }
        return sum;
    }
    
    
    /** 162. Find Peak Element
     * [1, 2, 3, 1], 3 is a peak element, 返回index 2
     * 用 二分法 快
     */
    public int findPeakElementI(int[] nums) {
        if (nums.length <= 1)   return 0;
        
        int s = 0, e = nums.length - 1;
        while (s < e) {
            int mid = s + (e - s) / 2;
            if (nums[mid] > nums[mid + 1]) {		// 只跟n[mid+1]比，因为mid靠左，+1不会越界
                e = mid;					// 说明这个mid有可能是peak, 所以e=mid..但不要+1
            } else {
                s = mid + 1;
            }
        }


        // 可以在while里 nums[mid] > nums[mid + 1] 时，加上这个 早点break
//        if (nums[mid] < nums[mid + 1] && mid + 2 < nums.length - 1 && nums[mid + 1] > nums[mid + 2]) {
//            return mid;
//        }

        return s;       // 也可以是 e
    }
    
    
    // 正常for  更慢
    public int findPeakElementSeq(int[] nums) {
        int i = 0;
        while (i + 1 < nums.length && nums[i] < nums[i + 1]) {
            i++;
        }
        return i;
    }
    
    
    /** Find Peak II
     * 在2D数组里找peak
     * @param A
     * @return
     * nlog(n) method 也是二分法
     * 1. 先挑中间行
     * 2. 在该行找到最大的col
     * 3. 拿2得到的这行最大的 VS 上下两行比。
     * 再循环2，3
     */
    public List<Integer> findPeakII(int[][] A) {
        int low = 1, high = A.length-2;
        List<Integer> ans = new ArrayList<Integer>();
        while(low <= high) {
            int mid = (low + high) / 2;
            int col = find(mid, A);
            if(A[mid][col] < A[mid - 1][col]) {
                high = mid - 1;
            } else if(A[mid][col] < A[mid + 1][col]) {
                low = mid + 1;
            } else {
                ans.add(mid);
                ans.add(col);
                break;
            }
        }
        return ans;
    }
    
    // 在row这一行，找最大的col
    int find(int row, int [][]A) {
        int col = 0;
        for(int i = 0; i < A[row].length; i++) {
            if(A[row][i] > A[row][col]) {
                col = i;
            }
        }
        return col;
    } 
    
    
    
    /** 302. Smallest Rectangle Enclosing Black Pixels - 二分法
     * 0 as a white pixel and 1 as a black. 1的点都是连在一起的，找能包住1的最小area
     * [  "0010",
		  "0110",
		  "0100"  ]  并且给出任意1的点，比如 (0,2). 返回最少需要area为6
     * @param image
     * @param x
     * @param y
     * @return
     * 这里用二分法主要是因为 给了1的点，这样才能二分找。如果不给的话就只能dfs找了
     * 
     *  left = 二分search col[0...y], find first col contain 1 所以row是O(m)
		right = 二分search col[y+1, col], find first col contain all 0
     *  top = 二分search row [0...x], find first row contain 1, 所以col是O(n)
		bottom = 二分search row[x+1, row], find first row contian all 0
		
     */
    public int minAreaBS(char[][] image, int x, int y) {
        if (image == null || image.length == 0)     return 0;
        int m = image.length, n = image[0].length;
        
        int left = bsearchCol(image, 0, y, 0, m, true);		//找1
        int right = bsearchCol(image, y + 1, n, 0, m, false);	//找0
        int top = bsearchRow(image, 0, x, left, right, true);
        int bottom = bsearchRow(image, x + 1, m, left, right, false);
        
        return (right - left) * (bottom - top);
    }
    
    public int bsearchCol(char[][] image, int s, int e, int top, int bottom, boolean isLeft) {
        while (s < e) {
            int cmid = (s + e) / 2;    //binary search col
            int r = top;    			// 查每行
            while (r < bottom && image[r][cmid] == '0') {
                r++;
            }
            if (r < bottom == isLeft) {     //found 1, need to go left and find first col
                e = cmid;
            } else {            //这col没有，只能往right找
                s = cmid + 1;
            }
        }
        return s;
    }
    
    public int bsearchRow(char[][] image, int s, int e, int left, int right, boolean isTop) {
        while (s < e) {
            int c = left;    // check every col
            int rmid = (s + e) / 2;    //binary search row
            while (c < right && image[rmid][c] == '0') {
                c++;
            }
            if (c < right == isTop) {     //found 1, need to go up and find first row
                e = rmid;
            } else {            //这row没有，只能往下找
                s = rmid + 1;
            }
        }
        return s;
    }
    
    
    // 或者合并只用一个search的话
    public int minAreaBS2(char[][] image, int x, int y) {
        if (image == null || image.length == 0)     return 0;
        int m = image.length, n = image[0].length;
        
        int left = bsearch(image, 0, y, 0, m, true, true);
        int right = bsearch(image, y + 1, n, 0, m, true, false);
        int top = bsearch(image, 0, x, left, right, false, true);
        int bottom = bsearch(image, x + 1, m, left, right, false, false);
        
        return (right - left) * (bottom - top);
    }
    
    public int bsearch(char[][] image, int i, int j, int min, int max, boolean vertical, boolean goLower) {
        while (i < j) {
            int mid = (i + j) / 2;
            int k = min;
            while (k < max && (vertical ? image[k][mid] : image[mid][k]) == '0') {
                k++;
            }
            if (k < max == goLower) {
                j = mid;
            } else {
                i = mid + 1;
            }
        }
        return i;
    }
    
    
    
    
    
    
    public static void main(String[] args) {
    	
    }
	
}
