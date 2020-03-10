
import java.util.*;
import java.util.stream.Collectors;


/**
 * @author Yan
 *
 */
public class dpSolution {
	
	/**
     * 120. Triangle
	 * Given a triangle, find the minimum path sum from top to bottom. 
	 * Each step you may move to adjacent numbers on the row below.
	 * 	[ 其实是左对齐矩阵，只能走[x+1][y]或[x+1][y+1]
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]   The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
	 * @param triangle
	 * @return
	 * DP的话不能从上往下，因为6,5,7是终点，但不知道3,4哪个是谁的起点。而且最后往下有n种可能，不确定返回哪个
	 * 而且从下往上, sum[i][j]表示以i,j做起点。最后往上到[0][0]就是确定的
	 */
	public int minimumTotal(ArrayList<ArrayList<Integer>> triangle) {
        if (triangle == null || triangle.size() == 0) {
            return 0;
        }
        int n = triangle.size();
        int[][] sum = new int[n][n];	// 其实可以化为一维数组，去掉行数
        
        // store the last row 否则DP中sum[i+1][j]会溢出
        for (int i = 0; i < n; i++) {
            sum[n - 1][i] = triangle.get(n - 1).get(i);		//一维数组直接是sum[i]=...
        }
        
        // start DP, from bottom to top	从下到上					
        for (int i = n - 2; i >= 0; i--) {		//倒数第二行开始
            for (int j = 0; j <= i; j++) {	//记得是 j<=i.第4行，那下标i=3. 所以j从0-3=4. 也可以j < triangle.get(i).size()
            	// 每行每个找 [i][j]+下一行的minSum
                sum[i][j] = Math.min(sum[i+1][j], sum[i+1][j+1]) + triangle.get(i).get(j);
                
                // 一维数组的话是
           //     sum[j] = Math.min(sum[j], sum[j+1]) + triangle.get(i).get(j);
            }
        }
        return sum[0][0];
    }
	
	
	 /** Triangle
	  * 这个用一位数组。
	  * 方法1 new int[n], 刚开始先initialize最后一行的数，这样下面的循环从n-2行开始
	  * 方法2 new int[n + 1], 下面循环从n-1行开始，这样就合并了initialize。但是会比上面慢，因为要比较min(0,0)
	 * @param triangle
	 * @return
	 */
	public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle == null || triangle.size() == 0) {
            return 0;
        }

        int n = triangle.size();
        int[] a = new int[n+1];		//方法2，需要n+1，防止index溢出，只是为n-1行的初始化做准备
        /*
        for (int i = 0; i < n; i++) {
            a[i] = triangle.get(n-1).get(i);
        }
        */
        for (int i = n - 1; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                a[j] = Math.min(a[j], a[j+1]) + triangle.get(i).get(j);
                // n-1 行的a[j]这些初始值都是0，所以min(0,0)+tri[i][j]
            }
        }
   //     return a[0];

        //还有一种方法，直接改list里面的值，但是改变不好，而且更慢
        for (int i = n - 2; i >= 0; i--) {		//直接改值的话，就也是n-2开始
            for (int j = 0; j <= i; j++) {
                List<Integer> nextRow = triangle.get(i+1);
                int sum = Math.min(nextRow.get(j), nextRow.get(j+1)) + triangle.get(i).get(j);
                triangle.get(i).set(j, sum);
            }
        }
        return triangle.get(0).get(0);
    }


    /**
     * 931. Minimum Falling Path Sum
     * 给一个 N * N 的矩阵，找Minimum Falling Path Sum，fall下来时只能左右相邻的找min.. 看最后一行加起来的min sum
     * @param A
     * @return
     *
     * 跟上面的triangle一样... 还有 64. Minimum Path Sum 也一样
     */
    public int minFallingPathSum(int[][] A) {
        int len = A.length;

        for (int i = 1; i < len; i++) {
            for (int j = 0; j < len; j++) {
                // min = min(A[i-1][j], A[i-1][j-1], A[i-1][j+1])
                int min = A[i - 1][j];
                if (j > 0) {
                    min = Math.min(min, A[i - 1][j - 1]);
                }
                if (j + 1 < len) {
                    min = Math.min(min, A[i - 1][j + 1]);
                }

                A[i][j] += min;
            }
        }

        int result = Integer.MAX_VALUE;
        for (int sum : A[len - 1]) {
            result = Math.min(result, sum);
        }

        return result;
    }

	
	
	/**
     * 70. Climbing Stairs
	 * 一次爬1或2步
	 * @param n
	 * @return
	 * a[i]时，可以从[i-1]走一步到；也可以从[i-2]走2步到，看他们原先有多少种走法到他们位置
     *
     * dp[i] = dp[i-1] + dp[i-2]            // 可以爬1步或者2步
	 */
	public int climbStairs(int n) {
        if (n < 2) {
            return n;
        }
        int[] result = new int[n];
        result[0] = 1;					//先把这2个特殊情况考虑进去
        result[1] = 2;
        for (int i = 2; i < n; i++) {
            result[i] = result[i-1] + result[i-2];		//状态转移方程
        }								//到这一步的话可以前一格或前2格，看他们原先有多少种走法到他们位置
        return result[n-1];
    }


    /** 91. Decode Ways
     * A-1, B-2...Z-26.
     * Given an encoded message containing digits, determine the total number of ways to decode it.
     * "12", it could be decoded as "AB" (1 2) or "L" (12). 有2种ways
     * @param s
     * @return
     *
     * 其实挺想fibonacci和climb stair.. 一步还是两步.. 只是加了限制条件而已 [10,26]以内
     * dp[i] = dp[i-1] + dp[i-2]
     *
     * 注意每次cur开始，应该是 dp[i] = dp[i-1] （0除外），因为表示 最多几种可能性，而不是只是1
     *
     * 也可以用O(1)空间，而不用数组
     *
     * if (s.charAt(i - 2) == '1' || (s.charAt(i - 2) == '2') && s.charAt(i - 1) <= '6') {
     * 这种还能判断是否valid数字
     */
    public int numDecodings(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int len = s.length();
        int[] ways = new int[len + 1];                                   // O(1) space的话
        ways[0] = 1;	     // suppose 至少一种                              // pre1
        ways[1] = s.charAt(0) != '0' ? 1 : 0;  	                            // pre2

        for (int i = 2; i <= len; i++) {		//注意 i=2 & <=len
            // 注意，是 ways[i-1] 而不是1..因为dp[i]是最多的可能性..不光是1而已
            ways[i] = s.charAt(i - 1) == '0' ? 0 : ways[i - 1];             // int cur = .. 0 : pre2

            // 看当前i的前两位是否10~26，是就算多些可能性 + 上[i-2]有几种可能性
            int twoDigits = Integer.parseInt(s.substring(i-2, i));

            if (twoDigits >= 10 && twoDigits <= 26) {
                ways[i] += ways[i - 2];		                             //cur += pre1
            }
                                                                        // pre1 = pre2; pre2 = cur;  往后移一位
        }
        return ways[len];			                                        // return pre2
    }


    // 如果想要 ways[]长度还是 len..  那后面需要判断 i-2 >= 0 的情况.. 比较麻烦
    public int numDecodings1(String s) {
        int len = s.length();
        int[] ways = new int[len];

        ways[0] = s.charAt(0) == '0' ? 0 : 1;

        for (int i = 1; i < len; i++) {                             // len+1 的话 就是 = 2 <= len
            ways[i] = s.charAt(i) == '0' ? 0 : ways[i - 1];
            int twoDigits = Integer.parseInt(s.substring(i - 1, i + 1));        // len + 1的话就是 sub(i-1, i)
            if (twoDigits >= 10 && twoDigits <= 26) {
                if (i == 1)     ways[i] += 1;                          // len+1 的话 就不用判断这个 直接 ways[i] += ways[i - 2];
                else            ways[i] += ways[i - 2];
            }
        }
        return ways[len - 1];
    }

    // 或者麻烦的 dfs
    private int dfsDecod(String s, int i) {
        if (i == s.length())        return 1;
        if (s.charAt(i) == '0')     return 0;
        if (i == s.length() - 1)    return 1;

        if (s.charAt(i) == '1' || (s.charAt(i) == '2' && s.charAt(i + 1) <= '6')) {
            return dfsDecod(s, i + 1) + dfsDecod(s, i + 2);
        }
        return dfsDecod(s, i + 1);
    }


    /**
     * 639. Decode Ways II
     * A-1, B-2...Z-26. 其中可以有 * , 表示 1-9任何数.. 看最多有几种可能性
     * 因为可能性太多了，需要 mod 10^9 + 7
     * @param s
     * @return
     *
     * 那就是分 cur 情况考虑.. 0，*，正常数.. 而且每个需要看前面一位pre的情况
     *
     * 注意每次cur开始，应该是 dp[i] = dp[i-1] （0除外），因为表示 最多几种可能性，而不是只是1
     */
    public int numDecodingsII(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int mod = 1000000007;

        int len = s.length();
        long[] ways = new long[len + 1];        // 可能性太大了.. 用 long
        ways[0] = 1;

        if (s.charAt(0) == '0')
            return 0;

        ways[1] = s.charAt(0) == '*' ? 9 : 1;

        for (int i = 2; i <= len; i++) {		//注意 i=2 & <=len
            int cur = s.charAt(i - 1);
            int pre = s.charAt(i - 2);
            if (cur == '0') {
                if (pre == '1' || pre == '2') {
                    ways[i] += ways[i - 2];
                } else if (pre == '*') {
                    ways[i] += ways[i - 2] * 2;     // 可为 1 或 2
                } else {
                    return 0;
                }
            } else if (cur == '*') {
                ways[i] += ways[i - 1] * 9;

                if (pre == '1') {
                    ways[i] += ways[i - 2] * 9;
                } else if (pre == '2') {
                    ways[i] += ways[i - 2] * 6;
                } else if (pre == '*') {
                    ways[i] += ways[i - 2] * 15;    // pre 可为 1 或 2
                }
            } else {                                // 正常num
                ways[i] += ways[i - 1];

                if (pre == '1' || (pre == '2' && cur <= '6')) {
                    ways[i] += ways[i - 2];
                } else if (pre == '*') {
                    ways[i] += cur <= '6' ? ways[i - 2] * 2 : ways[i - 2];
                }
            }

            ways[i] %= mod;
        }
        return (int) ways[len];
    }


    // O(1) 节省空间..  preEndAny是 dp[i-1], preEnd1 & preEnd2相当于之前以1/2结束的两位digit.. dp[i-2]
    public int numDecodingsII2(String s) {
        int mod = 1000000007;

        long preEndAny = 1;             // base case  dp[i - 1]
        long preEnd1 = 0, preEnd2 = 0;      // pre num ends in 1 / 2  dp[i - 2] 组成2个digit

        for (char c : s.toCharArray()) {
            long curEndAny = 0;            // dp[i]

            if (c == '*') {
                curEndAny = preEndAny * 9 + preEnd1 * 9 + preEnd2 * 6;
                preEnd1 = preEndAny;
                preEnd2 = preEndAny;
            } else {
                curEndAny = (c == '0' ? 0 : preEndAny) + preEnd1 + (c <= '6' ? preEnd2 : 0);
                preEnd1 = c == '1' ? preEndAny : 0;
                preEnd2 = c == '2' ? preEndAny : 0;
            }

            preEndAny = curEndAny % mod;
        }
        return (int) preEndAny;
    }



    /** Jump Game   --- 贪心算法Greedy
     * 非零int[], 刚开始指到1st index of the array.
     * 每个元素的值代表能跳的max jump
     * 看能否跳到最后一个index
     * A = [2,3,1,1,4], return true.
     * A = [3,2,1,0,4], return false.
     * @param A
     * @return
     */
    public boolean canJump(int[] A) {
        if (A == null || A.length == 0) {
            return false;
        }
        int max = 0;
        for (int i = 0; i <= max; i++) {
        	// 当前i(pos) + 元素(能跳几步). 跟之前max比。
            max = Math.max(max, i + A[i]);    // in case of 0, then A[i] become smaller
            if (max >= A.length - 1) {			// last ele (len-1)
                return true;
            }
        }
        return false;
    }
    
    //用DP，但是慢TLE
    public boolean canJumpSlower(int[] nums) {
        boolean[] dp = new boolean[nums.length];
        dp[0] = true;
        
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && (j + nums[j]) >= i) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[nums.length-1];
    }
    
    /** Jump Game II 用greedy
     * 求出max来看最远能跳多少range。每次走完range(i==end时)就用max来更新end，而且也jump多一步
     * 中间还没到end时，即使max更新了也不用急着更新end和jump，否则跳多了
     * @param A
     * @return
     */
    public int jumpGreedy(int[] A) {
    	int jump = 0, end = 0, max = 0;
    	for (int i = 0; i < A.length - 1; i++) {	// < len-1 !!因为最后一个就到了，要忽略
    		max = Math.max(max, i + A[i]);	//最远能跳多少
    		
    		if (i == end) {			//到这次的end,那就jump++一次就行
    			jump++;
    			end = max;			//设置新的end
    			if (end >= A.length-1) {
    				break;		//也可以直接return
    			}
    		}
    	}
    	return jump;
    }
    
    /** Jump Game II
     * 返回jump最少能到达的次数
     *  A = [2,3,1,1,4]，最少次数是2
     * @param A
     * @return
     * 这个太慢，每次 都要从头找i
     */
    public int jump(int[] A) {
        int jmp = 0;
        int dest = A.length - 1;
        
        while (dest != 0) {			// 往 前 移dest--
            for (int i = 0; i < dest; i++) {
                if (i + A[i] >= dest) {         // 跳一次就能到dest时的 i位置
                    dest = i;                   //往前更新dest位置，哪个i 能跳到当前i(dest)位置
                    jmp++;
                    break;						//没必要再找。越早找到i 越靠前
                }
            }
        }
        return jmp;
    }
    
    
    //slower  DP
    public int jumpSlower(int[] A) {
        int[] steps = new int[A.length];
        
        steps[0] = 0;	// 0 not A[0]
        for (int i = 1; i < A.length; i++) {
            steps[i] = Integer.MAX_VALUE;		
            for (int j = 0; j < i; j++) {
                if (steps[j] != Integer.MAX_VALUE && j + A[j] >= i) {	//i之前的某个j 能到i这位置
                    steps[i] = steps[j] + 1;
                    break;
                }
            }
        }
        
        return steps[A.length - 1];
    }
    
    
    
    
    /** 198. House Robber
     * 不能偷相邻的房子.. 不过不是每次都要偷
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0)   return 0;
        int n = nums.length;
        if (n == 1)     return nums[0];
        if (n == 2)     return Math.max(nums[0], nums[1]);
        
        int[] dp = new int[n];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < n; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[n - 1];
    }
        
    // 把一维数组 变成 2个int代表
    public int rob1(int[] nums, int lo, int hi) {
        int first = 0;
        int second = 0;

        for (int i = lo; i <= hi; i++) {
            int max = Math.max(second, nums[i] + first);
            first = Math.max(first, second);
            second = max;
        }
        return Math.max(first, second);
    }
    
    
    /** 213. House Robber II
     * 如果是个环，头尾不能同时偷。
     * @param nums
     * @return
     */
    public int robII(int[] nums) {
        if (nums == null || nums.length == 0)   return 0;
        int n = nums.length;
        if (n == 1) return nums[0];
        
        // 拆成2个数组就行, one with no end, one with no 1st
        return Math.max(rob1(nums, 0, n - 2), rob1(nums, 1, n - 1)); 
    }
    
    
    /** 337. House Robber III
     * 是tree，从root进去。parent和孩子之间不能一起偷。所以看偷root + 孙子，还是 左子+右子
     * @param root
     * @return
     * 记得加个HashMap来存算过的值，这样不用重复计算  --- DFS + 剪枝
     */
    public int robIII(TreeNode root) {
        Map<TreeNode, Integer> map = new HashMap<>();
        return dfsrob(root, map);
    }
    
    public int dfsrob(TreeNode root, Map<TreeNode, Integer> map) {
        if (root == null) {
            return 0;
        }
        if (map.containsKey(root)) {    
            return map.get(root);   //just return value, no repeated computation
        }
        
        int val = 0;
        
        if (root.left != null) {  			//所有孙子都加一起.. 
            val += dfsrob(root.left.left, map) + dfsrob(root.left.right, map);
        }
        if (root.right != null) {
            val += dfsrob(root.right.left, map) + dfsrob(root.right.right, map);
        }
        val = Math.max(val + root.val, dfsrob(root.left, map) + dfsrob(root.right, map));
        map.put(root, val);
        return val;
    }
    
    
    /** 337. House Robber III
     * DP 方法，用int[2]分别存 root有被rob，或者没rob的情况   - O(1) space 省空间
     * root被rob：root + 孙子
     * root 没被rob：看left / right的最大，孩子可以偷或不偷，没所谓，因为不选root
     * @param root
     * @return
     */
    public int robIIIbest(TreeNode root) {
        int[] arr = dprob(root);     
        return Math.max(arr[0], arr[1]);
    }
    
    public int[] dprob(TreeNode root) {
        if (root == null) {
            return new int[2];
        }
        
        int[] left = dprob(root.left);
        int[] right = dprob(root.right);
        
        int[] arr = new int[2];   // 0 - root robbed, 1 - root not robbed
        arr[0] = root.val + left[1] + right[1];     //means left & right child not robbed
        arr[1] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);	 //可以被偷或不，没所谓因为root没被偷
        return arr;
    }
    
    class TreeNode {
    	int val;
    	TreeNode left;
    	TreeNode right;

    	TreeNode(int x) {
    		val = x;
    	}
    }
    
    
	
	/**
     * 62. Unique Paths
	 * m*n的矩阵，从左上角开始走，只能向右或下走1步，走到右下角
	 * How many possible unique paths are there?
	 */
	public int uniquePaths(int m, int n) {
        if (m < 1 || n < 1) {
            return 0;
        }
        int[][] result = new int[m][n];             
        for (int i = 0; i < m; i++) {           //first col = 1 因为只有一种走法
            result[i][0] = 1;
        }
        for (int j = 0; j < n; j++) {           // first row = 1
            result[0][j] = 1;
        }
        
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {       // up + left
                result[i][j] = result[i-1][j] + result[i][j-1];
            }
        }
        return result[m-1][n-1];
    }
	
	// 可以合并
/*
	for(int i=0;i<m;i++){  
        for(int j=0;j<n;j++){  
            if (i==0 || j==0) dp[i][j] = 1;  
            else dp[i][j] = dp[i-1][j] + dp[i][j-1];  
        }  
    } 
   */
	
	/**
     * 一维数组
     * 循环是从上到下扫描行的，在每一行里，我们又是从左到右来扫描的
	可以用一个滚动数组来表示每一行，算下一行的时候，只需要更新这个数组里的值便可以了。
	 */
	public int uniquePathsBetter(int m, int n) {
        if (m == 0 || n == 0) {
            return 0;
        }
        int[] path = new int[n];
        Arrays.fill(path, 1);

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {	      //j从1开始!! 因为p[j-1]
                path[j] = path[j] + path[j - 1];
                       // p[j]上一行的值   p[j-1]左格的值
            }
        }
        return path[n - 1];
    }
	
	
	/**
     * Unique Paths的数学方法
	 * 总共有N步要走，其中有m-1步是向下（或右），所以就组合方法
	 */
	public int uniquePathsMath(int m, int n) {
            int N = m + n - 2;  //total steps need to reach
            int k = Math.min(m-1, n-1);      //steps need to go down
            long sum = 1;
            
           //combination of (N, k) = N! / k!, only see when to go down
            // i要从小算，这样容易被sum整除。否则如果i从3开始--，sum是8，就整除不了
            for (int i = 1; i <= k; i++) {
                sum = sum * N / i;	// 不能 *=，而要sum * ...，否则后面可能不能整除
                N--;
            }
            return (int)sum;
    }
	
	
	
	/**
     * 63. Unique Paths II
	 * 有障碍物的话是1表示.. 有多少unique paths
	 */
	public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;                // row length
        int n = obstacleGrid[0].length;                
        if (obstacleGrid == null || m == 0 || n == 0 || obstacleGrid[0][0] == 1) {
            return 0;
        }
        int[][] result = new int[m][n];
        result[0][0] = obstacleGrid[0][0] == 1 ? 0 : 1;      // remember to initialize 1st

        for(int i = 1; i < m; i++) {			//同样是第一行和第一列初始化
        	if (obstacleGrid[i][0] == 0) {
                result[i][0] = result[i-1][0];
            } else {
                break;
            }
        }
        for(int j = 1; j < n; j++) {
        	if (obstacleGrid[0][j] == 0) {
                result[0][j] = result[0][j-1];
            } else {
                break;
            }
        }
        
        for (int i = 1; i < m; i++) {
            for(int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    result[i][j] = 0;           
                } else {
                    result[i][j] = result[i][j-1] + result[i-1][j];
                }
            }
        }
        
        // ------------------- 合并 -------------------------------
        for (int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    result[i][j] = 0;           
                } else {
                    if (i == 0 && j == 0) {
                        result[0][0] = obstacleGrid[0][0]== 1 ? 0 : 1;  
                    } else if (i == 0) {
                        result[i][j] = result[i][j-1];
                    } else if (j == 0) {
                        result[i][j] = result[i-1][j];
                    } else {
                        result[i][j] = result[i][j-1] + result[i-1][j];
                    }
                }
            }
        }
        
        return result[m-1][n-1];
    }
	
	// 只用一维数组
	public int uniquePathsWithObstacles1dArr(int[][] obstacleGrid) {
		int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[] result = new int[n];
        result[0] = 1;              // 记得初始化

        for (int i = 0; i <  m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (obstacleGrid[i][j] == 1)
                    result[j] = 0;
                else if (j > 0)
                    result[j] += result[j-1];
            }
        }
        return result[n-1];
	}
	
	
	/**
     * 64. Minimum Path Sum
	 * m*n矩阵，从左上到右下找min path sum。返回path sum
	 * @param grid
	 * @return
	 */
	public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
       
        int rows = grid.length;
        int cols = grid[0].length;
        
        int[][] dp = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = grid[i][j];
                } else if (i == 0) {
                    dp[i][j] = dp[i][j-1] + grid[i][j];
                } else if (j == 0) {
                    dp[i][j] = dp[i-1][j] + grid[i][j];
                } else {
                    dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
                }
            }
        }
        return dp[rows-1][cols-1];
    }
	
	// 用一维数组
	public int minPathSum1dArr(int[][] grid) {
        int rows = grid.length;
        int cols = grid[0].length;
        
        int[] dp = new int[cols];
        dp[0] = grid[0][0];
        
        for (int j = 1; j < cols; j++) {    //initialize 1st row
            dp[j] = dp[j-1] + grid[0][j];
        }
        
        for (int i = 1; i < rows; i++) {
            dp[0] += grid[i][0];    // initialize 1st col
            for (int j = 1; j < cols; j++) {
                dp[j] = Math.min(dp[j], dp[j-1]) + grid[i][j];
            }
        }
        return dp[cols-1];
    }


    /**
     * Unique Paths  - Google
     *
     * 从左上角 走到 右上角
     * 只能 右上，右，右下 这么走
     *
     * dp[i][j] = dp[i - 1][j - 1] + dp[i][j - 1] + dp[i + 1][j - 1]， 注意i-1，i+1需要存在
     */
    public int uniquePathsIII(int rows, int cols) {
	    int[] cur = new int[rows];
	    int[] prev = new int[rows];
	    prev[0] = 1;

	    for (int j = 1; j < cols; j++) {
	        for (int i = 0; i < rows; i++) {
                cur[i] = prev[i];
                if (i > 1) {
                    cur[i] += prev[i - 1];
                }
                if (i + 1 < rows) {
                    cur[i] += prev[i + 1];
                }
            }
	        System.arraycopy(cur, 0, prev, 0, rows);     // 这次的dp当成下次的prev
        }
	    return prev[0];
    }


    /**
     * Unique Paths III - followup
     *
     * 给矩形里3个点，判断是否存在 遍历这3个点的路径..
     * 同样还是 只能 右上，右，右下 这么走
     *
     * 画一下会发现，每个点都在 prev点组成的扇形里面..
     * 所以只要看cur是不是 在prev的扇形里面就行   i - dy <= x <= i + dy
     */
    public boolean canReach(int[][] points) {
        List<int[]> list = new ArrayList<>();
        list.add(new int[]{0, 0});          // 起点

        for (int[] p : points) {
            list.add(p);
        }

        // 按照 列j 排序
        Collections.sort(list, (a, b) -> a[1] - b[1]);

        for (int i = 1; i < list.size(); i++) {
            int[] cur = list.get(i);
            int[] prev = list.get(i - 1);
            int dist = cur[1] - prev[1];        // 列的diff, 其实也就是row最大的diff

            if (dist == 0) {                            // 列相同.. 通常false（因为下一步肯定需要j+1），除非是一样的点
                if (cur[0] == prev[0]) continue;
                else return false;
            }

            int upper = prev[0] - dist;         // row的范围
            int lower = prev[0] + dist;

            if (cur[0] > lower || cur[0] < upper)
                return false;
        }

        return true;
    }


    /**
     * Unique Paths III - followup
     * 给定矩形里的三个点，找到遍历这三个点的所有路径数量
     *
     * 用hashmap保存要经过的points的 j / i, 这样算DP时，只有能到当前点 j & i 才有值，否则其他都为0.
     */
    public int uniquePathsPassPoints(int rows, int cols, int[][] points) {
        Map<Integer, Integer> pointsMap = new HashMap<>();      // key是要经过的point的col, val是对应row

        for (int[] p : points) {
            if (pointsMap.containsKey(p[1]))        // 无法遍历同一列的点，那就直接false返回0了
                return 0;

            pointsMap.put(p[1], p[0]);
        }

        int[] cur = new int[rows];
        int[] prev = new int[rows];
        prev[0] = 1;
        int result = 0;

        for (int j = 1; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                cur[i] = prev[i];
                if (i > 1) {
                    cur[i] += prev[i - 1];
                }
                if (i + 1 < rows) {
                    cur[i] += prev[i + 1];
                }
            }
            System.arraycopy(cur, 0, prev, 0, rows);     // 这次的dp当成下次的prev

            // 要判断 是否是 要经过的points
            if (pointsMap.containsKey(j)) {
                int row = pointsMap.get(j);

                for (int i = 0; i < rows; i++) {
                    if (i == row) {               // 只有 必经点才保留dp ways.. 其他点都到不了设为0
                        result = prev[i];
                    } else {
                        prev[i] = 0;
                    }
                }
            }
        }

        return result;
    }


    /**
     * Unique Paths  - Google  - followup
     *
     * 给定一个下界 lowerBound，找到能经过给定下界的所有从左上到右上的路径数量 (x >= lowerBound)
     *
     * 其实就是 整个ways - 在lowerBound之上的ways(不经过).. 最后的结果就是下半部分 经过lowerbound的ways
     */
    public int uniquePathsIII4(int rows, int cols, int lowerBound) {
        return uniquePathsIII(rows, cols) - uniquePathsIII(lowerBound, cols);
    }

    /**
     * Unique Paths - followup
     *
     * 起点和终点改成从左上到左下，每一步只能 ↓↘ ↙，求所有可能的路径数量
     *
     * 也差不多  dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1] + dp[i - 1][j + 1]
     * 也就是    dp[i] = dp[j] + dp[j-1] + dp[j+1]
     */
    public int uniquePathsIII5(int rows, int cols) {
        int[] cur = new int[rows];
        int[] prev = new int[rows];
        prev[0] = 1;

        for (int i = 1; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cur[i] = prev[j];
                if (j > 1) {
                    cur[i] += prev[j - 1];
                }
                if (j + 1 < rows) {
                    cur[i] += prev[j + 1];
                }
            }
            System.arraycopy(cur, 0, prev, 0, rows);     // 这次的dp当成下次的prev
        }
        return prev[0];
    }



    /**
     * 674. Longest Continuous Increasing Subsequence (subarray) - easy 连续的
     * @param nums
     * @return
     */
    public int findLengthOfLCIS(int[] nums) {
        int max = 0;
        int count = 0;

        for (int i = 0; i < nums.length; i++) {
            if (i == 0 || nums[i - 1] < nums[i]) {
                count++;
                max = Math.max(max, count);
            } else {
                count = 1;
            }
        }
        return max;
    }

	
	/**
     * 300. Longest increasing subsequence
	 * unsorted array of integers, find the length of longest increasing subsequence.
	 * 给[10, 9, 2, 5, 3, 7, 101, 18] --> [2, 3, 7, 101]
	 * @param arr
	 * @return
	 * O(n^2)的DP
     *
	 * lis[i]是前i个最长的LIS。
     *
	 * 每次i往前走，都需要回头看看之前的能否组成LIS, 所以要2层for循环
	 * i往前走，每次j要回去从[0~i)扫一遍，a[j]需要和a[i]比较，再更新lis[i]
	 */
	public static int LIS(int[] arr) {
		if (arr == null || arr.length == 0) {
			return 0;
		}
		int max = 0;
		int[] lis = new int[arr.length];        // lis[i]是前i个最长的LIS。注意 i 当前是最大的 情况下 LIS

        for (int j = 0 ; j < arr.length; j++) {
            lis[j] = 1;                          // 初始化，相当于 Arrays.fill(dp, 1);
            for (int i = 0; i < j; i++) {
                if (arr[i] < arr[j]) {              // 符合递增才算
                    lis[j] = Math.max(lis[i] + 1, lis[j]);      // lis[j]是以i为终点，算之前的每个不同i的可能性。
                }
            }
            max = Math.max(max, lis[j]);
        }
		return max;
	}
	
	
	/**
     * 300. Longest Increasing Subsequence 用二分法做 nlogn的解法
	 * @param nums
	 * @return
	 * 这里的DP跟上面的不同.. 主要靠（有可能跟原先顺序不同的的increasing）的len来表示
	 * 只要len++, 说明有个valid increasing顺序.. 不管后面怎么改前面的dp[]数值，len是不会受影响，除非又找到个新的len++
	 * 
	 * 这个dp[]也是sorted increasing。尽可能让里面的数小，这样才能有更长的可能性
	 * 
	 * 遍历nums里每个数字..然后二分查找dp[](因为sorted)里该num出现的地方..返回的基本是dp[]里第一个>=num的位子
	 * [10, 9, 2, 5, 3, 7, 101, 18]
	 * 找第一个大于num的数，插入到那个位置/或更新.. 
	 * 如果num小于dp里所有数，那会返回0，这时候就放在0
	 * 如果num > dp所有数，那就放到len的地方..比如现在是2,3然后7大于他们，所以放在index=2的地方。
	 * 		当 > all放在len那地方，也就是i==len, 这是len++, 因为又多了一个increasing的数
	 * 这么放，就保证都是increasing的.. 因此sorted就可以用binary search
	 * 
	 * 注意!!! dp[]里的值可能不是真实的LIS..只有len 是对的.. 
	 * 	因为后面如果有数是介于中间，那会改变之前的sequence..
	 * 比如input: [0, 8, 4, 12, 2]
		dp: [0]
		dp: [0, 8]
		dp: [0, 4]
		dp: [0, 4, 12]
		dp: [0 , 2, 12] ... 最后这个不是真实的LIS, 但len仍然是3
	 * 
	 */
	public int lengthOfLIS(int[] nums) {
        int[] sorted = new int[nums.length];
        int len = 0;
        
        for (int num : nums) {
            int i = Arrays.binarySearch(sorted, 0, len, num);
            if (i < 0)  i = -i - 1;

            sorted[i] = num;
            if (i == len) {
                len++;
            }
        }
        return len;
    }

    // ends[]里放smallest tail升序..尽可能放小的数，这样后面increasing机会更大
    // 为何O(NlongN),因为要替换中间的值时用binary search找index来换
    public int lengthOfLIS1(int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;

        int[] ends = new int[nums.length];
        ends[0] = nums[0];
        int maxLen = 0;

        for (int num : nums) {
            if (ends[maxLen] < num) {
                ends[++maxLen] = num;
            } else {
                // 要到已经排好序的ends[]里找第一个比num大的位置，替换掉它
                int index = binarySearchFirstGreater(ends, 0, maxLen, num);
                ends[index] = num;
            }
        }
        return maxLen + 1;  // ends[]只是长度是对的lis, 里面的数不一定是正确的
    }

    // 都适用于上面2种二分情况
    private int binarySearchFirstGreater(int[] arr, int lo, int hi, int target) {
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (arr[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return hi;      // 也可以 lo  一样
    }
	
	
	
	/**
     * 354. Russian Doll Envelopes 俄罗斯套娃
	 * @param envelopes
	 * @return
	 * 跟上面的longest increasing subsequence很像..用n^2的解法
	 */
	public int maxEnvelopes(int[][] envelopes) {
        int[] dp = new int[envelopes.length];
        int max = 0;
        
        // sort envelops
        Arrays.sort(envelopes, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return a[0] - b[0];
            }
        });
        
        
        for (int i = 0; i < envelopes.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (envelopes[j][0] < envelopes[i][0] && envelopes[j][1] < envelopes[i][1]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            max = Math.max(dp[i], max);
        }
        return max;
    }
	
	
	/**
     * 354. Russian Doll Envelopes 俄罗斯套娃 - Binary Search做法
	 * @param envelopes
	 * @return
	 * 跟上面的longest increasing subsequence很像..用nlogn的解法
	 * 
	 * 这里注意几点：
	 * a. 排序时width ascending, height descending。否则[3,3]和[3,4]都会append，但其实不能
	 * b. 因为这里的sorted[]要尽可能用small的数，这样排序后的先iterate到[3,4],然后才是[3,3]。那么sorted里最后更新是[3,3]，尽量保持小
	 * c. binary search时用height作为key，因为之前已经sort过width了，那么现在要比height，让它sort
	 */
	public int maxEnvelopesBS(int[][] envelopes) {
        int[] sorted = new int[envelopes.length];
        
        // sort envelops, width ascending, height descending 这样后面for循环时，小的在大的后面，可以overwrite掉它
        Arrays.sort(envelopes, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                if (a[0] == b[0]) {
                    return b[1] - a[1];     //height descending
                }
                return a[0] - b[0];
            }
        });
        
        int len = 0;

        for (int[] e : envelopes) {      // 之前已经按width排好了，那么sorted[]这次要按照height排..所以找height
            int i = Arrays.binarySearch(sorted, 0, len, e[1]);
            if (i < 0)  i = - i - 1;

            sorted[i] = e[1];
            if (i == len)
                len++;
        }
        return len;
    }
	
	
	/** Longest Common Subsequence
	 * LCS for input Sequences “ABCDGH” and “AEDFHR” is “ADH” of length 3.
	   LCS for input Sequences “AGGTAB” and “GXTXAYB” is “GTAB” of length 4.
	 * @param s1
	 * @param s2
	 * @return
	 * 如果某位不相等，那么 Consider the input strings “ABCDGH” and “AEDFHR. 最后一位不match的话
	 * . So length of LCS can be written as:
		L(“ABCDGH”, “AEDFHR”) = MAX ( L(“ABCDG”, “AEDFHR”)i少了h, L(“ABCDGH”, “AEDFH”)j少了r )
	 */
	public static int LCS(String s1, String s2) {
		if (s1 == null || s2 == null) {
			return 0;
		}
		int[][] lcs = new int[s1.length() + 1][s2.length() + 1];	
		for (int i = 0; i <= s1.length(); i++) {
			for (int j = 0; j <= s2.length(); j++) {
				if (i == 0 || j == 0) {			//初始化
					lcs[i][j] = 0;
				} else if (s1.charAt(i - 1) == s2.charAt(j - 1)) {	//记得else if 否则i,j=0的话i-1出错
					lcs[i][j] = lcs[i - 1][j - 1] + 1;
				} else {
					lcs[i][j] = Math.max(lcs[i - 1][j], lcs[i][j - 1]);
				}
			}
		}
		return lcs[s1.length()][s2.length()];
	}
	
	
	/** Longest Increasing Continuous Subsequence 连续的，所以简单很多
	 * @param A
	 * @return
	 */
	public int longestIncreasingContinuousSubsequence(int[] A) {
        if (A == null || A.length == 0) {
            return 0;
        }
        
        int n = A.length;
        int answer = 1;
        
        // from left to right
        int length = 1; // just A[0] itself
        for (int i = 1; i < n; i++) {
            if (A[i] > A[i - 1]) {
                length++;
            } else {
                length = 1;
            }
            answer = Math.max(answer, length);
        }
        
        return answer;
    }

	
	/** 329. Longest Increasing Path in a 2D Matrix 
	 * 在matrix里找，可以上下左右，随便的顺序
	 * @param A
	 * @return
	 * 因为是随便的顺序，无法for从上到下循环 && 初始状态找不到，任何点都有可能是起点
	 * 就只能用dfs暴力搜索，每次看[i,j]周围的上下左右能否继续下去
	 * Optimize --- 用dp[i][j]存 以i,j结尾最长的LIS是多少，这样减少重复搜索 O(m * n)
	 * 所以是 dfs + memory nextNum
	 */
	public int longestIncreasingContinuousSubsequenceII(int[][] A) {
        if(A.length == 0)
            return 0;
        int m = A.length;
        int n = A[0].length;
        int ans = 0;
        int[][] cache = new int[n][m];		//i,j结尾最长的LIS是多少
        
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) { 
            	cache[i][j] = dfsSearch(A, cache, i, j);
            	ans = Math.max(ans, cache[i][j]);
            }
        }
        
        return ans;
	}
	
	int[] d = {0, -1, 0, 1, 0};
	
	// 在这会recursive地搜很多遍，直到把dp[][]搜完
	public int dfsSearch(int[][] A, int[][] cache, int x, int y)   {
        if (cache[x][y] != 0) {
        	return cache[x][y];	//证明visited过，直接返回，减少重复搜索
        }
        
        int ans = 1;	//至少有一个
        int nx, ny;
        for (int i = 0; i < 4; i++) {
        	nx = x + d[i];
        	ny = y + d[i + 1];
        	if (nx >= 0 && nx < A.length && ny >= 0 && ny < A[0].length) {
        		if (A[x][y] > A[nx][ny]) {		//因为以(x,y)结尾是最长的值，所以需要 > 邻居
        			ans = Math.max(ans, 1 + dfsSearch(A, cache, nx, ny));
        		}
        	}
        }
        cache[x][y] = ans;
        return ans;
	}
	
	
	/** 72. Edit Distance 
	 * 最少改几次变一样find the minimum number of steps required to convert word1 to word2.
     *
     * dp[i][j] 代表 the shortest edit distance between word1[0,i) and word2[0, j).
     * 这2个string总共有m * n的组合可能性
     *
	 * 跟LCS类似. 一样时就不用变=dp[i-1][j-1]. 否则就要考虑3种情况
	 * delete s1是dp[i - 1][j], 因为要删掉i那位
	 * s1 insert是加在最后面,相当于[i+1]==[j]，所以看前面的dp[i][j - 1]如何
	 * replace是dp[i - 1][j - 1] + 1
	 * @param s1
	 * @param s2
	 * @return
	 */
	public int minDistance(String s1, String s2) {
		int[][] dp = new int[s1.length() + 1][s2.length() + 1];
		for (int i = 0; i <= s1.length(); i++) {
			for (int j = 0; j <= s2.length(); j++) {
				if (i == 0) {			//初始化，当s1或s2为空""，dp要1，2，3，4.。。
                    dp[i][j] = j;
                } else if (j == 0) {
                    dp[i][j] = i;		//要比i-1 & j-1 因为dp比字符串多一位
                } else if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
					dp[i][j] = dp[i - 1][j - 1];		//一样就不用加
				} else {	//不一样就要 +1，并且比较3个中最小的
					dp[i][j] = 1 + min3(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
				}
			}
		}

		return dp[s1.length()][s2.length()];
    }
	
	// 变成1D数组   按照上面的来修改
	public int minDistance1D(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[] dp = new int[n + 1];
        for (int j = 0; j <= n; j++) {
            dp[j] = j;          	//初始化，需要删多少才变空
        }
        
        for (int i = 1; i <= m; i++) {
            int prev = dp[0];           //也可以i - 1, 代表 dp[i-1][j-1]
            dp[0] = i;				//初始化这一行，相当于第i行要删掉多少才为空，跟j的初始化差不多
            
            for (int j = 1; j <= n; j++) {
                int nextPrev = dp[j];		//保留上次的dp[i-1][j-1]，最后赋给prev.for next j's dp[i-1][j-1]

                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[j] = prev;           // dp[i-1][j-1]
                } else {
                    dp[j] = 1 + min3(dp[j-1], dp[j], prev);
                }
                prev = nextPrev;                 // update dp[i-1][j-1] 给后面的j用
            }
        }
        return dp[n];
    }
	
	
    private int min3(int a, int b, int c) {
        return Math.min(a, Math.min(b, c));
    }
    
    
    /** 161. One Edit Distance
     *  S and T, determine if they are both one edit distance apart.也不能相同
     * @param s
     * @param t
     * @return
     * 比后面的substring很方便，简洁
     */
    public boolean isOneEditDistance(String s, String t) {
        int m = s.length();
        int n = t.length();
        
        for (int i = 0; i < Math.min(m, n); i++) {
            if (s.charAt(i) != t.charAt(i)) {
                if (m == n) {
                    return s.substring(i+1).equals(t.substring(i+1));
                } else if (m > n) {
                    return s.substring(i+1).equals(t.substring(i));
                } else {
                    return s.substring(i).equals(t.substring(i+1));
                }
            }
        }
        return Math.abs(m-n) == 1;		//不能<1, 不能长度相同.. 剩下情况可能是 "abc", ""或者都空
    }
    
    public boolean isOneEditDistanceMine(String s, String t) {
        int m = s.length();
        int n = t.length();
        if (s.equals(t)) {
            return false;
        }
        if (Math.abs(m-n) > 1) {
            return false;
        }
        
        boolean one = true;
        for (int i = 0, j = 0 ; i < m && j < n; i++, j++) {
            if (s.charAt(i) != t.charAt(j)) {
                if (one) {
                    if (m > n) {
                        j--;		//因为之后都要i++,j++, 所以这里其实j不变，i往前
                    } else if (m < n) {
                        i--;
                    }
                    one = false;
                } else {
                    return false;
                }
            }
        }
        return true;
    }
    
    

    
    
    /** 115. Distinct Subsequences
     * Given a string S and a string T, count the number of distinct subsequences of S equal to T.
     * the number of distinct subsequences of S equal to T
     * S = "rabbbit", T = "rabbit". 返回3  找出S里跟T一样的sub
     * 遇到这种两个串的问题，很容易想到DP
     * http://blog.csdn.net/abcbc/article/details/8978146
     * @param S
     * @param T
     * @return
     * 这里先for T 再S。 无论T的字符与S的字符是否匹配，dp[i][j] = dp[i][j - 1].
     * 因为要匹配的是S, 所以以S的index为准，那S之前的数就是s[j-1]
     * 
     * 就是说，假设S已经匹配了j - 1个字符，得到匹配个数为dp[i][j - 1].
     * 现在无论S[j]是不是和T[i]匹配，匹配的个数至少是dp[i][j - 1]。
     * 	if S[j]和T[i]相等时，我们可以让S[j]和T[i]匹配，然后让S[j - 1]和T[i - 1]去匹配。
     * 
     * ps: 下面做法是先S再T, 都一样
     */
    public int numDistinct(String S, String T) {
        if (S == null || T == null) {
            return 0;
        }   
        int lenS = S.length();
        int lenT = T.length();
        int[][] nums = new int[lenS + 1][lenT + 1];
        
        for (int i = 0; i <= lenS; i++) {
            nums[i][0] = 1;			// T为空时，S每个字母都返回1，因为包含""。而n[0][j]=0 因为S空的话就不会=T
        }
        			// !! <= 
        for (int i = 1; i <= lenS; i++) {
            for (int j = 1; j <= lenT; j++) {
                nums[i][j] = nums[i - 1][j];		// 如果这次不匹配(其实match也要加他)，那就用之前的s[i-1]
                if (S.charAt(i - 1) == T.charAt(j - 1)) {	// i-1 !!
                    nums[i][j] += nums[i - 1][j - 1];
                }
            }
        }
        return nums[lenS][lenT];
    }
    
    // 用2个一维数组，pre存上一行的
    public int numDistinct2Arr(String S, String T) {
    	int sl = S.length();
    	int tl = T.length();
    	
    	int [] preComb = new int[sl+1];	//s 的长
    	int [] comb = new int[sl+1];
    	
    	
    	for(int i=0; i<=sl; i++)
    		preComb[i] = 1;		

    	for(int t=1; t<=tl; ++t){
    		for(int s=1; s<=sl; ++s){	// S在里面那层
    			if(T.charAt(t-1) != S.charAt(s-1)){
    				comb[s] = comb[s-1];
    			}else{
    				comb[s] = comb[s-1] + preComb[s-1];
    			}
    		}
    		
    		for(int i=0; i<=sl; ++i){		//不能直接preCom = comb.. 因为会变的
    			preComb[i] = comb[i];
    		}
    	}
    	
    	return preComb[sl];
    }
    
    // 最优解。1个一维数组。但是要从后往前+。 有点像三角形那题
    public int numDistinct1Arr(String s, String t) {
        int m = s.length();
        int n = t.length();
        
        int[] dp = new int[n + 1];  // t's len
        dp[0] = 1;
        for (int i = 1; i <= m; i++) {
            for (int j = n; j > 0; j--) {  //start from end , > 0 , --
                if (s.charAt(i-1) == t.charAt(j-1)) {
                    dp[j] += dp[j-1];		// != 时就还是原来的数
                }
            }
        }
        return dp[n];
    }

    
    
    /**
     * 139. Word Break I
     * 字符串s 看能否用空格分的单词跟字典对应。单词要完全对应，不是contains
     * @param s
     * @param dict
     * @return
     * 可以先用recursion方法想.. 假设前面(0,j)都对应了，那就要看剩下的(j, i)是否在字典里。
     * 这样就能想到可以用DP。外面的i决定end到哪
     * 
     * dp[i]放前i个是否能break
     * 2个for循环，外层是i, 内层 j从0~i, 看dp[j]=true && s(j,i)是否再dict里  （有点像LIS）
     * j如果从 i-1 从后往前扫会更快
     */
    public boolean wordBreak(String s, Set<String> dict) {
        int n = s.length();
        boolean[] canBreak = new boolean[n + 1];			//dp[i]放前i个是否能break
        canBreak[0] = true;	//第0个是空，默认可以

        // start在里面那层，因为需要算[0, start]的切割点，及[start,end]的substring
        // 不能里层是end = start+1，否则start无法算上0的情况
        for (int end = 1; end <= n; end++) {
//            for (int start = 0; start < end; start++) {       // 也可
            for (int start = end - 1; start >= 0; start--) {            // 从后往前扫会更快，因为更快找到i前面一点的单词
                // 存的时候是dp[end], end是+1的，所以可以用dp[start]来判断
                if (canBreak[start] && dict.contains(s.substring(start, end))) {
                    canBreak[end] = true;
                    break;                      //记得break，否则后面会变成false
                }
            }
        }
        return canBreak[s.length()];
    }
    
    
    /** Word Break - faster
     * 主要扫了 maxWordLength 最长的单词len..然后外循环i, 内循环len <= i && len <= max, 这样提前停
     * 其他思路都一样
     * @param s
     * @param dict
     * @return
     */
    public boolean wordBreak1(String s, Set<String> dict) {
    	boolean[] canBreak = new boolean[s.length() + 1];
    	canBreak[0] = true;					//初始化，后面才能跑
    	int max = getMaxWord(dict);
    	
    	for (int i = 1; i <= s.length(); i++) {		//len <= i, 这样可以包含sub(0,i)的情况
    		for (int len = 1; len <= max && len <= i; len++) {	//len是substring的长度，只能<=max
    			if (canBreak[i - len] && dict.contains(s.substring(i - len, i))) {
    				canBreak[i] = true;			// 如果canBreak[i]为true，则s[0...(i-1)]能被拆分
    				break;
    			}
    		}
    	}
    	return canBreak[s.length()];
    }
    // 上面内for(j)可以早点结束 if 超出剩下单词范围的话
    private int getMaxWord(Set<String> dict) {			
    	int max = 0;
    	for (String word : dict) {
    		max = Math.max(max, word.length());
    	}
    	return max;
    }


    // memory search - recursion
    public boolean wordBreak(String s, List<String> wordDict) {
        return canBreak(s, new HashSet<>(wordDict), 0, new Boolean[s.length()]);
    }

    private boolean canBreak(String s, Set<String> dict, int start, Boolean[] memo) {
        if (start == s.length())
            return true;

        // 不用重复计算，知道后面整个substring是否能break
        if (memo[start] != null)
            return memo[start];

        for (int end = start + 1; end <= s.length(); end++) {
            if (dict.contains(s.substring(start, end)) && canBreak(s, dict, end, memo)) {
                return memo[start] = true;
            }
        }

        return memo[start] = false;
    }
    
     
    /** Word Break  可以不用看 = =
     * @param s
     * @param dict
     * @return
     * i往后跑，到了匹配新word就开始执行后面的match。
    // 因为扫一遍dict 再一次次比，所以比较慢
     */
    public static boolean[] wordBreak(String s, Set<String> dict, boolean[] canBreak) {
    	// 如果canBreak[i]为true，则s[0...(i-1)]能被拆分
    	canBreak[0] = true;					//初始化，后面才能跑
    	
    	for (int i = 0; i < s.length(); i++) {
    		if (canBreak[i] == false) {		//新的匹配word起点通常是true
    			continue;
    		}
    		for (String word : dict) {
    			int end = i + word.length();	//str里第几个匹配的Word
    			if (end > s.length()) {
    				continue;					// 换word
    			}
    			if (s.substring(i, end).equals(word)) {
    				canBreak[end] = true;		
    			}
    		}
    	}
    	return canBreak;			//len 不是len-1
    }


    
    /** Word Break II  返回结果
     * 先在主函数里用上面的DP, 然后dfs.. 递归里看到canBreak[i+1]FALSE就不查了，省时间
     * @param s
     * @param dict
     * @return
     * 利用canBreak[]，dfs里，如果canBreak[i+1] = true, 那就看 sub(pos,i]有没在dict里
     */
    public List<String> wordBreakII(String s, Set<String> dict) {
        List<String> result = new ArrayList<String>();
        
        boolean[] canBreak = new boolean[s.length() + 1];
        wordBreak(s, dict, canBreak);
        
        if (!canBreak[s.length()])              //can't break
            return result;                      // not null
        
        
        StringBuilder sb = new StringBuilder();
        recBreak(result, dict, s, sb, 0, canBreak);
        return result;
    }
    
    private void recBreak(List<String> result, Set<String> dict, String s, StringBuilder sb, int pos, boolean[] canBreak) {
        if (pos == s.length()) {						
            String rst = sb.toString().trim();
            result.add(rst);
            return;
        }
        
        // 记得是 i < s.len, 不是dp的
        for (int i = pos; i < s.length(); i++) {
            String str = s.substring(pos, i + 1);
            //要判断[i+1]，因为[0]是空，所以往后移。 而且要再查下是否contain
            if (!canBreak[i + 1] || !dict.contains(str)) {      //[i+1] 因为dp比s多了一位
                continue;
            }
            
            //或者直接用string，不用回溯delete  更方便
          //  dfs(result, s, dict, dp, cur + sub + " ", i + 1);
            
            sb.append(str).append(" ");
            recBreak(result, dict, s, sb, i + 1, canBreak);
            sb.delete(sb.length() - 1 - str.length(), sb.length());		//记得还要-str长
        }

        // 或者  for (i = pos + 1 <= len ..) { sub(pos, i)... canBreak[i]... dfs(.., i);
    }
    
    
    
    public List<String> wordBreakIIBetter(String s, Set<String> wordDict) {
    	return wordBreak(s, wordDict, new HashMap<String, ArrayList<String>>());
    }       

    
    /** Word Break II - Divide & Conquer
     * 把string分成左右两边，如果是先分左，那就dict里要有左，然后dfs(右)
     * dfs右 会输出 List<str> 表示右sub 的不同Word组合。这样 一个左单词 + n个右valid组合
     * 用map缓存每次的string + 他的组合，这样节约时间
     * @return
     * 如果先substring右边，dfs里放前半段，会更快????
     * 
     * 另外的做法，如果dictionary比较小，那可以外循环一下dict。
     * for (String word : wordDict) {		//循环字典
        if (s.startsWith(word)) {
        	.....dfs 之类的
        }
       }
     */
    public List<String> wordBreak(String s, Set<String> dict, HashMap<String, ArrayList<String>>map) {
    	if (map.containsKey(s)) {
    		return map.get(s);
    	}
    	
    	ArrayList<String> result = new ArrayList<String>();
    	
    	if (dict.contains(s)) {
    		result.add(s);			//记得加这个，否则为空!! 
    	}		//可能是leetcode在dict里，然后又能分leet, code,所以2种可能
    	
    	for (int i = 1; i < s.length(); i++) {			//如果先分left，i=1 <= len, left=sub(0, i).后面rec(sub(i)
    		String right = s.substring(i);
    		if (dict.contains(right)) {
    			List<String> lefts = wordBreak(s.substring(0, i), dict, map);
    			for (String left : lefts) {
					result.add(left + " " + right);	//this s has multiple word combinations
				}
    		}
    	}
    	// s有List<Str>个组合。如果放在上面的if-for{}里会超时!! 这个就每次放，即使是""也放，这样返回得快
    	map.put(s, result);	 
    	return result;
    }
    
    
    
    /** 516. Longest Palindromic Subsequence
     * 这个是subsequence.. 中间能断开的 给"bbbab"，return 4，因为bbbb
     * @param s
     * @return
     * 大多数subsequence用DP来做。然后这个跟每个区间是否为palindrome相关，所以是区间型DP
     */
    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];   // max palindrome length from i ~ j

        // i从后往前.. 这样初始化也在循环体里。 要确保 短/小的都算过了，或者状态转移方程的值都求过了
        for (int i = n - 1; i >= 0; i--) {
            dp[i][i] = 1;
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }

        // ===========或者可以下面这样 == 区间型DP, 最外层是长度length =======================

        // 注意，需要提前先初始化!!!!!  否则只有一个a 时，进不去后面的for(i + len < n) 里
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }

        // 区间型DP
        for (int len = 1; len <= n; len++) {
            for (int i = 0; i + len < n; i++) {
                int j = i + len;        //end
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n-1];
    }


    // 一维 省空间
    public int longestPalindromeSubseq1D(String s) {
        int n = s.length();
        int[] dp = new int[n];   // max palindrome length from i ~ j
        dp[n - 1] = 1;

        for (int i = n - 1; i >= 0; i--) {
            int prev = dp[i];       // dp[i + 1][j - 1]
            dp[i] = 1;
            for (int j = i + 1; j < n; j++) {
                int preJ = dp[j];       // j 往后走，所以这层j 循环的preJ 相当于后面j+1的 dp[i + 1][j - 1]

                if (s.charAt(i) == s.charAt(j)) {
                    dp[j] = 2 + prev;        // 往里面找
                } else {
                    dp[j] = Math.max(dp[j], dp[j - 1]);
                }

                prev = preJ;
            }
        }
        return dp[n - 1];
    }

    // 一维 省空间
    public int longestPalindromeSubseq1D1(String s) {
        int n = s.length();

        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int j = 1; j < n; j++){
            int len = 0;
            for (int i = j - 1; i >= 0; i--){
                int tmp = dp[i];
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i] = len + 2;
                }
                len = Math.max(tmp, len);
            }
        }
        int res = 1;

        for (int i: dp) {
            res = Math.max(i, res);
        }
        return res;
    }

    
    // 用 dfs + nextNum
    public int longestPalindromeSubseqDFS(String s) {
        int n = s.length();
        return helper(s, 0, n - 1, new int[n][n]);
    }
    
    public int helper(String s, int i, int j, int[][] cache) {
        if (cache[i][j] > 0)      return cache[i][j];
        if (i > j)                return 0;
        if (i == j)               return 1;
        
        if (s.charAt(i) == s.charAt(j)) {
            cache[i][j] = 2 + helper(s, i + 1, j - 1, cache);
        } else {
            cache[i][j] = Math.max(helper(s, i + 1, j, cache), helper(s, i, j - 1, cache));
        }
        return cache[i][j];
    }



    /**
     * 730. Count Different Palindromic Subsequences
     * 算不同palindromic subsequnce. 需要 modulo 10^9 + 7
     * @param s
     * @return
     * 跟上面的Longest Palindromic Subsequence 类似，但是这里还需要考虑 重复 的情况
     *
     * a. 如果 s[i] != s[j] 不同，那就 dp[i][j] = dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1];
     *    因为前两者都算过 dp[i + 1][j - 1]，所以需要减去重复的
     *
     * b. s[i] == s[j] 相同
     *    那就dp[i][j] = 2 * dp[i + 1][j - 1];
     *    里面自己 单独 的情况，还要加上被外面s[i],s[j]包裹起来 的情况，所以 * 2
     *     'bcb' is pre computed to 3 ('b', 'c', 'bcb')
     *     现在外面包了'a' 这样'abcba', 那么 * 2变成 6 ('b', 'c', 'bcb', 'aba', 'aca', 'abcba')
     *
     *  需要看 中间里面的 字母是否相同，减去重复计算的
     *  1. left > right   里面没有重复的，加上最外层 'a'和'aa'
     *       dp[i][j] += 2
     *  2. left == right  有一个重复。比如"aaa"的情况，里面中间那个'a'之前算过了，这次只能加外层'aa'
     *       dp[i][j] += 1
     *  3. left < right   有多个重复 (跟s[i]一样)字母，需要减去之前算过的
     *       dp[i][j] -= dp[left + 1][right - 1]  减 ...
     *
     * https://leetcode.com/problems/count-different-palindromic-subsequences/discuss/112757/Java-solution-using-simple-DP.-O(n2)-run-time-and-O(n2)-space
     */
    public int countPalindromicSubsequences(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];     // i~j 的不同palindrome个数

        // 初始化
//        for (int i = 0; i < n; i++) {
//            dp[i][i] = 1;
//        }
//
//        for (int len = 1; len < n; len++) {
//            for (int i = 0; i < n - len; i++) {
//                int j = i + len;

        for (int i = n - 1; i >= 0; i--) {
            dp[i][i] = 1;
            for (int j = i + 1; j < n; j++) {

                if (s.charAt(i) != s.charAt(j)) {       // i,j 字母不同
                    // 因为前两个都重复算过dp[i + 1][j - 1]，所以记得要减去

                    dp[i][j] = dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1];

                } else {                                // i,j字母一样
                    // 里面自己 单独 的情况，还要加上被外面s[i],s[j]包裹起来 的情况，所以 * 2
                    // aba, 里面是'b' = 1. 那么加上外面的'a'以后，总共就是'b', 'aba' 所以是 里面 * 2
                    dp[i][j] = 2 * dp[i + 1][j - 1];

                    // 需要查下里面有没重复计算的
                    int left = i + 1, right = j - 1;
                    char val = s.charAt(i);

                    while (left <= right && s.charAt(left) != val) {
                        left++;
                    }
                    while (left <= right && s.charAt(right) != val) {
                        right--;
                    }

                    if (left > right) {
                        dp[i][j] += 2;      // 里面没有重复的，加上最外层 'a'和'aa'
                    } else if (left == right) {
                        dp[i][j] += 1;      // 比如"aaa"的情况，里面中间那个'a'之前算过了，这次只能加外层'aa'
                    } else {
                        dp[i][j] -= dp[left + 1][right - 1];    //里面有重复的(跟s[i]一样)字母，需要减去之前算过的
                    }
                }

                // 最后记得 mod一下，题目要求 防止overflow
                dp[i][j] = dp[i][j] < 0 ? dp[i][j] + 1000000007 : dp[i][j] % 1000000007;
            }
        }
        return dp[0][n - 1];
    }



    /**
     * 131. Palindrome Partitioning
     * 返回所有palindrome结果
     * 
     * 先用Boolean[][] 存是否为palindrome.  也可以用 普通的 双指针往外扩散 判断
     *
     * 然后在dfs里，if (dp[pos][i]为true, 就dfs后面
     * @param s
     * @return
     */
    public List<List<String>> partitionDP(String s) {
        List<List<String>> res = new ArrayList<>();

        int n = s.length();
        boolean[][] dp = new boolean[n][n];

        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                dp[i][j] = s.charAt(i) == s.charAt(j) && (j - i <= 1 || dp[i + 1][j - 1]);
            }
        }

        helperPal(res, new ArrayList<>(), dp, s, 0);

        return res;
    }
    
    private void helperPal(List<List<String>> res, List<String> path, boolean[][] dp, String s, int pos) {
        if(pos == s.length()) {
            res.add(new ArrayList<>(path));
            return;
        }
        
        for(int i = pos; i < s.length(); i++) {
            if(dp[pos][i]) {                // 或者正常的 isPalindrome()双指针方法就行
                path.add(s.substring(pos,i + 1));
                helperPal(res, path, dp, s, i + 1);
                path.remove(path.size() - 1);
            }
        }
    }
    
    
    
    /**
     * 132. Palindrome Partitioning II
     * For example, given s = "aab", 返回cut次数最少的
     * Return 1 since the palindrome partitioning ["aa","b"] could be produced using 1 cut.
     * @param s
     * @return
     * dp[i]表示s[0, i]第i位到以前，能最少的cut
     * for i是end。然后需要for j，是start。如果 s[j, i]是palindrome，那么min可能为 s[0,j-1] + 1
     *  -- 为了快速算出 s[j, i]是否为palindrome, 就用了boolean[n][n]来存。否则每次call for走一遍会慢
     *  
     *  Q: 为何不需要看 isP[j,i] && isP[i-j, i] 前后两段
     *  A: 因为现在是以i 结尾，只算了前面，后面还没算.. 不要搞混
     */
    public int minCut(String s) {
    	int len = s.length();
    	boolean[][] isP = new boolean[len][len];
    	// i从后，j从i--len结尾。整个结果是右上三角形。对角线都是true.
    	for (int i = len - 1; i >= 0; i--) {
    		for (int j = i; j < len; j++) {		
    							// i-j=0或1 这判断要写在前面    外面i,j判断完后，要判断里面的
    			if (s.charAt(i) == s.charAt(j) && (j - i < 2 || isP[i + 1][j - 1]))
    				isP[i][j] = true;
    		}
    	}
    	
    	int[] minC = new int[len];

        for (int j = 0; j < len; j++) {
            minC[j] = j;            //初始化，最多割i次
            for (int i = 0; i <= j; i++) {
                if (isP[i][j]) {
                    if (i == 0) {           // i = 0, [i ~ j]是palindrome, 那么不用cut
                        minC[j] = 0;
                    } else {
                        minC[j] = Math.min(minC[j], 1 + minC[i - 1]);
                    }
                }
            }
        }
    	
    	// *****可以把上面2个for合并起来***************************
        for (int j = 0; j < len; j++) {
            minC[j] = j;
            for (int i = 0; i <= j; i++) {
                if (s.charAt(i) == s.charAt(j) && (j - i <= 1 || isP[i + 1][j - 1])) {
                    isP[i][j] = true;
                    minC[j] = i == 0 ? 0 : Math.min(minC[j], 1 + minC[i - 1]);      //如果i==0,整串都是palindrome. 否则dp[i-1]越界
                }
            }
        }


        // follow-up.. 打印结果的话   从后往前加
        // "aaacabcc"  , cut[] 结果是 00011233  打印结果是 [aa, aca, b, cc]
        List<String> list = new LinkedList<>();
        int last = s.length() - 1;
        int lastCut = minC[last];
        for (int i = s.length() - 2; i >= 0; i--) {   	//每次找最少cut的前一个 & 后面is palindrome的地方
            if (minC[i] == lastCut - 1 && isP[i + 1][last]) {
                list.add(0, s.substring(i + 1, last + 1));		// 加最后面的 pal[i+1, end]
                last = i;
                lastCut--;
            }
        }
        list.add(0, s.substring(0, last + 1));          // 最后记得加.. 因为cut在 0,1,2位置都不变还是0
        System.out.println(list);


        return minC[len - 1];
    	
    	
    /* 也可以从后往前。 如果发现T[i][j]是一个Palindrome，那么从T[i][j]我们中间就不用切刀啦，然后去加上j位后面一位需要切刀的数量（+1是因为要保存最左边的一刀），
     * 然后我们和之前的cut[i]这一位进行比较，如果比之前的刀数少，我们便保留这个值，
     * 在循环到结尾的过程中一直最小值就能保证第i位以后的字串保持最小切刀数目，一直到cut[0]为止，
     * 最后因为第一个字符最左边不需要切，我们返回cut[0] - 1
    	cut[len] = 0;  
        for(int i=len-1;i>=0;i--){  
            cut[i] = len - i;  
            for(int j=i;j<len;j++){  
                if (T[i][j]){  
                        cut[i] = Math.min(cut[i],1+cut[j+1]);  
                }  
            }  
        }  
        return cut[0] - 1;  
     */
    }
    
    
    /**
     * 更快，且不需要额外的二维数组
     *
     * 从中点center向外扩散
     */
    public int minCutExpandFromCenter(String s) {
        int n = s.length();
        int[] cut = new int[n];

        Arrays.fill(cut, n - 1);        // 在外面初始化

        for (int i = 0; i < n; i++) {
            expandPalindrome(s, cut, i, i);
            expandPalindrome(s, cut, i, i + 1);
        }
        return cut[n - 1];
    }

    private void expandPalindrome(String s, int[] cut, int i, int j) {
        while (i >= 0 && j < s.length() && s.charAt(i) == s.charAt(j)) {
            // 只算end即可
            cut[j] = i == 0 ? 0 : Math.min(cut[j], 1 + cut[i - 1]);
            i--;
            j++;
        }
    }

    /**
     * 更快，且不需要额外的二维数组   (比上面麻烦一点点)
     *
     * 从中点center向外扩散
     *
     * i作为中点，j作为radius半径.. dp[i]表示 s[0,i) 或 s[0, i - 1]的最少cut数
     * 比如s[i] = b, and s[i-1,i+1] is palindrome "aba"
     * .......aba...
     * |<-X->| ^
     * |<-- Y -->|
     * 最少cut of s[0,i-1) is X, 那么least cut for s[0, i+1] Y 就 <= X+1. 也就是s[0,x+1+1) --> dp[i+j+1]
     * Each cut at i+j is calculated by scanning (i-j)'s minimum cut + 1 if s[i-j, i+j] is a palindrome.
     *
     * 注意cuts[0]= -1. 如果s[0,i]刚好都是palindrome, 那需要看cut[0] + 1, 所以这时[0]要为-1
     */
    public int minCut2(String s) {
        if(s == null || s.length() == 0) return 0;
        int len = s.length();
        int[] cuts = new int[len + 1]; 		// len + 1
        for (int i = 0; i <= len; i++) 
        	cuts[i] = i - 1; // max cuts。 注意cuts[0]= -1. 如果s[0,i]刚好都是palindrome, 那需要看cut[0] + 1, 所以这时[0]要为-1 
        
      //i是中点center  。 j是半径, j从0到大，所以 i+j范围从里面扩开来
        for (int i = 0; i < len; i++) {			//i从0开始，这样奇数，自己也能算到 
            // odd palin 奇数
            for (int j = 0; i - j >= 0 && i + j < len && s.charAt(i - j) == s.charAt(i + j); j++) 
            	cuts[i + j + 1] = Math.min(cuts[i + j + 1], 1 + cuts[i - j]);
            
            // even palin 偶数，s[i-j+1]代表当前i的位置，或者往左。 s[i+j]代表当前i+1，或者往右
            for (int j = 1; i - j + 1 >= 0 && i + j < len && s.charAt(i - j + 1) == s.charAt(i + j); j++) 
            	cuts[i + j + 1] = Math.min(cuts[i + j + 1], 1 + cuts[i - j + 1]);
        }
        return cuts[len];
    }
    
    
    
    
    /**
     * 97. Interleaving String
     * s1 = "aabcc", s2 = "dbbca", 
     * When s3 = "aadbbcbcac", return true.W
     * when s3 = "aadbbbaccc", return false. 因为最后的a顺序往前，不对
     * http://blog.csdn.net/u011095253/article/details/9248073
     * 
     * dfs方法比较好理解.. 要加cache, 否则超时
     * This is very similar to top down dp
		Only need to nextNum invalid[i][j] since most of the case s1[0 ~ i] and s2[0 ~ j] does not form s3[0 ~ k].
		记得存 invalid.. 否则如果存valid的话，刚开始都是FALSE，那很容易返回错误
	 * @return
	 */
	public boolean isInterleaveRec(String s1, String s2, String s3) {
	    char[] c1 = s1.toCharArray(), c2 = s2.toCharArray(), c3 = s3.toCharArray();
		int m = s1.length(), n = s2.length();
		if(m + n != s3.length()) return false;
		return dfs(c1, c2, c3, 0, 0, 0, new boolean[m + 1][n + 1]);
	}
	
	public boolean dfs(char[] c1, char[] c2, char[] c3, int i, int j, int k, boolean[][] invalid) {
		if (invalid[i][j]) 		return false;
		if (k == c3.length) 	return true;
		
		boolean valid = 
		    i < c1.length && c1[i] == c3[k] && dfs(c1, c2, c3, i + 1, j, k + 1, invalid) || 
	        j < c2.length && c2[j] == c3[k] && dfs(c1, c2, c3, i, j + 1, k + 1, invalid);
		
		if (!valid) 
			invalid[i][j] = true;
	    return valid;
	}
    
    
	
    /** 97. Interleaving String    DP 解法
     * @param s1
     * @param s2
     * @param s3
     * @return
     * dp[i][j]表示s1取前i位，s2取前j位，是否能组成s3的前i+j位
     * 如果第i,j位都 = s3[k]位，那就用 ||来判断
     * dp[i][j] = s1[i]==s3[i+j-1] && dp[i-1][j], 因为i的前一位决定了true或false.  j也是一样 
     */
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s3.length() != s1.length() + s2.length())
            return false;
        boolean[][] dp = new boolean[s1.length() + 1][s2.length() + 1];
        
        // intiallize .... string index from 0, dp[][] from 1
        for (int i = 0; i <= s1.length(); i++) {
            for (int j = 0; j <= s2.length(); j++) {
            	if (i == 0 && j == 0) {
                    dp[0][0] = true;
                } else if (i == 0) {
                    dp[0][j] = dp[i][j-1] && s2.charAt(j-1) == s3.charAt(j-1);
                } else if (j == 0) {
                    dp[i][0] = dp[i-1][j] && s1.charAt(i-1) == s3.charAt(i-1);
                } else {
                    dp[i][j] = (dp[i-1][j] && s1.charAt(i-1) == s3.charAt(i+j-1)) || (dp[i][j-1] && s2.charAt(j-1) == s3.charAt(i+j-1));
                }
            }
        }
        
        return dp[s1.length()][s2.length()];
    }
    
    // 用一维数组
    public boolean isInterleave1Darr(String s1, String s2, String s3) {
        int len1 = s1.length();
        int len2 = s2.length();
        if (s3.length() != len1 + len2) {
            return false;
        }
        
        boolean[] dp = new boolean[len2 + 1];
        dp[0] = true;
        for (int j = 1; j <= len2; j++) {	
            if (dp[j-1] && s2.charAt(j-1) == s3.charAt(j-1)) {
                dp[j] = true;
            }
        }
        
        for (int i = 1; i <= len1; i++) {
            if (dp[0] && s1.charAt(i-1) == s3.charAt(i-1)) {
                dp[0] = true;
            } else {
                dp[0] = false;	// 记得false，因为只有一维了需要更新
            }
            for (int j = 1; j <= len2; j++) {	//dp[j] -> dp[i-1][j] 上一行的
                if (dp[j] && (s1.charAt(i-1) == s3.charAt(i+j-1)) || 
                		dp[j-1] && (s2.charAt(j-1) == s3.charAt(i+j-1))) {
                    dp[j] = true;		// dp[j-1] -> dp[i][j-1] 这行的前一个
                } else {
                    dp[j] = false;
                }
            }
        }
        return dp[len2];
    }
    
    
     
    /** 87. Scramble String - Recursion
     * "rgeat" is a scrambled string of "great".
     * @param s1
     * @param s2
     * @return
     * separate s1 into two parts, namely --s11--, --------s12--------
	 * separate s2 into two parts, namely --s21--, --------s22--------, 看 (s11 and s21) && (s12 and s22) 是否 isScramble.
	 * separate s2 into two parts, namely --------s23--------, --s24--, 看(s11 and s24) && (s12 and s23) 是否isScramble.
     */
    public boolean isScrambleRec(String s1, String s2) {
        if (s1 == null || s2 == null || s1.length() != s2.length()) {
            return false;
        }
        if (s1.equals(s2)) {
            return true;
        }
        
        int len = s1.length();
        
        int[] count = new int[256];
        for (int i = 0; i < len; i++){
            count[s1.charAt(i)]++;
            count[s2.charAt(i)]--;
        }
        
        // 或者如果都是小写字母, count[s1.charAt(i) - 'a]++
        // 然后这里for < 26, count[i] != 0
        for (int i = 0; i < len; i++){
            if (count[s1.charAt(i)] != 0) return false;
        }
        
        /* 跟上面一样，但是sort会慢点
        char[] c1 = s1.toCharArray();
        char[] c2 = s2.toCharArray();
        Arrays.sort(c1);
        Arrays.sort(c2);
        
        if (!Arrays.equals(c1, c2)) {
            return false;
        }
        */
        
        // check substring
        for (int i = 1; i < len; i++) {
        	//s11 , s21 && s12, s22 可能s1前半段 跟 s2前半段一样
            if (isScramble(s1.substring(0, i), s2.substring(0, i)) && 
                isScramble(s1.substring(i), s2.substring(i))) {
                return true;
            }
            
            //s11, s22 && s12, s21  可能s1前半段 跟 s2后半段一样（调转过）
            if (isScramble(s1.substring(0, i), s2.substring(len - i)) && 
                isScramble(s1.substring(i), s2.substring(0, len - i))) {
                return true;
            }
        }
        return false;
    }
    
     
    /** Scramble String
     * Second method: comes up with DP 
     * dp[i][j][len]，i, j表示起点，长度为len的词是否互为scramble
     * 表示从s1的第i个字符开始长度为len的子串，和从s2的第j个字符开始长度为len的子串，是否互为scramble。

		初始化为dp[i][j][1] = s1.charAt(i) == s2.charAt(j)，即长度为1的子串是否互为scramble。
       f[n][i][j] means isScramble(s1[i: i+n], s2[j: j+n])
       dp[len][i][j] = (dp[k][i][j] && dp[len-k][i+k][j+k]) 
                     || (dp[k][i][j+len-k] && dp[len-k][i+k][j]);
                     
     * 最后返回dp[n][0][0]，因为i,j要从0开始
     * @param s1
     * @param s2
     * @return
     */
    public boolean isScramble(String s1, String s2) {    	
    	if (s1.length() != s2.length())     return false;
        if (s1.equals(s2))                  return true;

        int n = s1.length();
        boolean[][][] dp = new boolean[n + 1][n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                dp[1][i][j] = s1.charAt(i) == s2.charAt(j);
            }
        }
        
        // len means the s1, s2的长度. 至少要为2，才能把它们切分substring
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i + len <= n; i++) {    //i是s1的起点， s1[i...i+k]
                for (int j = 0; j + len <= n; j++) {    //j is start of s2
                    for (int k = 1; k < len; k++) {     // 以k为切割点把s1, s2 分2 parts。代表不同length
                        if (dp[len][i][j]) {
                            break;      //pruning
                        }  
                        dp[len][i][j] = (dp[k][i][j] && dp[len-k][i+k][j+k]) ||
                                        (dp[k][i][j+len-k] && dp[len-k][i+k][j]);
                    }
                }
            }
        }
        return dp[n][0][0];     //start from 0, length is n
    }
    
    
    

    /**
     * 471. Encode String with Shortest Length
     * Input: "aaa". Output: "aaa"
     * Input: "aaaaa"  .Output: "5[a]"
     * Input: "abbbabbbcabbbabbbc"  . Output: "2[2[abbb]c]"
     * @param s
     * @return
     * dp[i][j]表示i~j(inclusive)的substring
     * 1. 切割时成两半，dp[i][j] = min(dp[i][k] + dp[k+1][j], dp[i][j]) 看左右两半是否加起来更小
     * 		比如abbbbb abbbbb，分成两半时都已经缩写成a5[b], 所以dp[i][j]不是整个，而是a5[b] a5[b].
     * 2. 分配完dp[i][j]后，要看这个substring是否有重复可以缩写，所以变成 2[a5[b]]
     * 		查t是否repeat时，组成2个t, 查t是否从1开始重新出现，并idx < t.len。
     * 					比如abab, 组成2个t就是abababab,能找到重复的t,idx为2, < 4, 就是找到了
     * 	  !!在这步查repeat时，是根据sub查，而不是dp[i][j]..
     * 		如果一个sub重复了3次 ababab, 那repeat后idx = 2，重复次数为 sub.len / idx -> 6 / 2 = 3
     */
    public String encode(String s) {
        int n = s.length();
        String[][] dp = new String[n][n];       // [i,j] inclusive
        
        for (int l = 0; l < n; l++) {			//长度在最外层
            for (int i = 0; i < n - l; i++) {		// i, j 一直在变. 这样后面算k啥的才能有确定的dp值，否则都算不全
                int j = i + l;      //end
                String sub = s.substring(i, j + 1);
                dp[i][j] = sub;
                if (l < 4) {
                    continue;
                } else {
                    for (int k = i; k < j; k++) {       //k to cut string into 2 substr
                        if (dp[i][k].length() + dp[k+1][j].length() < dp[i][j].length()) {
                            dp[i][j] = dp[i][k] + dp[k+1][j];
                            
                            //为何不在这里保留k的位置作为重复点？
                            // NO..因为这样就只是重复2次，但有时不一定，有可能重复3，4次
                            // 比如 aba bab。其实有可能是3[ab], 而不是2[aba]
                        }
                    }
                    
                 // check for repeat pattern ，用sub来查，而不是dp[i][j]
                    String replace = "";
                    // 看sub里面是否有重复的。如果找到的idx < sub.len，就说明有重复。
                    int idx = (sub + sub).indexOf(sub, 1);		
                    if (idx >= sub.length()) {
                        replace = sub;			//记得要有，否则最后 replace.len < dp.len会只取""
                    } else {									//从idx重复的那位之前的一位为end点
                        replace = sub.length() / idx + "[" + dp[i][i + idx - 1] + "]";
                    }
                    if (replace.length() < dp[i][j].length()) {
                        dp[i][j] = replace;		
                    }
                    
                    /*
                    // 这下面跟上面的replace判断差不多，都是看有没重复。但是这里repaceAll()是 O(n)，所以更慢
                    for (int k = 0; k < sub.length(); k++) {
                        String repeat = sub.substring(0, k + 1);
                        if (repeat != null && sub.length() % repeat.length() == 0
                            && sub.replaceAll(repeat, "").length() == 0) {  //能否全匹配能替换O(n)
                            String ss = sub.length() / repeat.length() + "[" + dp[i][i+k] + "]";
                            if (ss.length() < dp[i][j].length()) {
                                dp[i][j] = ss;
                            }
                        }
                    }
                    */
                }
            }
        }
        return dp[0][n-1];
        
        /*
        // column by column, faster. but still O(n^4). 
        for (int j = 0; j < n; ++j) {
            int i = j;
            dp[i][j] = s.substring(j, j+1);
            for (int p = 0; p < i; ++p) {
                dp[p][j] = dp[p][j - 1] + dp[i][j];
            }
            for (i = j - 1; i + 1 >= j - i; --i) {
                String sub = s.substring(i + 1, j + 1); // s[i+1..j]
                for (int k = i - (j - i) + 1; k >= 0 && sub.equals(s.substring(k, k + j - i)); k -= j - i) {
                    String str = Integer.toString((j + 1 - k) / (j - i)) + "[" + dp[i+1][j] + "]";
                    if (str.length() < dp[k][j].length()) {
                        dp[k][j] = str;
                        for (int p = 0; p < k; ++p) {
                            if (dp[p][k - 1].length() + str.length() < dp[p][j].length()) {
                                dp[p][j] = dp[p][k - 1] + str;
                            }
                        }
                    }
                }
            }
        }
        */
    }
    
    
    
    /**
     * 312. Burst Balloons
     * 每次扎爆气球，会跟两边邻居相乘 得到值 n[l] * n[i] * n[r]。直到burst所有气球，求出MAX
     * @param nums
     * @return
     * 
     * 这种没有顺序需要搜所有情况的，最好用recursive
     * 每次都可以分个更小的问题，divide & conquer。
     * 
     * !!!! 每次算时取决于没爆的气球。所以要从 last ballon to burst算起。如果没爆的话，如何切分区间
     * 想象最后只剩第k个气球，那么就算nums[-1]*nums[k]*nums[n]..那之前爆完的都要加上这3个乘积结果
     * 
     * dp[i][j]是 气球在[i~j]之间的max sum
     * 1. 找哪个点burst？ for loop里k去burst
     * 2. 对于[i,j]区间来说，每次burst都切三份：
     * 		左边区间已经爆完recur后的max + 中间只剩第k个气球的max(3项相乘) + 右边区间已经burst后recurs后的max
     *       rec(i, k - 1) + (n[i - 1] * n[k] * n[j + 1]) + rec(k + 1, j))
     *       k的左右两边都爆完了..那么现在只剩中间第k个气球没爆. 那么就算 i-1, k, j+1 这3个气球的max(3项相乘)
     *       		
     * 3. 返回[1,n]区间的max
     */
    public int maxCoins(int[] nums) {
        if (nums == null || nums.length == 0)   return 0;
        
        int n = nums.length;
        int[][] dp = new int[n+2][n+2];     //range[i-j]'s max value
        
        // 在头和为放dummy 1, and copy nums[] to arr[]
        int[] arr = new int[n+2];  
        arr[0] = 1;
        arr[n+1] = 1;
        for (int i = 1; i <=n; i++) {
            arr[i] = nums[i-1]; 
        }
        
        return memSearch(arr, dp, 1, n);    //finally return range[1,n] is the range we need
    }
    
    public int memSearch(int[] arr, int[][] dp, int left, int right) {
        if (left > right)           return 0;
        if (dp[left][right] > 0)    return dp[left][right];     //visited
            
        int res = 0;
        //find the max val in range[left, right]
        for (int k = left; k <= right; k++) {
            int midVal = arr[left - 1] * arr[k] * arr[right+ 1];      
            int leftVal = memSearch(arr, dp, left, k - 1);				//divide
            int rightVal = memSearch(arr, dp, k + 1, right);	
            res = Math.max(res, leftVal + midVal + rightVal);		// conquere
        }
        dp[left][right] = res;
        return res;
    }
    
    
    /** 312. Burst Balloons
     * 用普通DP for循环 。思路跟上面一样
     * 因为burst一个气球的话，右边可能分成另一个区间，之后再分这个右区间.. 所以这算是 区间DP问题
     * dp[i][j]表示 在 第i到j [i,j]区间(inclusive)内气球打爆的max值
     * @param nums
     * @return
     */
    public int maxCoinsDP(int[] nums) {
        if (nums == null || nums.length == 0)   return 0;
        
        int n = nums.length;
        int[][] dp = new int[n+2][n+2];     //range[i-j]'s max value
        
        int[] arr = new int[n+2];  
        arr[0] = 1;
        arr[n+1] = 1;
        for (int i = 1; i <=n; i++) {
            arr[i] = nums[i-1]; 
        }
     
        for (int len = 1; len <= n; len++) {      
            for (int i = 1; i <= n - len + 1; i++) {      //left，从第1个开始.。假设len=n, 那i <= 1, 所以要+1
                int j = i + len - 1;                      //第j是包含在内，要-1
                for (int k = i; k <= j; k++) {      // 在 [i~j]里的x
                	int tmp = dp[i][k-1] + arr[i-1] * arr[k] * arr[j+1] + dp[k+1][j];
                    dp[i][j] = Math.max(dp[i][j], tmp);
                }
            }
        }
        return dp[1][n];
    }
    
    
    /** 375. Guess Number Higher or Lower II
     * 二分法猜数字，但是猜错的话需要花x元。给n, 看需要多少钱能保证win
     * @param n
     * @return
     * 这里用dp, 因为都是基于前面的。
     * dp[i][j] 是 在范围[i~j]中猜数所要的min cost.
     *  
     * 从len=2开始，因为len=1时，那就算是自己，猜中自己就cost为0，所要对角线 dp[i][i] = 0.
     * 整个dp的数字是右上角的对角线填数
     * 
     * 在[i,j]每一轮找k这个pivot时，要找local max,这样保证才能有足够的钱赢
     * 从后半段开始找max(dp[i][k - 1], dp[k + 1][j]).. 
     * 		这里k切割完后，会有不同的区间，还是有必要比的，因为可能前面区间[3,4,5], 后面只有[7].
     * 
     * optimize
     * k是从后半部分right开始找，因为要找local max才能保证赢，当然要找大的。前半部分都太小，没必要比
     */
    public int getMoneyAmount(int n) {
        int[][] dp = new int[n + 1][n + 1];
        for (int len = 2; len <= n; len++) {
            for (int i = 1; i <= n - len + 1; i++) {
                int j = i + len - 1;            // i is start, j is end with the len
                int min = Integer.MAX_VALUE;
          //      for (int k = i; k < j; k++) {      //pivot  这个也可以，但是慢点
                // 下面这个是从 后半right 开始找，这样更快，因为后面的数大，肯定比前半部分left大
                for (int k = (i + j) / 2; k < j; k++) {      //pivot
                    int maxCost = k + Math.max(dp[i][k - 1], dp[k + 1][j]);     //local max cost
                    min = Math.min(min, maxCost);                               //global min
                }
                dp[i][j] = min;
            }
        }
        return dp[1][n];
    }
    
    
    /** Stone Game - Lintcode
     * 有一堆石子，把他们两两合并，每次+合并所需要挪的次数，最后找出minimum合并次数的sum
     * @param A
     * @return
     * 需要枚举，搜索不同的合并方法..但是费时，要想到memorize search。需要从大到小的问题拆分，这样才能记忆化搜索
     * 要考虑不同的切分方法
     */
    public int stoneGame(int[] A) {
        int n = A.length;
        
        // initialize
        int[][] f = new int[n][n];		//第i到第j个石子合并到一起的最小花费
        boolean[][] visit = new boolean[n][n];
        
        // preparation 
        int[][] sum = new int[n][n];
        for (int i = 0; i < n; i++) {
            sum[i][i] = A[i];
            for (int j = i + 1; j < n; j++) {
                sum[i][j] = sum[i][j - 1] + A[j];
            }
        }
        
        return search(0, n-1, f, visit, sum);
        
    }
    
    public int search(int l, int r, int[][] f, boolean[][] visit, int[][] sum) {
        if(visit[l][r])
            return f[l][r];
        
        visit[l][r] = true;
        
        if(l == r) {
            return f[l][r];
        }
        
        f[l][r] = Integer.MAX_VALUE;
        // 切分区间，要循环 r-l次
        for (int k = l; k < r; k++) {	//记得还要加上这次 区间合并的sum
            f[l][r] = sum[l][r] + Math.min(f[l][r], search(l, k, f, visit, sum) + search(k + 1, r, f, visit, sum));
        }
        return f[l][r];
    }
    
    
    
    
    /** 221. Maximal Square
     * matrix里有0，1 找到最大的正方形
     * @param matrix
     * @return
     * brute force 做法，看是否为1，while看往下一格(i+1,j+1)（右下对角线）的那一行和一列是否都为1，是就边长++
     * O((mn)^2)
     */
    public int maximalSquare(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return 0;
            
        int m = matrix.length;
        int n = matrix[0].length;
        int max = 0;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == '1') {
                    int len = 1;
                    boolean isSquare = true;
                    while (isSquare && i + len < m && j + len < n) {
                        for (int k = i; k <= i + len; k++) {
                            if (matrix[k][j + len] == '0') {		//后面那列
                                isSquare = false;
                                break;
                            }
                        }
                        for (int k = j; k <= j + len; k++) {		//下面那行
                            if (matrix[i + len][k] == '0') {
                                isSquare = false;
                                break;
                            }
                        }
                        
                        if (isSquare) {
                            len++;
                        }
                    }
                    max = Math.max(max, len);
                }
            }
        }
        return max * max;
    }
        
        
    /** 221. Maximal Square - DP
     * @param matrix
     * @return
     *
     * 用DP[i][j]代表以[i][j]为止最大的正方形边长.. max是边长!!!
     *
     * 1. 初始化.. 当matrix==1时，dp[i][0]或[0][j]=1
     * 2. 要比较3各值，左dp[i-1][j], 上dp[i][j-1]，左上对角dp[i-1][j-1]
     * 		根据3个值中的min边长，来+1.. 因为如果左边和上边都1，但[i-1][j-1]那个角为0，那就不行
     * 为1时，dp(i, j) = min(dp(i−1, j), dp(i−1, j−1), dp(i, j−1))+1.
     * 
     * follow up 如果求对角线是1，其他是0的话，也可以用这方法
     */
    public int maximalSquareDP(char[][] matrix) {
        if (matrix == null || matrix.length == 0)   return 0;
        
        int m = matrix.length;
        int n = matrix[0].length;
        int max = 0;		//边长

        int[][] dp = new int[m + 1][n + 1];

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (matrix[i - 1][j - 1] == '1') {      // [i-1, j-1]
                    dp[i][j] = Math.min(Math.min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1;
                    max = Math.max(max, dp[i][j]);
                }
            }
        }
        
        
        max = 0;
        //==========用一维数组========
        // 因为需要额外用到的只有[i-1][j-1], 所以用一个pre代表就行.
        //每次更新前先把当前d[j]存tmp, 这样下次j+1时，pre就是以前的值
        int[] dp2 = new int[n + 1];   
        int pre = 0;	//the previous left one [i-1][j-1]
        for (int i = 1 ; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                int nextPre = dp2[j];        //在更新前先赋值
                if (matrix[i - 1][j - 1] == '1') {
                                // [i][j-1]      [i-1][j], [i-1][j-1]
                    dp2[j] = Math.min(dp2[j-1], Math.min(dp2[j], pre)) + 1;
                    max = Math.max(max, dp2[j]);
                } else {
                    dp2[j] = 0;		//记得要设0
                }
                pre = nextPre;		//这次的pre是d[j]，下次j+1时, pre就代表了新j的前一列[j-1]
            }
        }
               
        return max * max;
    }
    
    
    

	/** Trapping Rain Water
	 * Given n non-negative integers representing an elevation map where the width of each bar is 1,
	 * compute how much water it is able to trap after raining.
	 * Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6
	 * // 即第i块地方的存水量 = min(第i块左边最高的bar高度, 第i块右边最高的bar的高度) - 第i块地方bar的高度  
    // 例如图中，第5块地方的存水量 = min(2,3)-0 = 2  
    // 2为其左边最高的bar，即第3块地方的bar  
    // 3为其右边最高的bar，即第7块地方的bar，  
    // 0为其自身的bar高度  
	 * @param A
	 * @return
	 */
	public int trap(int[] A) {
        if (A == null || A.length == 0) {
            return 0;
        }
        
        // find the max in the right part，到 i 为止
        int[] right = new int[A.length];
        right[A.length - 1] = A[A.length - 1];
        for (int j = A.length - 2; j >= 0; j--) {
            right[j] = Math.max(right[j], A[j + 1]); 	// DP,right[i]都是存目前最大（包括自己），所以不用max
        }
        
        int sum = 0;
        int trap = 0;
//        int[] left = new int[A.length];   //不用声明数组，直接声明int就行
//        left[0] = A[0];
        int left = A[0];
        for (int i = 0; i < A.length - 1; i++) {
       // 	left[i] = Math.max(left[i - 1], A[i]);
        	left = Math.max(left, A[i]);		// 跟原先比即可
            trap = Math.min(left, right[i]) - A[i];		// 包括left,right包括自己。所以trap最少为0，不会负数
            sum += trap;
        }
        return sum;
    }
	
	
    
    

    /**  Coins in a Line - lintcode
     *  n个coins，2个player，一人一次可以拿1/2个，谁拿到最后的谁win。
     * @param n
     * @return
     * 记忆化搜索，因为每次2种选择，之后又很多可能性，所以要recursive
     * dp[i] = !dfs(i-1) || !dfs(i-2) 表示上一个玩家的结果，需要他false才行
     * 
     * 
     * 这题慢慢分析的话，可以发现只要是3的倍数就一定输，所以只用一句话
     * return n % 3 != 0;
     */
    public boolean firstWillWin(int n) {
        if (n == 0)     return false;
        boolean[] dp = new boolean[n + 1]; 		//现在剩i个硬币，当前取硬币的人能否赢
        boolean[] visit = new boolean[n+1];
        return memSearch(n, dp, visit);
    }
    
    public boolean memSearch(int i, boolean[] dp, boolean[] visit) {
        if (visit[i])   return dp[i];		//可以不用visit[], 那dp[]需要0代表空，1为false，2为true
        
        if (i == 0) {		//现在没coin了，所以当前player输了
            dp[i] = false;
        } else if (i == 1) {    //1 coin left for 1st player,win
            dp[i] = true;
        } else if (i == 2) {
            dp[i] = true;
        } else {
            dp[i] = !memSearch(i-1, dp, visit) || !memSearch(i-2, dp, visit);
        }
        
        visit[i] = true;
        return dp[i];
    }
    
    
    /**
     * Coins in a Line II - lintcode
     * 每个硬币有value，看最后谁拿的多谁赢
     * @param values
     * @return
     * dp[i] 现在还剩i个硬币，现在当前取硬币的人最后最多取硬币价值
     *  i 是所有硬币数目
		sum[i] 是 *后*i个硬币的总和
		dp[i] = sum[i]-min(dp[i-1], dp[i-2])
     */
    public boolean firstWillWinII(int[] values) {
        if (values == null || values.length == 0)    return false;
        
        int n = values.length;
        int[] dp = new int[n + 1];      //
        boolean[] flag = new boolean[n+1];
        
        int total = values[n-1];      //total sum
        int[] sum = new int[n];     //在i时还剩下多少数量
        sum[n-1] = values[n-1];
        for (int i = n - 2; i >= 0; i--) {		//sum[]要从后往前算，不能从前。因为后面dfs时sum[i] - i+1/i+2
        	sum[i] = sum[i+1] + values[i];
        	total += values[i];
        }
        
        int res = memSearch(values, sum, flag, dp, n, 0);
        return res > total / 2;
    }
    
    public int memSearch(int[] v, int[] sum, boolean[] flag, int[] dp, int n, int i) {
        if (flag[i])   return dp[i];
        
        if (i == n) {
            dp[i] = 0;
        } else if (i == n - 1) {    //last 1
            dp[i] = v[i];
        } else if (i == n - 2) {
            dp[i] = v[i] + v[i+1];
        } else {			//剩下的数要减去之前的min, 才能保证自己拿的是最大
            dp[i] = sum[i] - Math.min(memSearch(v, sum, flag, dp, n, i + 1), memSearch(v, sum, flag, dp, n, i + 2));
        }
        flag[i] = true;
        return dp[i];
    }
    
    
    /** Coins in a Line III - lintcode
     * 每次只能从头或尾取一个，看谁的sum多
     * @param values
     * @return
     */
    public boolean firstWillWinIII(int[] values) {
        int n = values.length;
        int[] sum = new int[n + 1];
        for (int i = 1; i <= n; i++) {
        	sum[i] = sum[i-1] + values[i-1];
        }
        
        int[][] dp = new int[n + 1][n + 1];
        boolean[][] visit = new boolean[n + 1][n + 1];

        return search(1, n, sum, dp, visit) > sum[n] / 2;
    }
    
    private int search(int i, int j, int[] sum, int[][] dp, boolean[][] visit){
        if (visit[i][j]) {
            return dp[i][j];
        }
        
        visit[i][j] = true;
        
        if (i == j) {
            return dp[i][j] = sum[j] - sum[i - 1];
        }
        
        //            [i-j]区间的和							取尾								取头
        dp[i][j] = (sum[j] - sum[i-1]) - Math.min(search(i, j - 1, sum, dp, visit), search(i + 1, j, sum, dp, visit));
        
        return dp[i][j];
    }
    
    
    
    
    /** BackPack 背包问题
     * 给串数组A[]，和容量为m的背包，从A里挑size最大的放包里，不能超过m容量
     *
     * bp[i][j]代表 前i个物品，能否拼出size为j的背包里
     * bp[i][j] = bp[i-1][j] || bp[i-1][j - A[i-1]];
     * 		   若这次不加item    这次加item，所以看上次 没有cur item时能否装到j
     * !!! 由于多了一行0和一列0作为初始值，当前item表示成A[i-1]。而不是A[i]
     * 
     * O(m*n)的时间复杂度
     */
    public int backPack(int m, int[] A) {
        if (A == null || A.length == 0 || m == 0)   return 0;
        
        int len = A.length;
        boolean[][] bp = new boolean[len + 1][m + 1];	//记得长度都+1
        bp[0][0] = true;
        
        for (int i = 1; i <= len; i++) {
            for (int j = 0; j <= m; j++) {
            	// 这里 减去现在的物品需要 A[i-1]代表current item
                if (j >= A[i - 1]) {		//记得check容量j是否够
                    bp[i][j] = bp[i-1][j] || bp[i-1][j - A[i-1]];	//选当前item的话
                } else {
                    bp[i][j] = bp[i-1][j];		//这里也把j=0时，初始化为T
                }
            }
        }
        
        // j不一定能把m装满，所以找最大的j 返回
        for (int j = m; j >= 0; j--) {
            if (bp[len][j]) {
                return j;
            }
        }
        return 0;
    }
    
    
    /** BackPack 背包问题  - 从二维数组 转 一维
     * @param m
     * @param A
     * @return
     * 因为只需要用到 bp[i-1][j] 和 bp[i-1][j - A[i-1]]，所以转成一维数组
     * 记得要从后外前!!!!!
     * 倒序：因为状态转移方程是通过 *上一层的前面* 得到的bp[i-1][j - A[i-1]].
     * 如果从左往右的话，那就会影响值, 更新了这层的，变成bp[i][j - A[i-1]] .即使有pre也不行
     * 所以 从后往前，先更新坐标大的，这样计算时 *前面的值*还是保留 *上一层*的
     */
    public int backPack1D(int m, int[] A) {
        if (A == null || m == 0) {
            return 0;
        }

        boolean[] dp = new boolean[m + 1];
        dp[0] = true;
        for (int i = 1; i <= A.length; i++) {
            for (int j = m; j >= 0; j--) {		// !!!!记得从后往前算
                if (j - A[i - 1] >= 0 && dp[j - A[i - 1]]) {
                	dp[j] = true;
                }
            }
        }

        for (int j = m; j >= 0; j--) {
            if (dp[j]) {
                return j;
            }
        }
        return 0;
    }
    
    
    
    /**
     * BackPack 0/1背包问题 - 取一次
     * 给定 n 种物品, 每种物品都有只能取一次. 第 i 个物品的体积为 W[i], 价值为 V[i].
     * 再给定一个容量为 target 的背包. 问可以装入背包的最大价值是多少?
     *
     * j要从后往前 逆序
     * 
     * 初始化：1. 若背包可以不装满，那初始化就0
     * 		  2. 若要刚好装满背包，需要dp[0]=0, 其他都是min_value. 
     * 				什么都不装且价值为0，所以d[0]=0. 后面 容量增加，但是价值都是0，所以要MIN.VALUE
     */
    public int zeroOneBackpack(int target, int[] W , int V[]) {
        int[] dp1 = new int[target + 1];
   //     Arrays.fill(dp1, Integer.MIN_VALUE);
        dp1[0] = 0;
        
        // 一维数组 < v.len , not <= 
        for (int i = 0; i < V.length; i++) {
            for (int j = target; j >= W[i]; j--) {       // 逆序!!!  不用考虑 < W[i], since no changes
                dp1[j] = Math.max(dp1[j], dp1[j - W[i]] + V[i]);
            }
        }
        
        return dp1[target];
    	
    	/* 二维数组.. 
        int[][] dp = new int[V.length + 1][m + 1];
        for (int i = 1; i <= V.length; i++) {
            for (int j = 0; j <= m; j++) {		//!!!!!这是 <= m, 不是w.len.  可以顺序
                if (j >= W[i-1]) {
                    dp[i][j] = Math.max(dp[i-1][j], dp[i-1][j - W[i-1]] + V[i-1]);
                } else {
                    dp[i][j] = dp[i-1][j];
                }
            }
        }        
        return dp[V.length][m];
        */
    }
    
    
    /**
     * BackPack 完全背包问题 - 无限次
     * 给定 n 种物品, 每种物品能用无限次. 第 i 个物品的体积为 W[i], 价值为 V[i].
     * 再给定一个容量为 target 的背包. 问可以装入背包的最大价值是多少?
     *
     * 跟0/1背包很类似，但是内循环从0开始顺序算，而不是逆序
     * 因为每个物品i可以取无限次，所以判断max时那2项 是 当前current的值，而不是 上一层的值
     * 代表容量为j-x 时的max val.那么容量为j时, j-W[i]可以直接用之前算的j-x时的数据
     * 
     */
    public int CompleteBackpack(int target, int[] W , int V[]) {
        int[] dp = new int[target + 1];
   //     Arrays.fill(dp1, Integer.MIN_VALUE);
        dp[0] = 0;
        
        for (int i = 0; i < V.length; i++) {
            for (int j = W[i]; j <= target; j++) {       // 顺序!!!   从W[i]开始
            	dp[j] = Math.max(dp[j], dp[j - W[i]] + V[i]);
            }
        }
        
        return dp[target];
    }
    
    
    
    /** Backpack 多重背包问题
     * 有一个容量为target的背包和N件物品，第i件物品最多有num[i]件，每件物品的重量是weight[i]，收益是V[i]。
     *
     * 跟 完全背包的二维做法差不多
     * http://blog.csdn.net/insistgogo/article/details/11176693
     */
    public int MultiBackpack(int target, int[] W , int V[], int[] nums) {
    	int[][] dp = new int[V.length + 1][target + 1];
    	
        for (int i = 0; i < V.length; i++) {		//有多少种物品, 所以正常0 ~ n-1就行
            for (int j = 0; j <= target; j++) {		//!!!!!这是 <= m, 不是w.len.  可以顺序
            	for (int k = 0; k <= nums[i] && k * W[i] <= j; k++) {
            		dp[i][j] = Math.max(dp[i][j], dp[i-1][j - k * W[i]] + k * V[i]);
            	}
            }
        }      
        return dp[V.length][target];
        
        /* 一维
         * for(int i = 1; i <= n; ++i)
		    for(int j = V; j >= 0; --j)
		        for(int k = 1; k <= c[i] && k * w[i] <= j; ++k)
		            f[j] = max(f[j], f[j - k * w[i]] + k * v[i]);
         */
    }

    
    
    
    /** 322. Coin Change
     * 给coins[] 包含不同value的货币，求amount最少用多少个coins。每个coin可以无限次用
     * coins = [1, 2, 5], amount = 11  return 3 (11 = 5 + 5 + 1)。 如果没的话就-1
     * @param coins
     * @param amount
     * @return
     * 
     * 挺典型的 完全 背包问题 --- 用 VS 不用 这个硬币
     * dp[i]：拼出i的amount 要用到的最少coins数
     * 从小往大思考 bottom up
     * O(n * amount), n为coins数组长度
     */
    public int coinChange(int[] coins, int amount) {
        if (coins == null || coins.length == 0 || amount == 0)  return 0;
        int tmp = amount + 1;
        int[] dp = new int[amount + 1];     
        Arrays.fill(dp, tmp);    //初始化为最大amount+1
        dp[0] = 0;
        
        // dp[i] means the min coins for amount i
        for (int i = 0; i <= amount; i++) {				// for i,j顺序可以调换
            for (int coin : coins) {
                if (i >= coin) { // not use ||  use it,
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] == tmp ? -1 : dp[amount];
        
    }
    
    
    /** 322. Coin Change
     * 比上面的DP慢，这个是 记忆化搜索
     * 从大往小思考，top down
     * @param coins
     * @param amount
     * @return
     */
    public int coinChangeSlower(int[] coins, int amount) {
        if (coins == null || coins.length == 0 || amount == 0)  return 0;
        
        return coinChange(coins, new int[amount + 1], amount);    //memorize search (slower)
    }
    private int coinChange(int[] coins, int[] count, int rest) {
        if (rest < 0)               return -1;      //previous coin > rest amount
        if (rest == 0)              return 0;
        if (count[rest] != 0)     return count[rest];   //初始化是0，有值就!=0,所以直接返回
        
        int min = Integer.MAX_VALUE;
        for (int coin : coins) {
            int res = coinChange(coins, count, rest - coin);    //use this coin
            if (res >= 0 && res < min) {
                min = res + 1;
            }
        }
        
        count[rest] = (min == Integer.MAX_VALUE) ? -1 : min;
        return count[rest];
    }


    /**
     * 518. Coin Change 2
     * 算出number of combinations that make up that amount.
     * amount = 5, coins = [1, 2, 5]
     * 5=5
     * 5=2+2+1
     * 5=2+1+1+1
     * 5=1+1+1+1+1
     * 不算order的..
     *
     * 需要coins在外层for loop，
     * 否则amount在外层的话，会重复计算，ways会double出错
     * 比如只有coins 2, 5. 现在算 amount=7的情况，如果for(amount)在外的话，会算 dp[2] + dp[5] 重复算了
     * 而且.. 求amount=3时，也会出错，其实是 0 ways，但是会返回错误的1 种ways
     */
    public int coinChangeII(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;

        // 注意要coins在外层... 否则amount在外层的话，会重复计算，ways会double出错
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] += dp[i - coin];
            }
        }
        return dp[amount];
    }
    
    
    
    /** 377. Combination Sum IV
     * Given an integer array with all positive numbers and no duplicates, 
     * find the number of possible combinations that add up to a positive integer target.
     * 求所有组合 加起来为target的 组合个数.. 
     * 可以重复用num，如[1,2,3], target=4, 可以(1,1,1,1), (1,1,2), (2,1,1)
     * @param nums
     * @param target
     * @return
     * 一看到个数，就可以想到DP . 跟coin change挺像
     * dp[i]为 如果target总和为i时，有多少组合个数
     * 1. 初始化dp[0]=1
     * 2. for (i=1, <= target). 这代表dp[i]..因为dp都是基于之前的结果，所有要当前dp[i]算完以后，再i++
     * 3. 对于每个target i, 看是否>= 数组里的num.. 这样能拆成更小的.. d[i] += d[i - num]
     * 
     * 当nums[]很少数，target很大时，可以先sort(nums), for里面if(i < num) 直接break，因为后面肯定更大
     */
    public int combinationSumIV(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;          //若nums里有1，那d[1] += d[0], 所以需要d[0]为1

        // 因为1,1,2 & 2,1,1算不同ways..所以可以算重复计算了
        for (int i = 1; i <= target; i++) {     //dp[i]表示target为i. 需要在外层. 不能跟for num调换
            for(int num : nums) {   
            	if (i < num) 			//可以先sort(nums)
                    break;
                dp[i] += dp[i - num];	//只有当target i有大于nums[]里的数，才能拆分成更小的
            }
        }
        return dp[target];
    }
    
    Map<Integer, Integer> cache = new HashMap<>();
    // dfs + nextNum
    public int combinationSum4(int[] nums, int target) {
        if (target < 0)     return 0;
        if (target == 0)    return 1;
        if (cache.containsKey(target))
            return cache.get(target);
            
        int ways = 0;
        for (int num : nums) {
            ways += combinationSum4(nums, target - num);
        }
        cache.put(target, ways);
        return ways;
    }


    /**
     * 416. Partition Equal Subset Sum
     * 看数组能否分成2个sum相等的subset
     *
     * 其实就是找 某个subset能否sum为  sum / 2。
     * 跟coin change那个背包算法一样
     */
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }

        if (sum % 2 != 0)   return false;

        int target = sum / 2;

        int len = nums.length;

        /*
        // 二维
        boolean[][] dp = new boolean[len + 1][target + 1];
        dp[0][0] = true;

        for (int i = 0; i <= len; i++) {
            dp[i][0] = true;
        }

        for (int i = 1; i <= len; i++) {
            for (int j = 1; j <= target; j++) {
                if (j < nums[i - 1]) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]];
                }
            }
        }
        return dp[len][target];
        */

        // 从上面的二维 节省空间
        boolean[] dp = new boolean[target + 1];
        dp[0] = true;

        // target不能在外层，否则变成 里面num的值用的是之前target的，然而这次target不一定能有这个sum
        for (int num : nums) {
            for (int j = target; j >= num; j--) {       // !!!! target倒序，且内层
                dp[j] = dp[j] || dp[j - num];
            }
        }
        return dp[target];


        // return dfsSubSetSum(nums, target, 0, new boolean[len]);
    }

    // 这个超时了.. 结果应该是对的
    private boolean dfsSubSetSum(int[] nums, int sum, int pos, boolean[] used) {
        if (sum == 0)
            return true;

        for (int i = pos; i < nums.length; i++) {
            if (sum - nums[i] < 0 || used[i])
                continue;

            used[i] = true;
            if (dfsSubSetSum(nums, sum - nums[i], i + 1, used)) {
                return true;
            }
            used[i] = false;
        }

        return false;
    }


    /**
     * 698. Partition to K Equal Sum Subsets
     * 看能否分成K个sum一样的subset.. 是上面那题的k 版本
     *
     * 这个就用正常DFS做... 要注意的是，要dfs K轮，所以多一个k这个round参数.. 等round==0才true
     */
    public boolean canPartitionKSubsets(int[] nums, int k) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }

        if (sum % k != 0)   return false;

        return dfsSubSetSum(nums, sum / k, 0, 0, k, new boolean[nums.length]);
    }

    private boolean dfsSubSetSum(int[] nums, int target, int sum, int pos, int round, boolean[] used) {
        if (round == 0)     return true;
        if (sum > target)   return false;
        // 要保证下一轮也true才整体true
        if (sum == target)  return dfsSubSetSum(nums, target, 0, 0, round - 1, used);

        for (int i = pos; i < nums.length; i++) {
            if (used[i])
                continue;

            used[i] = true;
            if (dfsSubSetSum(nums, target, sum + nums[i], i + 1, round, used)) {
                return true;
            }
            used[i] = false;
        }
        return false;
    }
    
    
    
    /** 276. Paint Fence
     * n个post可以paint k种颜色。最多能有2个相邻的颜色相同，也就是1,2可相同，但3不能
     * @param n
     * @param k
     * @return
     * 这题用DP, 因为取决于前面的情况，且返回个数
     * 第3个post要么和1st不同颜色(1,2 diff)，要么跟2nd不同颜色(1,2 same)
     * 算完后，往后挪，第1个就变成第2个，第2变成第3
     * 
     * 也可以用int diff, 和same来表示
     */
    public int numWays(int n, int k) {
        int[] dp = {0, k, k*k, 0};
        if (n <= 2)         
            return dp[n];

        for (int i = 3; i <= n; i++) {
            // 3rd要么和1st不同颜色(1,2 diff)，要么跟2nd不同颜色(1,2 same)
            dp[3] = (dp[1] + dp[2]) * (k - 1);
            dp[1] = dp[2];      //算完这第3个柱子，就往后挪
            dp[2] = dp[3];
        }
        return dp[3];
    }

    // 或者换成普通变量
    public int numWays1(int n, int k) {
        if (n == 0)     return 0;
        if (n == 1)     return k;

        // base case n = 2;
        int diff = k * (k - 1);     // 前两个different，第3个跟第2个一样，所以3和1 diff
        int same = k;               // 前两个same，第3个跟他们不一样
        for (int i = 3; i <= n; i++) {
            int nextSame = diff;
            int nextDiff = (diff + same) * (k - 1);
            same = nextSame;
            diff = nextDiff;
        }
        return same + diff;
    }
    
    
    
    /** 256. Paint House
     * n个房子，有3种颜色给你涂，每个相邻的不能一样颜色
     * @param costs
     * @return
     * 简单的DP。 这次red的话，那就看 min(last蓝, last绿) + 现在red的cost
     */
    public int minCost(int[][] costs) {
        int n = costs.length;
        int[][] dp = new int[n + 1][3];
        for (int i = 1; i <= n; i++) {
            dp[i][0] = costs[i-1][0] + Math.min(dp[i-1][1], dp[i-1][2]);    //choose red for this time 
            dp[i][1] = costs[i-1][1] + Math.min(dp[i-1][0], dp[i-1][2]);     
            dp[i][2] = costs[i-1][2] + Math.min(dp[i-1][0], dp[i-1][1]);   
        }
        
        return Math.min(dp[n][0], Math.min(dp[n][1], dp[n][2]));
     }
    
    // 用O(1)的空间。用变量表示，或者多一行preRow
    public int minCost1(int[][] costs) {
        int n = costs.length;
        if (n == 0)      return 0;
        int lastR = costs[0][0];
        int lastB = costs[0][1];
        int lastG = costs[0][2];
        for (int i = 1; i < n; i++) {
            int curR = costs[i][0] + Math.min(lastB, lastG);
            int curB = costs[i][1] + Math.min(lastR, lastG);
            int curG = costs[i][2] + Math.min(lastR, lastB);
            lastR = curR;
            lastB = curB;
            lastG = curG;
        }
        return Math.min(lastR, Math.min(lastB, lastG));
    }
    
    
    
    /** 265. Paint House II
     * 如果是n个房子，k种颜色
     * @param costs
     * @return
     * 这个用二维DP.. 比较费空间，下面有个更省得
     */
    public int minCostII(int[][] costs) {
        if(costs.length == 0 || costs[0].length == 0)
            return 0;
        int n = costs.length;
        int k = costs[0].length;        // colors
        if (n == 1 && k == 1)   return costs[0][0];
        
        int[][] dp = new int[n + 1][k];
        int min = Integer.MAX_VALUE;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < k; j++) {
                dp[i][j] = costs[i-1][j] + minLastColor(dp, i - 1, j);
            }
        }
        return minLastColor(dp, n, -1);
    }
    
     // search colors  to find min
    public int minLastColor(int[][] dp, int row, int k) {
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < dp[0].length; i++) {    
            if (i == k)     
                continue;
            min = Math.min(min, dp[row][i]);
        }
        return min;
    }
    
    
    
    /** 265. Paint House II
     * 如果是n个房子，k种颜色
     * @param costs
     * @return
     * 注意观察后，其实只需要3个变量来决定 preMin1之前最小的，preMin2之前第二小的。(前i个房子(dp)目前最小的cost)
     * 有可能上一个房子house[i-1]和这个房子house[i]的颜色cost都是最小的。
     * if 这样，那么当前house[i]只能用第二小的颜色，因为相邻的2个房子不能相同的颜色.
     * 
     * 同时也要知道之前的最小color是谁。
     * 
     * 刚开始preMin1和2都初始化为0，这样for里面从0开始也可以算..
     * preMinColor的坐标初始化为-1 就行
     */
    public int minCostII2(int[][] costs) {
        if(costs.length == 0 || costs[0].length == 0)
            return 0;
        int n = costs.length;
        int k = costs[0].length;        // colors
        if (n == 1 && k == 1)   return costs[0][0];
        
        int preMin1 = 0, preMin2 = 0;       //刚开始初始化为0，这样下面第一行时可以是当前值 。 // 上一个house至今的cost
        int preMinColor = -1;           //so we can compare with current color to avoid it

        for (int i = 0; i < n; i++) {
            int min1 = Integer.MAX_VALUE, min2 = Integer.MAX_VALUE;
            int minColor = -1;
            
            for (int j = 0; j < k; j++) {
                int val = costs[i][j] + (j == preMinColor ? preMin2 : preMin1);
                							// 跟之前上一个house涂的minColor一样 那就选第二小的dp[i-1]值
                if (val < min1) {
                    min2 = min1;
                    min1 = val;
                    minColor = k;
                } else if (val < min2) {
                    min2 = val;
                }
            }
            preMinColor = minColor; preMin1 = min1; preMin2 = min2;
        }
        return preMin1;
    }


    /**
     * 688. Knight Probability in Chessboard
     * knight可以走8个方向.. given n * n的矩阵，knight的初始位置[r,c]. 走k 次，看走完以后留在board里的概率是多少..
     * @return
     *
     * 总共可以走 8^k 种可能..
     * 在board里的可能就需要用 DP.. 如果在board里就 + dp的值
     */
    public double knightProbability(int n, int k, int r, int c) {

        int[][] moves = {{1, 2}, {1, -2}, {2, 1}, {2, -1}, {-1, 2}, {-1, -2}, {-2, 1}, {-2, -1}};

        double[][] dp = new double[n][n];

        for (double[] row : dp) {
            Arrays.fill(row, 1);
        }

        for (int step = 0; step < k; step++) {
            double[][] nextDP = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    for (int[] move : moves) {
                        int x = i + move[0];
                        int y = j + move[1];
                        if (isInBoard(x, y, n)) {
                            nextDP[i][j] += dp[x][y];
                        }
                    }
                }
            }
            dp = nextDP;            // next move (K个moves)
        }
        return dp[r][c] / Math.pow(8, k);
    }

    private boolean isInBoard(int x, int y, int len) {
        return x >= 0 && x < len && y >= 0 && y < len;
    }
    
    
    
	
	public static void main(String[] args) {
		dpSolution dp = new dpSolution();
		System.out.println(dp.lengthOfLIS(new int[]{10,9,2,5,3,7,101,18}));
		
		System.out.println("min cut " + dp.minCut("abccddccacb"));
		
		int[] a = {2,3,1,4};
		System.out.println("jump " + dp.jumpSlower(a));
		String s1 = "adeggt";
		String s2 = "dteg";
		System.out.println(dp.minDistance1D(s1, s2));
		
		List<List<Integer>> result = new ArrayList<>();
		List<Integer> row1 = new ArrayList<Integer>();
        row1.add(2);
        result.add(row1);

        List<Integer> row2 = new ArrayList<Integer>();
        row2.add(3);
        row2.add(4);
        result.add(row2);

        List<Integer> row3 = new ArrayList<Integer>();
        row3.add(6);
        row3.add(5);
        row3.add(7);
        result.add(row3);

        List<Integer> row4 = new ArrayList<Integer>();
        row4.add(4);
        row4.add(1);
        row4.add(8);
        row4.add(3);
        result.add(row4);
        
        System.out.println(dp.encode("aabcaabcd"));
        
        Set<String> dict = new HashSet<>();
        dict.add("cat");
        dict.add("cats");
        dict.add("and");
        dict.add("sand");
        dict.add("dog");
        List<String> list = dp.wordBreakIIBetter("catsanddog", dict);
   //     System.out.println(list.toString());
        
        System.out.println(dp.numDecodings("10"));
        String s = "123";
        int i = s.charAt(2) - '0';
        System.out.println(i);
        
        char[][] m = new char[][]{{'1','0','1','0','0'},{'1','0','1','1','1'},{'1','1','1','1','1'},{'1','0','0','1','0'}};
        System.out.println(dp.maximalSquare(m));
        
        int[][] mtr = new int[][]{{1,2,3},{8,2,4},{7,6,5},{10,11,12}};
     //   System.out.println("lics in matrix : " + dp.longestIncreasingContinuousSubsequenceII(mtr));
        
        System.out.println("paint " + dp.minCostII2(new int[][]{{2,3,5,6}, {1,5,2,8}}));

//        int[] nums = {4,3,2,3,5,2,1};
        int[] nums = {-1, -1, -1, 1};
        System.out.println("\n ***** can partition : " + dp.canPartitionKSubsets(nums, 2));
	}
}


