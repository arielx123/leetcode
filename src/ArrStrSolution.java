
import java.util.*;

public class ArrStrSolution {
	
	/**
     * Length of Last Word
	 * "Hello Worlddd", return 7. 注意空格情况
	 * @param s
	 * @return
	 */
	public int lengthOfLastWord(String s) {
        if (s == null || s.length() == 0 ) {
            return 0;
        }
        String[] strs = s.trim().split(" ");	//记得trim()去掉头尾空格
        //"     ".trim() 就消掉所有空格。是""，长度0
        //  没东西 "".split(" ")后长度也为1. 所有下面不会outOfBound
        return strs[strs.length - 1].length();
    }

	public int lengthOfLastWord2(String s) {
		int length = 0;
		char[] chars = s.trim().toCharArray();
		for (int i = s.length() - 1; i >= 0; i--) {
			if (chars[i] == ' ') {
				break;
			} else {
				length++;
			}
		}
		return length;
	}
	
	
	
	/**
     * 389. Find the Difference - 很easy
	 * 给s跟t字符串，t比s多一位，其他都一样，但顺序是乱的，找出这一个
	 * @param s
	 * @param t
	 * @return
	 * 用array存。
	 * 还可以分别算2个string的char code, 最后相减就行
	 * for (int i = 0; i < s.length(); ++i) charCodeS += (int)s.charAt(i);
	 */
	public char findTheDifference(String s, String t) {
        int[] arr = new int[26];
        for (char c : s.toCharArray()) {
            arr[c - 'a']++;
        }
        
        for (char c : t.toCharArray()) {
            if (--arr[c - 'a'] < 0) {
                return c;
            }
        }
        return 0;
    }
	
	
	
	/**
     * 53. Maximum Subarray
	 * Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
	 * 用max 算跟以前sum的大小。 将0, sum相比，负数就置为0 重新算
	 * @param a
	 * @return
	 */
	public int maxSubArray(int[] a) {
		if (a == null || a.length == 0) {
			return 0;
		}
		int max = Integer.MIN_VALUE;
		int sum = 0;
		for (int i = 0; i < a.length; i++) {
			sum += a[i];                // 至少需要一个num
			max = Math.max(max, sum);			// 看以前的大还是新加的
			sum = Math.max(sum, 0);				// 如果是负数，还不如不加，就置为0 重新算。。 顺序不能换
		}
		return max;
		
		 /* Solution 2: 也可以减掉min  - prefix sum.  sum[i~j] = sum[j] - sum[i-1]
        int minSum = 0;
        for (int n : nums) {
            sum += n;
            max = Math.max(max, sum - minSum);
            minSum = Math.min(sum, minSum);
        }
        */
	}
	
	
	/**
     * Maximum Subarray II
	 * 找2个 non-overlapping的subarrays 使和最大
     *
	 * 对于每个i点，找到左边max和右边max, 就能得到答案..
	 * 为了得到left max和right max， 需要有2个数组存
	 * 
	 * 用I的解法，扫两遍
     * 分别顺序和逆序扫max,存到left[]和right[]里。
	 * 最后在每个i点比较一下左右sum。
	 * 用3次for循环扫就行
     *
     * 这个跟 238. Product of Array Except Self  挺像..
     * 也是左右 扫两遍，最后跟左右对比
	 */
	public int maxTwoSubArrays(List<Integer> nums) {
        int n = nums.size();
        
        int[] left = new int[n];
        int[] right = new int[n];
        int max1 = Integer.MIN_VALUE;
        int max2 = Integer.MIN_VALUE;
        int sum = 0;
        
        // left to right
        for (int i = 0; i < n; i++) {
            sum += nums.get(i);
            max1 = Math.max(max1, sum);
            sum = Math.max(sum, 0);
            left[i] = max1;
        }
        
        //right to left 
        sum = 0;
        for (int j = n - 1; j >= 0; j--) {
            sum += nums.get(j);
            max2 = Math.max(max2, sum);
            sum = Math.max(sum, 0);
            right[j] = max2;
        }
        
        int result = Integer.MIN_VALUE;
        // choose left & right
        for (int i = 0; i < n - 1; i++) {
            result = Math.max(result, left[i] + right[i + 1]);
        }
        
        return result;
    }
	
	
	/**
     * Maximum Subarray III
	 * 找到 k个不重叠的 subarray 和最大
	 * @param nums
	 * @param K
	 * @return
	 * 跟K次买卖股票stock 类似
	 * 
	 * d[i][k]代表0->i-1元素中k个subarray的maxsum  (注意不包含元素i)
		d[i][k] = max(d[i][k], d[j][k-1] + max)
		第3层for循环里，是求最后面那段sub sum + 之前dp[k-1]的max
		
		(m = k-1 ~ i-1; max需要单独求，是从元素i-1到m的max subarray, 用求max subarray的方法，需要从后往前算）
	 */
	public int maxSubArrayIII(int[] nums, int K) {
        int n = nums.length;
        int[][] d = new int[n+1][K+1];

        for (int k = 1; k <= K; k++) {      //j <= k
            for (int i = k; i <= n; i++) {      // j <= i <= n, 因为i至少>=k, 这样前面才有足够的分组
                d[i][k] = Integer.MIN_VALUE;
                int max = Integer.MIN_VALUE;
                int localMax = 0; 
                
                // 因为有n+1个，所以j要-1. 也可以j = i; j >= k，后面都用nums[j-1]
                for (int j = i - 1; j >= k - 1; j--) {		//用max subarray的方法，从后往前！！
                    localMax = Math.max(nums[j], nums[j] + localMax);
                    max = Math.max(localMax, max);
                    d[i][k] = Math.max(d[i][k], d[j][k-1] + max);
                }
            }
        }
        
        // return d[n][k];
        
        
        //==========转成一维数组==== 记得i要从后往前==================
        int[] dp = new int[n+1];
        for (int j = 1; j <= K; j++) {      //j <= k
            for (int i = n; i >= j; i--) {      // i should from right to left
                dp[i] = Integer.MIN_VALUE;
                int max = Integer.MIN_VALUE;
                int localMax = 0; 
                
                for (int m = i - 1; m >= j - 1; m--) {
                    localMax = Math.max(nums[m], nums[m] + localMax);
                    max = Math.max(localMax, max);
                    dp[i] = Math.max(dp[i], dp[m] + max);
                }
            }
        }
        
        return dp[n];
    }
	
	
	

    
    /**
     * 121. Best Time to Buy and Sell Stock I - 一次
     * 在arr[] 里找最小的，返回差价最大的. 交易异常（买卖算一次交易）
     * @param prices
     * @return
     * 跟maximum subarray一样  prefix sum.  sum[i~j] = sum[j] - sum[i-1]
     */
    public int maxProfit(int[] prices) {
    	if (prices == null || prices.length == 0) {
    		return 0;
    	}
    	int buy = Integer.MAX_VALUE;
    	int max = 0;
    	for (int price : prices) {
    		buy = Math.min(buy, price);
    		max = Math.max(max, price - buy);		// 差最大（以前max or 现在）
    	}
    	return max;
    }
    
    /**
     * 122. Best Time to Buy and Sell Stock II - 多次
     * 可以操作多次买卖，但buy前要sell
     * 一个个往后移时减前面
     * @param prices
     * @return
     */
    public int maxProfitII(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int profit = 0;
        for (int i = 0; i < prices.length - 1; i++) {
        	if (prices[i+1] > prices[i]) {
                profit += prices[i+1] - prices[i];
            }
        }
        return profit;
    }
    
    
    /**
     * 123. Best Time to Buy and Sell Stock III - 2次
     * 只能有两次买卖，但buy前要sell
     * @param prices
     * @return
     * 跟max subarray sumII很像   （下面有个更好地方法)
     * 
     * 也是分别顺序和逆序循环得出2段max. 在某天i, 看前半段和后半段最大profit 相加
     * i的前半段到i为止只能买，所以求min.  i的后半段到i只能卖，所以求Max
     */
    public int maxProfitIII(int[] prices) {
        if (prices == null || prices.length <= 1) {
            return 0;
        }
        
        int n = prices.length;
        
        // DP from left to right
        int[] left = new int[n];
        left[0] = 0;
        int min = prices[0];
        for (int i = 1; i < n; i++) {
            min = Math.min(min, prices[i]);
            left[i] = Math.max(left[i - 1], prices[i] - min);		//也可以跟max比.left[]存的是当前最大值
        }
        
        int profit = 0;
        
        // DP from right to left 右边那段以当前i开始的max，
        int[] right = new int[n];
        right[n - 1] = 0;			//最后那个只能sell，所以profit是0
        int max = prices[n - 1];		//记得是最后，不是INT_MAX
        	
        for (int i = n - 2; i >= 0 ; i--) {
            max = Math.max(max, prices[i]);		//后面的max - cur
            right[i] = Math.max(right[i + 1], max - prices[i]);		//注意跟right[i+1]比
            
            //合并后的
            profit = Math.max(left[i] + right[i], profit);		
        }
        
        /*//合并到上面的right[]
        for (int i = 0; i < n; i++) {
            profit = Math.max(left[i] + right[i], profit);
        }
        */
        return profit;
    }
    
    
    /**
     * 123. Best Time to Buy and Sell Stock III  2次买卖 - better
     * @param prices
     * @return
     * 这几个数都是buy / sell操作完后 剩下的钱
     * 时间O(n)，空间才O(1)
     */
    public int maxProfitIII2(int[] prices) {
        if (prices == null || prices.length <= 1) {
            return 0;
        }
        
        // profit after these transactions
        int buy1 = Integer.MIN_VALUE;		//one buy
        int sell1 = 0;						//one buy, one sell
        int buy2 = Integer.MIN_VALUE;		//two buys, one sell
        int sell2 = 0;						//two buys, two sells

        for (int price : prices) {
            buy1 = Math.max(buy1, -price);
            sell1 = Math.max(sell1, buy1 + price);
            buy2 = Math.max(buy2, sell1 - price);   //剩了 sell1-price 的钱 after buy twice
            sell2 = Math.max(sell2, buy2 + price);
        }
        
        return sell2;
    }
    
    
    
    /**
     * 188. Best Time to Buy and Sell Stock IV - k次
     * complete at most k transactions.
     * @param k
     * @param prices
     * @return
     * dp[i][j]代表有i次交易(1<= i <= k), 用到前j个的价格(0, n)的prices区间，的最大profit
     * 那么dp[i][j]就分2中情况 a)这次不交易 b)要交易
     * 	if要交易，那比较feasible是用当前price来算，(很难keep住minBuy因为还要知道不同区间）
     * 	所以，就用 buy完后 + 之前交易dp[i-1][j-1]剩下的 preMax来算.. 跟III很像，都是算buy/sell后剩的钱
     * 那么就用preMax，给下一次sell用
     */
    public int maxProfitIV(int k, int[] prices) {
        int n = prices.length;
        if (n <= 1)     return 0;

        if (k >= n / 2)             // 相当于无限制了
            return maxProfitII(prices);
        
        int[][] dp = new int[k + 1][n];
        
        for (int i = 1; i <= k; i++) {			//k 在外层 不能换，这样后面才知道 k-1次的值是多少
            int preMax = -prices[0];
            for (int j = 1; j < n; j++) {
            				// 这次不交易   VS   这次sell + 之前buy了以后剩的max钱
                dp[i][j] = Math.max(dp[i][j-1], prices[j] + preMax);
                preMax = Math.max(preMax, dp[i-1][j-1] - prices[j]);	//i-1次交易得到的max - buy的费用
            }
        }
        
        //====转成一维=======
        
        int[] cur = new int[n];
        for (int i = 1; i <= k; i++) {
            int preMax = -prices[0];
            for (int j = 1; j < n; j++) {
                int temp = cur[j];		//上一次的交易
                cur[j] = Math.max(cur[j-1], prices[j] + preMax);
                preMax = Math.max(preMax, temp - prices[j]);
            }
        }
     //   return cur[n-1];
        
        return dp[k][n - 1];
    }
    
    
    /** 变成2个一维数组。 更容易理解 Better
     */
    public int maxProfitIV2(int k, int[] prices) {
    	int[] buy = new int[k + 1];
        int[] sell = new int[k + 1];
        
        Arrays.fill(buy, Integer.MIN_VALUE);        //初始化buy[]  sell[i]都是0
        
        for (int price : prices) {				    // 顺序不能换. 每天需要之前上一天的交易情况
            for (int i = 1; i <= k; i++) {		    // 如果price在里面，那么第2次交易又会从第一个price算起，那就错了
            	buy[i] = Math.max(buy[i], sell[i-1] - price);
                sell[i] = Math.max(sell[i], buy[i] + price);
            }                   //这的sell[i]是上一次sell[i-1]的结果
        }
        return sell[k];
    }
    
    
    /**
     * 309. Best Time to Buy and Sell Stock with Cooldown
     * as many transactions as you like. After you sell your stock, you cannot buy stock on next day. (ie, cooldown >=1 day)
     * @param prices
     * @return
     */
    public int maxProfitCoolDown(int[] prices) {
        if(prices == null || prices.length <= 1) return 0;
        
        /*
        int[] buys = new int[prices.length];
        int[] sells = new int[prices.length];
        
        // max profit till i ending with BUY. rest的话不买，就看b[i-1]; 买的话要cooldown，要在i-2天或之前sell
        buy[0] = -prices[0];
        buy[1] = Math.max(-prices[0], -prices[1]);
        sell[1] = Math.max(0, prices[1] - prices[0]);
        
        for (int i = 2; i < n; i++) {			//这两个顺序可换... 跟IV很像
	        buys[i] = Math.max(buys[i-1], sell[i-2] - price);   
	        sells[i] = Math.max(sells[i-1], buy[i-1] + price);	//记得是buy[i-1] + price,不是i-2
	    }
    	return sells[n-1];
     * */
        
        int buy = -prices[0];    //buy[i]
        int sell = 0;     //sell[i]
        int lastBuy, lastSell = 0;			//只用lastSell代表sell[i-2]
        for (int price : prices) {
            lastBuy = buy;		
            buy = Math.max(lastBuy, lastSell - price);
            lastSell = sell;			//顺序不能变..	
            sell = Math.max(lastSell, lastBuy + price);
        }
        return sell;
    }
    
	
	
	
	// 求最小和
	public int minSubArray(ArrayList<Integer> nums) {
        int sum = 0;
        int maxSum = 0;
        int min = Integer.MAX_VALUE;
        
        for (int i = 0; i < nums.size(); i++) {
            sum += nums.get(i);
            min = Math.min(min, sum - maxSum);
            maxSum = Math.max(maxSum, sum);
        }
        return min;
	}
	
	
	

    /**
     * 334. Increasing Triplet Subsequence
     * 给unsorted array, 看有没长度为3的increasing subsequence
     * @param nums
     * @return
     * 用min 和middle来跟当前n比较. 
     */
    public boolean increasingTriplet(int[] nums) {
        int min = Integer.MAX_VALUE;
        int middle = Integer.MAX_VALUE;     
        for(int n : nums) {
            if (n <= min) {
                min = n;
            } else if (n <= middle) {
                middle = n;
            } else {            //min < middle < x
                return true;
            }
        }
        return false;
    }


    /**
     * 152. Maximum Product Subarray
     * Find the contiguous subarray within an array (containing at least one number) which has the largest product
     * might < 0
     *
     * 一有负数 n < 0, 就swap(min, max)
     *
     * better
     */
    public int maxProduct1(int[] nums) {
        int min = nums[0];
        int max = nums[0];
        int result = nums[0];

        for (int i = 1; i < nums.length; i++) {
            if (nums[i] < 0) {
                int tmp = min;      // swap
                min = max;
                max = tmp;
            }
            max = Math.max(max * nums[i], nums[i]);
            min = Math.min(min * nums[i], nums[i]);

            result = Math.max(result, max);
        }
        return result;
    }

	/**
     * 152. Maximum Product Subarray
     * 算出pre和cur的max 和 min 乘积
     * 有点像paint houseII.. 留个备选
     */
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        
        int maxPre = nums[0];
        int minPre = nums[0];
        int result = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            int max = Math.max(Math.max(maxPre * nums[i], minPre * nums[i]), nums[i]);
            int min = Math.min(Math.min(maxPre * nums[i], minPre * nums[i]), nums[i]);
            result = Math.max(max, result);
            maxPre = max;
            minPre = min;
        }
        return result;
    }


    /**
     * 628. Maximum Product of Three Numbers - easy
     * @param nums
     * @return
     * 可能为负数，关键找 2个min & 3个max  O(n)
     *
     * naive是直接sort再找乘积 慢
     */
    public int maximumProduct(int[] nums) {
        // 找2个min & 3个max
        int min1 = Integer.MAX_VALUE, min2 = Integer.MAX_VALUE;
        int max1 = Integer.MIN_VALUE, max2 = Integer.MIN_VALUE, max3 = Integer.MIN_VALUE;

        for (int n : nums) {
            if (n < min1) {
                min2 = min1;
                min1 = n;
            } else if (n < min2) {
                min2 = n;
            }

            if (n > max1) {
                max3 = max2;
                max2 = max1;
                max1= n;
            } else if (n > max2) {
                max3 = max2;
                max2= n;
            } else if (n > max3) {
                max3 = n;
            }
        }
        return Math.max(min1 * min2 * max1, max1 * max2 * max3);
    }


    
    
    /** 238. Product of Array Except Self
     * Given an array of n integers where n > 1, nums, return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].
     * 除去i本身的其他乘积
     * @param nums
     * @return
     * 不能用 除法，那就two pass 扫两遍 分别算left和right的值（不包括自己），再相乘
     * 但费空间.. 下面有O(1)空间的更好
     */
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] left = new int[n];
        int[] right = new int[n];
        
        left[0] = 1;
        for (int i = 1; i < n; i++) {
            left[i] = left[i-1] * nums[i-1];
        }
        
        right[n-1] = 1;
        for (int i = n - 2; i >= 0; i--) {
            right[i] = right[i+1] * nums[i+1];
        }
        
        int[] res = new int[n];
        for (int i = 0; i < n; i++) {
            res[i] = left[i] * right[i];
        }
        return res;
    }
    
    
    /** 238. Product of Array Except Self
     * 其实不用另外的left[], right[].. 因为乘积都会放到res, 那么就先正常i++扫，把left sum放到res里
     * 然后从后往前i-- 把int right sum乘到res即可
     * @param nums
     * @return
     */
    public int[] productExceptSelfBetter(int[] nums) {
        if (nums == null || nums.length == 0) {
            return nums;
        }
        
        int[] arr = new int[nums.length];
        arr[0] = 1;
        // multiply the left side til i 从左开始 (不包括i)
        for (int i = 1; i < nums.length; i++) {
            arr[i] = arr[i-1] * nums[i-1];  
        }
        
        int rightSum = 1;
        // multiply from right side 从右再乘
        for (int i = nums.length - 2; i >= 0; i--) {
            rightSum *= nums[i + 1];
            arr[i] *= rightSum;
        }
        
        return arr;
    }
    
    
    /**
     * Subarray Sum as 0
	 * find a subarray where the sum of numbers is zero
	 * Given [-3, 1, 2, -3, 4], return [0, 2] or [1, 3].
	 * @param nums
	 * @return
	 */
	public List<Integer> subarraySum(int[] nums) {
        List<Integer> list = new ArrayList<>();
        
        int n = nums.length;
        
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);      // put the dummy sum 0, index -1
        
        int prefixSum = 0;
        
        for (int i = 0; i < n; i++) {
            prefixSum += nums[i];
            if (map.containsKey(prefixSum)) {
                list.add(map.get(prefixSum) + 1);	// 记得+1
                list.add(i);
                return list;
            } else {
                map.put(prefixSum, i);   
            }
        }
        
        return list;
    }
	
	

    
    /**
     * 325. Maximum Size Subarray Sum Equals k
     * 找出最长的subarray使和为target。  这题可能有正负数，所以不像上面那么容易
     *
     * 利用prefixSum的原理  sum[i~j] =  sum[j] - sum[i - 1] = k
     * 只用sum记录当前的总sum就行，不用另外一个数组..
     * 
     * 由于是equals，所以用HashMap
     * 
     * 跟上面的Subarray Sum as 0 很像
     */
    public int maxSubArrayLen(int[] nums, int k) {
        if (nums == null || nums.length == 0)   return 0;
        
        int ans = 0;
        int sum = 0;
        Map<Integer, Integer> map = new HashMap<>();
        
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (sum == k) {			//可以不用这个，但是初始化要map.put(0,-1). 这样下面的map.contains(sum-k)能包含
                ans = i + 1;
            } else if (map.containsKey(sum - k)) {   //记得是sum-k. sum[b] - sum[a] = k.  [a~b]=k
                ans = Math.max(ans, i - map.get(sum - k));
            }
            
            if (!map.containsKey(sum)) {	//每次都要放，但i尽量左，所以在不contain时才放
                map.put(sum, i);
            }
        }
        
        // ==========更简单========================
        map.put(0, -1);		//但要在这初始化
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (map.containsKey(sum - k)) {   // sum[b] - sum[a] = k.  [a~b]=k
                ans = Math.max(ans, i - map.get(sum - k));
            }
            
            if (!map.containsKey(sum)) {		//不管是否存在sum-k, 都要放sum到map里
                map.put(sum, i);
            }
        }

        return ans;
        
        /*
        // 下面这个是naive方法， O(N^2)
        for (int i = 0; i < nums.length; i++) {
            sum = 0;
            for (int j = i; j < nums.length; j++) {
                sum += nums[j];
                if (sum == k) {
                    ans = Math.max(ans, j - i + 1);
                }
            }
        }
        */
    }


    /**
     * 560. Subarray Sum Equals K
     * find the total number of continuous subarrays whose sum equals to k.  返回结果个数
     *
     * 简单的prefix sum + hashmap
     */
    public int subarraySum(int[] nums, int k) {
        int sum = 0;
        int count = 0;
        Map<Integer, Integer> map = new HashMap<>();        // sum, freq
        map.put(0, 1);

        for (int num : nums) {
            sum += num;
            if (map.containsKey(sum - k)) {
                count += map.get(sum - k);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return count;
    }

    // 不用额外space.. 但慢 O(n^2)  naive
    public int subarraySum2(int[] nums, int k) {
        int count = 0;

        for (int i = 0; i < nums.length; i++) {
            int sum = 0;
            for (int j = i; j < nums.length; j++) {
                sum += nums[j];
                if (sum == k) {
                    count++;
                }
            }
        }
        return count;
    }
    
    
    
    /**
     * 525. Contiguous Array
     * 给一个数组，只有0，1.找出max length使得0, 1个数一样
     * 比如(0,1,1)返回2
     * @param nums
     * @return
     * 这题2种思路，一样都用prefix sum
     * 
     * a. 转换成 0和1的差值diff.. 
     * 需要个数一样，那么差值需要一样，那就找 diff[j] - diff[i] = 0 的情况
     * 
     * b. 或者把0变为-1，然后算sum.。 如果sum为0，就OK
     * 
     * 那么就转化成跟 prefix sum=0一样，用HashMap
     */
    public int findMaxLength(int[] nums) {
        int max = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);         //放初始化diff=0时. 记得放，为了(0,1)这种情况

        // 法一
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i] == 1 ? 1 : -1;				// 如果n[i]为0，那就设成-1，这样和1相加就为0
            if (map.containsKey(sum)) {
                max = Math.max(max, i - map.get(sum));
            } else {
                map.put(sum, i);
            }
        }

        // 法二
        int diff = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) {			//算0， 1个数的差值
                diff++;
            } else {
                diff--;
            }
            if (map.containsKey(diff)) {
                max = Math.max(max, i - map.get(diff));
            } else {
                map.put(diff, i);
            }
        }
        return max;
    }
    
    
    
    /**
     * 523. Continuous Subarray Sum
     * 非负整数的array，找是否有 至少size为2个subarray sum，他们的和为k的倍数
     * k的倍数可为0或者负数
     *
     * 要注意 k==0的情况.. 如果有连续2个0，也是TRUE，因为0*k
     * 
     * 也是求prefix sum..
     * 求k的倍数时，从最接近sum的最大k倍数求起，n*k <= sum，直到k..不能小于k因为前面的sum不为负数
     * 
     * 后面有更优解，更简单
     */
    public boolean checkSubarraySum(int[] nums, int k) {
        if (nums.length <= 1) return false;

        // 如果有连续2个0，那就可以，这样0*k=0
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] == 0 && nums[i + 1] == 0) return true;
        }
        
        if (k == 0) return false;       //没有连续2个0，那么k==0是FALSE
        
        if (k < 0) k = -k;
        
        Set<Integer> set = new HashSet<>();
        int sum = 0;
        set.add(0);
        
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            
            if (i > 0) {
                // Validate from the biggest possible n * k to k
                for (int preSum = (sum / k) * k; preSum >= k; preSum -= k) {
                    if (set.contains(sum - preSum)) 
                        return true;
                }
            }
            
            set.add(sum);
        }
        
        return false;
    }
    
    
    /**
     * 523. Continuous Subarray Sum
     * 很重要一点是 把和为k的倍数 -> (sums - sum[i]) % k == 0.
     * 那么就是 sums % k == sum[i] % k. 那么用map记录之前出现过的prefix sum
     * 
     * 刚开始map.put(0, -1). 因为如果能被k整除，那sum % k = 0, 结果是0. 所以先把0放进去
     * map的val放index，为了满足 map.get(sum) != i不是当前的..
     * 但有个Corner case [0,1,0] 第一个0时就符合，这样不行，需要i-map.get(sum) > 1
     */
    public boolean checkSubarraySumBetter(int[] nums, int k) {
        if (nums.length <= 1)   return false;
        
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);         //如果能 sum % k == 0,说明能被整除，所以先放0
        int sum = 0;

        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (k != 0)
                sum %= k;
            if (map.containsKey(sum)) {
                if (i - map.get(sum) > 1)       //为了[0,1,0],k=0的case
                    return true;
            } else {
                map.put(sum, i);
            }
        }
        return false;
    }
    
    
    /**
     * 209. Minimum Size Subarray Sum
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
        int i = 0, j = 0;
        
        //比较简单的写法。以j为外循环!! 方便
        for (; j < nums.length; j++) {	//也可以while{ sum += n[j++]
            sum += nums[j];
            while (sum >= s) {
                ans = Math.min(ans, j - i + 1);
                sum -= nums[i++];
            }
        }
        
        //=========================另一种写法=====================
        int min = Integer.MAX_VALUE;
        for (; i < nums.length; i++) {
            while (j < nums.length && sum < s) {
                sum += nums[j++];
            }
            // when previous sum >= s even after -n[i], so add ans
            if (sum >= s) {
                min = Math.min(min, j - i);
            }
            sum -= nums[i];     // subtract n[i] to see i can ++
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
     * 56. Merge Intervals
     * Given a collection of intervals, merge all overlapping intervals 从小到大
     * @param intervals
     * @return
     *
     * 首先按照start来sort
     * 然后看是否有重叠 ，有的话if (pre.end >= cur.start) {。 就pre.end = max(cur.end, pre.end)
     * 否则result.add(pre), pre = cur
     */
    public ArrayList<Interval> merge(ArrayList<Interval> intervals) {
        if (intervals == null || intervals.size() <= 1) {
            return intervals;
        }
        
        Collections.sort(intervals, new IntervalComparator());
        
        ArrayList<Interval> result = new ArrayList<Interval>();
        Interval pre = intervals.get(0);
        
        for (int i = 1; i < intervals.size(); i++) {
            Interval cur = intervals.get(i);
            if (cur.start <= pre.end) {                //有重叠
            	pre.end = Math.max(pre.end, cur.end);	//记得要max !!
            } else {
                result.add(pre);                       
                pre = cur;                             //move to next interval
            }
        }
        result.add(pre);                               //add the last one.
        return result;
    }

    /**
     * 56. Merge Intervals
     *
     * 法2：
     * 1. sort start & end..
     * 2. 然后 end往后走，如果start[end+1]比ends小 那就一直往后end++
     * 3. 直到 start[end+1] > ends[end]才停下来，
     * @param intervals
     * @return
     */
    public List<Interval> merge(List<Interval> intervals) {
        List<Interval> result = new ArrayList<>();
        int len = intervals.size();
        int[] starts = new int[len];
        int[] ends = new int[len];

        for (int i = 0; i < len; i++) {
            starts[i] = intervals.get(i).start;
            ends[i] = intervals.get(i).end;
        }

        Arrays.sort(starts);
        Arrays.sort(ends);

        int i = 0;  // start
        int j = 0;  // end

        while (j < len) {       // 直到找到后面的starts > 当前ends, 这样就不需要再merge了
            if (j == len - 1 || ends[j] < starts[j + 1]) {
                result.add(new Interval(starts[i], ends[j]));
                i = j + 1;
            }
            j++;
        }
        return result;
    }
    
    private class IntervalComparator implements Comparator<Interval> {
        public int compare(Interval a, Interval b) {
            return a.start - b.start;
        }
    }
    public class Interval {
        int start;
        int end;
        Interval() { start = 0; end = 0; }
        Interval(int s, int e) { start = s; end = e; }
    }
    

	
	/**
     * 57. Insert Interval
	 * 假如intervals都根据start排好序了，给一个新的interval，merge进去原先的list里
	 * Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9] in as [1,2],[3,10],[12,16].
	 * Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9].
	 * @param intervals
	 * @param newInterval
	 * @return
	 * while 3 次。主要就是中间那个while，要找到newInterval的新值。
	 * 里面每次newInterval要更新，且i++了，再放到result里
	 */
	public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        List<Interval> result = new ArrayList<>();
        
        int n = intervals.size();
        int i = 0;
        while (i < n && intervals.get(i).end < newInterval.start) {
            result.add(intervals.get(i++));
        }
        
        // 有重叠的.. 每次在里面比较就好，比较简洁.. 
        while (i < n && intervals.get(i).start <= newInterval.end) {
            newInterval.start = Math.min(intervals.get(i).start, newInterval.start);
            newInterval.end = Math.max(intervals.get(i).end, newInterval.end);
            i++;
        }
        result.add(newInterval);
        
        while (i < n) {
            result.add(intervals.get(i++));
        }
        return result;
    }

    // 法2：跟merge interval很像.. 只是多了一种判断条件 cur在前
    public List<Interval> insert2(List<Interval> intervals, Interval newInterval) {
        List<Interval> result = new ArrayList<>();

        for (Interval cur : intervals) {
            if (cur.end < newInterval.start) {           // cur在前
                result.add(cur);
            } else if (newInterval.end < cur.start) {    // newInterval
                result.add(newInterval);
                newInterval = cur;
            } else {
                newInterval.start = Math.min(cur.start, newInterval.start);
                newInterval.end = Math.max(cur.end, newInterval.end);
            }
        }
        result.add(newInterval);        // 相当于之前的pre

        return result;
    }

	
	/**
     * 435. Non-overlapping Intervals
	 * find the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.
	 * 找到 remove最少个interval，使剩余的不重叠
	 * @param intervals
	 * @return
	 * 这问题跟 max个重叠的interval 一样 （meeting roomII）
	 * 是个greedy题
	 * 
	 * 主要按照end, 从小到大排。
	 * 如果preEnd <= intervals[i].start 不重叠，那么正常往后走，并更新 preEnd = int[i].end..
	 * 这里的end因为sorted，是最小的，这样后面不重叠的可能性更大
	 * 
	 * 也可以按照start排，然后赋值时尽可能找更小的end，防止overlap后面的
     *
     * 后面的Minimum Number of Arrows to Burst Balloons跟这个一样.
	 */
	public int eraseOverlapIntervals(Interval[] intervals) {
        if (intervals.length == 0)      return 0;
        
        // sort end !!!
        Arrays.sort(intervals, new Comparator<Interval>(){
        	public int compare(Interval a, Interval b) {
                return a.end - b.end;
            }
        });
        
        int remove = 0;
        int preEnd = intervals[0].end;
        for (int i = 1; i < intervals.length; i++) {
            if (preEnd <= intervals[i].start) {
                preEnd = intervals[i].end;      //因为按照end排序，这个end肯定是最小的，这样后面不重叠的可能性更大
            } else {
                remove++;
            }
        }

        // 也可以
        /*

        count = 1;      // 注意是 1  证明 normal情况 no overlap
        if (preEnd <= intervals[i].start) {
            preEnd = intervals[i].end;
            count++;
        }

        最后 return len - count;
         */

        return remove;
    }
	
	
	/**
     * 435. Non-overlapping Intervals
	 * 按照start排，然后赋值时尽可能找更小的end，防止overlap后面的
	 */
	public int eraseOverlapIntervals1(Interval[] intervals) {
        if (intervals.length == 0) {
            return 0;
        }
        // 按start排 
        Arrays.sort(intervals, new Comparator<Interval>(){
        	public int compare(Interval a, Interval b) {
                return a.start - b.start;
            }
        });
        
        int prev = 0, count = 0;
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[prev].end > intervals[i].start) {
                if (intervals[prev].end > intervals[i].end) {
                    prev = i;
                }
                count++;
            } else {
                prev = i;
            }
        }
        return count;
    }


    /**
     * 452. Minimum Number of Arrows to Burst Balloons
     * Find the minimum number of arrows that must be shot to burst all balloons.
     *
     * 其实跟上面的Non-overlapping Intervals一样
     *
     * 有overlap的就不用arrow++
     *
     * 也是按照ends排序，normal情况 no overlap的话 arrows++. 否则继续往前
     *
     * @param points
     * @return
     */
    public int findMinArrowShots(int[][] points) {
        if (points == null || points.length == 0)
            return 0;

        Arrays.sort(points, (i1, i2) -> Integer.compare(i1[1], i2[1]));

        int arrows = 1;
        int preEnd = points[0][1];

        for (int i = 1; i < points.length; i++) {
            if (preEnd < points[i][0]) {
                preEnd = points[i][1];          // normal, no overlap
                arrows++;
            }                               // else, overlap, just use the same arrow
        }
        return arrows;
    }
    
    
    /** 252. Meeting Rooms
     * 给interval的起始时间，看是否一个人能参加所有会议
     * @param intervals
     * @return
     * 考察 Comparator
     */
    public boolean canAttendMeetings(Interval[] intervals) {
        if (intervals == null || intervals.length <= 1)
            return true;

        Arrays.sort(intervals, (i1, i2) -> Integer.compare(i1.start, i2.start));
//        Arrays.sort(intervals, new IntervalComparator());
        
        for (int i = 0; i < intervals.length - 1; i++) {
            if (intervals[i].end > intervals[i+1].start)
                return false;
        }
        return true;
    }
    
    
    /**
     * 253. Meeting Rooms II - 最快最简单
     * 给串interval，看最少需要多少间会议室 （跟同时有最多架airplane一样）
     * @param intervals
     * @return
     * 每当有新的会（所以要sort start），要找最早能available的房间（所以要sort end）
     * 
     * 先把start和end分别排序
     * 之后一一对比starts[]和ends[]数组。
     * a)如果有一个新会start, 但start < end, 表示当前ie还没结束，开会ing，所以room++
     * b)一个新会start，目前 start >= end,证明当前有个会结束了，可以用.. 所以ie++
     *
     * 其实跟复杂的扫描线一个思想，只是用2个数组代替Point对象。而且也不用另外考虑时间一样时先排end的问题
     */
    public int minMeetingRooms(Interval[] intervals) {
        if (intervals == null || intervals.length == 0)
            return 0;
            
        int len = intervals.length;
        int[] starts = new int[len];
        int[] ends = new int[len];
        for (int i = 0; i < len; i++) {
            starts[i] = intervals[i].start;
            ends[i] = intervals[i].end;
        }
        Arrays.sort(starts);
        Arrays.sort(ends);
        
        int rooms = 0;
        int j = 0;
        for (int is = 0; is < len; is++) {
            if (starts[is] < ends[j]) {	//有一个新会start, 但当前ie还没结束，开会ing
                rooms++;
            } else {
                j++;   //no conflict，当前ie < is时间，说明结束了，继续用之前房间
            }
        }
        return rooms;
    }


    /**
     * 253. Meeting Rooms II  - 扫描线 跟airplane一样..
     *
     * 这个问题其实可以转换成： 一个时间点 最多要有几间房.. 跟airplane一样，一个时间点最多有几架飞机..
     *
     * 这种写法的好处是，如果有多个time 都有 max trip或飞机的话，可以在判断max时打印出这些结果。 上一种写法就不能了
     *
     * 注意 用Comparator和lambda sort时其实是用Object，所以不能光用int[] 要用 Integer[]
     */
    public int minMeetingRooms2(Interval[] intervals) {
        if (intervals == null || intervals.length == 0)
            return 0;

        int len = intervals.length;
        // 注意要用Integer，因为Comparator<? Super T>, upper bounds是Object，所以只能是Integer
        Integer[] list = new Integer[len * 2];      // 记住要用Integer而不是int。如果用lambda或者comparator的话需要Integer
        int i = 0;
        for (Interval in : intervals) {
            list[i++] = in.start;
            list[i++] = -in.end;        // end是负数，sort时排在前面。因为一结束就不算它了，所以end <= 0时count-- 是对的

//            list[i++] = new Point2(in.start, 1);   //using
//            list[i++] = new Point2(in.end, -1);    // not use
        }

        // 按照绝对值排序. 如果时间一样，end负数的 排在前面
        Arrays.sort(list, (a, b) -> Math.abs(a) == Math.abs(b) ? a - b : Math.abs(a) - Math.abs(b));

//        Arrays.sort(list, new PointComparator());

        int count = 0, rooms = 0;
        for (int n : list) {
            if (n > 0) {
                count++;
            } else {
                count--;
            }
            rooms = Math.max(rooms, count);
        }
        return rooms;
    }
    
    //也是先sort start，不过用min heap存ends，然后start跟最早结束时间比 （其实是有conflict的房间）最后算heap大小就行
    // 如果 ends.peek < start, 证明结束了，就可以poll出来
    public int minMeetingRooms3(Interval[] intervals) {
        if (intervals == null || intervals.length == 0)
            return 0;
            
        int len = intervals.length;
        
        // Sort the intervals by start time
        Arrays.sort(intervals, new Comparator<Interval>() {
            public int compare(Interval a, Interval b) { return a.start - b.start; }
        });
        
        //use minHeap to track min end time
        PriorityQueue<Integer> ends = new PriorityQueue<Integer>();
        
        ends.offer(intervals[0].end);    //put the 1st end in heap
        
        for (int i = 1; i < len; i++) {
            if (intervals[i].start >= ends.peek()) {
                ends.poll();        //最早结束的end < start, 这房间可用 so pop。下一次heap有最早结束时间
            }
            ends.offer(intervals[i].end);
        }
        return ends.size();
    }

    
    
    
    /**
     * Max Number of Airplanes in the Sky - 扫描线做法
     * 用一串interval表示飞机空中飞的时间段，看同一时间最多有几架
     * @param airplanes
     * @return
     * 跟meeting room的扫描线做法很像
     * 
     * 用扫描线做法.. 先把interval[2,8]拆成[2,true]和[8,false]. 
     * 把这些POINT放到list里，按time排序.. 如果time一样，就先放降落end
     * 然后扫描新的list，TRUE就++，否则--。 由于要算max 所以要另外有个max
     */
    public int countOfAirplanes(List<Interval> airplanes) { 
        List<Point> list = new ArrayList<>(airplanes.size() * 2);
        // put intervals in 2size list for further sort
        for (Interval in : airplanes) {
            list.add(new Point(true, in.start));		//flying
            list.add(new Point(false, in.end));			//不飞，降落
        }
        
        Collections.sort(list, new PointComparator());		//按time排序.. 如果time一样，就先放降落end
        
        int count = 0, ans = 0;
        for (Point p : list) {
            if (p.fly) {
                count++;
            } else {
                count--;
            }
            ans = Math.max(count, ans);
        }
        
        return ans;
    }
    
    
    class Point {
        boolean fly;
        int time;
        int using;
        
        Point(int t, int u) {
            time = t;
            using = u;
        }
        
        Point(boolean f, int t) {
            time = t;
            fly = f;
        }
        
  //      public static Comparator<Point2> pointComparator = new Comparator<Point2>() {
        
    }
    
    class PointComparator implements Comparator<Point> {
        public int compare(Point p1, Point p2) {
            if (p1.time == p2.time) {		//注意如果时间相同，要先排降落end的，否则会多算!!
                return p1.fly ? 1 : -1;
            }
            return p1.time - p2.time;	//从小到大
        }
    }


    /**
     * 986. Interval List Intersections
     * 找两个list的intersect
     * @param A
     * @param B
     * @return
     */
    public Interval[] intervalIntersection(Interval[] A, Interval[] B) {
        List<Interval> result = new ArrayList<>();
        int len1 = A.length;
        int len2 = B.length;
        int i = 0, j = 0;

        while (i < len1 && j < len2) {
            if (A[i].end < B[j].start) {
                i++;
            } else if (B[j].end < A[i].start) {
                j++;
            } else {
                result.add(new Interval(Math.max(A[i].start, B[j].start), Math.min(A[i].end, B[j].end)));
                if (A[i].end < B[j].end) {
                    i++;
                } else {
                    j++;
                }
            }
        }

        return result.toArray(new Interval[result.size()]);
    }

    // 或者直接查  maxLo, minHi.. 看是否  lo <= hi 代表 intersect
    public Interval[] intervalIntersection1(Interval[] A, Interval[] B) {
        List<Interval> ans = new ArrayList();
        int i = 0, j = 0;

        while (i < A.length && j < B.length) {
            int lo = Math.max(A[i].start, B[j].start);
            int hi = Math.min(A[i].end, B[j].end);
            if (lo <= hi) {                             // intersect了
                ans.add(new Interval(lo, hi));
            }

            if (A[i].end < B[j].end)
                i++;
            else
                j++;
        }

        return ans.toArray(new Interval[ans.size()]);
    }

    
    
    /**
     * 218. The Skyline Problem
     * 给一组三元组代表(s,e,h)的楼，找出skyline，返回(s,h)
     * 每次高不一样时，才需要加点到result里
     * @param buildings
     * @return
     *
     * 也是扫描线  --- 遇到interval要想到扫描线
     *
     * 1. 把bld拆成start和end线段组 (pos, height)的heights集合
     *
     * 2. 根据pos来排序，这样后面一个一个个扫
     * 
     * 3. 扫一遍这些pos的points，分start和end的情况
     *   3.1 因为同一个pos有好几个height.. 所以需要一个maxHeap把heights排序，这样才能挑出最高的加到result里
     *   3.2 若start就把height加到heap里，end就remove。 max heap存高
     *   3.3 对比一下 preHeight跟当前maxQ的 curHeight.. 如果不一样，说明状态变了，就加到result
     *
     * 		!!!排序时如果紧邻的话，pos一样，那要start在前，这样heap里有数..
     * 		height用-h代表start，h代表end。若pos一样就return a.h-b.h,覆盖了所有情况！
     *
     * 		
     *  分别offer, remove完后，再统一放result。
     * 		这时要初始push 0 和pre=0，每次操作后看pre和现在heap里的max，若不一样就要放result
     * 		  	初始的0 包括了heap.empty的情况，若中间没楼，height就是0
     * 
     * Note: Priorityqueue的remove是O(n), 会慢。可以用TreeMap或HashMap代替，看下面
     */
    public List<int[]> getSkyline(int[][] buildings) {
    	
    	// 1. 把bld拆成start和end线段组 (pos, height)的heights集合
    	List<int[]> points = new ArrayList<>();
        
        for (int[] bld : buildings) {
            points.add(new int[]{bld[0], -bld[2]});   //left pos(start), -h 
            points.add(new int[]{bld[1], bld[2]});      //end, h
        }
        
        // 2. sort这些线段，按POS(x轴)先后排序
        Collections.sort(points, new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                if (a[0] != b[0]) {         // pos从小到大ascending
                    return a[0] - b[0];
                }
                return a[1] - b[1];         //排height，从矮到高，start先于end
            }                    
        });

        // 专门放高..
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(1, Collections.reverseOrder());
        maxHeap.add(0);         //初始化。若中间没楼，empty，那height就是0
        
        List<int[]> result = new ArrayList<>();
        int preH = 0;
        
        // 3. 扫描这些线段points..start加高，end删掉高.. 如果状态有变（preh != curh)，那就加result 
        for (int[] p : points) {
            if (p[1] < 0)   maxHeap.add(-p[1]);     // start，加height
            else            maxHeap.remove(p[1]);     //end就要删掉那个高
            
            int curH = maxHeap.peek();       //当前max高
            if (preH != curH) {           //说明状态变了
                result.add(new int[] {p[0], curH});         //记得要用现在max height
                preH = curH;
            }
        }
        
        
       // ====================比较麻烦的做法===================================
        // put start & end points respectively in list
        List<Edge> list = new ArrayList<>();
        for (int i = 0; i < buildings.length; i++) {
            list.add(new Edge(buildings[i][0], buildings[i][2], true));
            list.add(new Edge(buildings[i][1], buildings[i][2], false));
        }
        
        Collections.sort(list, new EdgeComparator());
        
        for (Edge e : list) {
            if (e.isStart) {
                if (maxHeap.isEmpty() || maxHeap.peek() < e.h) {
                    result.add(new int[]{e.pos, e.h});
                }
                maxHeap.add(e.h);
            } else {
            	maxHeap.remove(e.h);   //remove the height, mean the bld ends
                if (maxHeap.isEmpty()) {
                	result.add(new int[]{e.pos, 0});     //no overlap, so add (i, 0)
                } else if (maxHeap.peek() < e.h) {		//记得end后, add the intersection
                	result.add(new int[]{e.pos, maxHeap.peek()});     
                }
            }
        }
        return result;
    }
    
    class Edge {
        int pos;
        int h;
        boolean isStart;
        
        public Edge(int i, int height, boolean start) {
            pos = i;
            h = height;
            isStart = start;
        }
    }
    
    // 当同样的pos时, 要保证高的在外层，矮的在里面(start & end)
    class EdgeComparator implements Comparator<Edge> {
        public int compare(Edge a, Edge b) {
            if (a.pos != b.pos)  return compareInt(a.pos, b.pos);
            // else starts with same pos, higher first
            if (a.isStart && b.isStart) return compareInt(b.h, a.h);
            
            //ends with same pos, lower first
            if (!a.isStart && !b.isStart) return compareInt(a.h, b.h);
            
          //若b1, b2同pos一前一后重叠时，需要b2的start first。这样可以先q.add，而不会先Poll导致空了加个错误坐标
            return a.isStart ? -1 : 1;  
        }
        
        int compareInt(int a, int b) {
            return a - b;
        }
    }
    
    
    /** 218. The Skyline Problem
     * @param buildings
     * @return
     * 用TreeMap代替PriorityQueue，这样remove从O(n)变成O(logn)
     * 并且用count记录重复的高，TreeSet不行
     * 
     * HashHeap的remove在查找时可以到O(1)，不过总的也还是O(logn) 比TreeMap稍微快一点，不过难实现
     */
    public List<int[]> getSkylineTreeMap(int[][] buildings) {
        List<int[]> res = new ArrayList<>();
        if (buildings == null || buildings.length == 0)  return res;
        
        int n = buildings.length;
        List<int[]> heights = new ArrayList<>();
        for (int[] b : buildings) {
            heights.add(new int[]{b[0], -b[2]});    //start is -h (negative)
            heights.add(new int[]{b[1], b[2]});     //end is h
        }
        
        //sort
        Collections.sort(heights, new Comparator<int[]>() {
            public int compare(int[] a, int[]b) {
                if (a[0] != b[0]) {
                    return a[0] - b[0];     //compare pos
                }
                return a[1] - b[1];     //pos same, compare height / b's start first before a's end
            }
        });
        
        TreeMap<Integer, Integer> heightMap = new TreeMap<>(Collections.reverseOrder());
        heightMap.put(0,1);
        int prevHeight = 0;
        List<int[]> skyLine = new LinkedList<>();
        for (int[] h: heights) {
            if (h[1] < 0) {
                Integer cnt = heightMap.get(-h[1]);
                cnt = ( cnt == null ) ? 1 : cnt + 1;
                heightMap.put(-h[1], cnt);		//更新count
            } else {
                Integer cnt = heightMap.get(h[1]);
                if (cnt == 1) {
                    heightMap.remove(h[1]);
                } else {
                    heightMap.put(h[1], cnt - 1);
                }
            }
            int currHeight = heightMap.firstKey();
            if (prevHeight != currHeight) {
                skyLine.add(new int[]{h[0], currHeight});
                prevHeight = currHeight;
            }
        }
        return skyLine;
    }



    /**
     * 759. Employee Free Time
     * 从很多employee的schedule list里，找出他们共同的free time
     * @param schedule
     * @return
     *
     * 用PriorityQueue，按照start时间排序.. 如果previous.end < pq.peek.start 就证明有free time
     *
     * 下面更快的，其实不需要用PriorityQueue，只用普通的List排序  就行
     */
    public List<Interval> employeeFreeTime(List<List<Interval>> schedule) {
        List<Interval> result = new ArrayList<>();

        PriorityQueue<Interval> pq = new PriorityQueue<>((a, b) -> a.start - b.start);
        for (List<Interval> employee : schedule) {
            pq.addAll(employee);
        }

        Interval pre = pq.poll();

        while (!pq.isEmpty()) {
            if (pre.end < pq.peek().start) {
                result.add(new Interval(pre.end, pq.peek().start));
                pre = pq.poll();
            } else {
                if (pre.end < pq.peek().end) {
                    pre = pq.peek();
                }
                pq.poll();
            }
        }
        return result;
    }

    // 其实无需 用PriorityQueue，直接用List就行, 反正也是一样的按照start来排序
    public List<Interval> employeeFreeTime1(List<List<Interval>> schedule) {
        List<Interval> result = new ArrayList<>();
        List<Interval> events = new ArrayList<>();
        for (List<Interval> employee : schedule) {
            events.addAll(employee);
        }

        Collections.sort(events, (a, b) -> a.start - b.start);

        Interval pre = events.get(0);

        for (int i = 1; i < events.size(); i++) {
            Interval cur =  events.get(i);

            if (pre.end < cur.start) {
                result.add(new Interval(pre.end, cur.start));
                pre = cur;
            } else {
                if (pre.end < cur.end) {
                    pre = cur;
                }
            }
        }
        return result;
    }


    /**
     * 759. Employee Free Time
     *
     * 扫描线做法，也跟meeting room II 很像
     * @param schedule
     * @return
     */
    public List<Interval> employeeFreeTime2(List<List<Interval>> schedule) {
        List<Integer> events = new ArrayList<>();

        for (List<Interval> employee : schedule) {
            for (Interval in : employee) {
                events.add(in.start);
                events.add(-in.end);
            }
        }

        // 一样时间的话，start在前 end在后.. 这样不会出现[3,3]的假期，因为不算..
        Collections.sort(events, (a, b) -> Math.abs(a) == Math.abs(b) ? b - a : Math.abs(a) - Math.abs(b));

        List<Interval> result = new ArrayList<>();
        int prev = -1;
        int busy_employees = 0;

        for (int event : events) {
            // 先在前面判断.. 因为要prev要更新用当前值.. 有了它，现在的event才能是end
            if (busy_employees == 0 && prev > 0) {
                result.add(new Interval(prev, Math.abs(event)));
            }

            if (event >= 0) {
                busy_employees++;
            } else {
                busy_employees--;
            }

            prev = Math.abs(event);
        }

        return result;
    }

    /**
     * 759. Employee Free Time - 比较复杂.. 代码多
     *
     * 这解法是 merge sort (merge k sorted list) + Interval List Intersections的结合版
     * 两两merge
     */
    public List<Interval> employeeFreeTimeMergeSort(List<List<Interval>> schedule) {
        if (schedule == null || schedule.size() == 0)
            return new ArrayList<>();

        List<Interval> result = splitHelper(schedule, 0, schedule.size() - 1);

        // remove 边界  [-inf, 1], [10, inf] 这些情况
        result.remove(0);
        result.remove(result.size() - 1);

        return result;
    }

    private List<Interval> splitHelper(List<List<Interval>> schedule, int start, int end) {
        if (start >= end) {
            return getHolidays(schedule.get(start));          // 转成holidays的interval
        }

        int mid = start + (end - start) / 2;
        List<Interval> list1 = splitHelper(schedule, start, mid);
        List<Interval> list2 = splitHelper(schedule, mid + 1, end);

        return mergeIntersection(list1, list2);
    }

    private List<Interval> mergeIntersection(List<Interval> l1, List<Interval> l2) {
        List<Interval> result = new ArrayList<>();
        int len1 = l1.size();
        int len2 = l2.size();

        int i = 0, j = 0;

        while (i < len1 && j < len2) {
            Interval in1 = l1.get(i);
            Interval in2 = l2.get(j);

            int lo = Math.max(in1.start, in2.start);
            int hi = Math.min(in1.end, in2.end);

            if (lo < hi) {
                result.add(new Interval(lo, hi));
            }

            if (in1.end < in2.end) {
                i++;
            } else {
                j++;
            }
        }

        return result;
    }

    private List<Interval> getHolidays(List<Interval> list) {
        List<Interval> holidays = new ArrayList<>();
        int len = list.size();

        if (len == 0)
            return holidays;

        holidays.add(new Interval(Integer.MIN_VALUE, list.get(0).start));

        for (int i = 0; i < len - 1; i++) {
            holidays.add(new Interval(list.get(i).end, list.get(i + 1).start));
        }

        holidays.add(new Interval(list.get(len - 1).end, Integer.MAX_VALUE));

        return holidays;
    }
    
    
    
	/**
     * 128. Longest Consecutive Sequence
	 * Given an unsorted array of integers, find the length of the longest consecutive elements sequence.
	 * Given [100, 4, 200, 1, 3, 2],
	 * The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.
	 * @param num
	 * @return
	 * unsorted. 找连续的，那就用hashmap或hashset.
	 * for循环一遍把num放进set去。然后把set里的值扫一遍，i++看是否包含在set里。
	 */
	public int longestConsecutive(int[] num) {
        Set<Integer> set = new HashSet<>();
        for (int n : num) {
            set.add(n);
        }
        
        int max = 0;
        for (int n : num) {		//也可以num
            if (set.remove(n)) {        //包含，顺便删掉
                int pre = n - 1, next = n + 1;
                while (set.remove(pre)) {
                    pre--;
                }
                while (set.remove(next)) {
                    next++;
                }
                max = Math.max(max, next - pre - 1);
            }
        }
        
        /*
        for (int n : set) {		//也可以num
            if (!set.contains(n - 1)) {     //前一个没有，那就n作为起点
                int end = n + 1;        
                while (set.contains(end)) {
                    end++;
                }
                max = Math.max(max, end - n);
            }
        }
        */
        return max;
    }
	
	//用HashMap - one pass
	public int longestConsecutiveMap(int[] nums) {
        if (nums == null || nums.length == 0)   return 0;
        
        int max = 1;
        Map<Integer, Integer> map = new HashMap<>();    // key, boundary的length
        
        for (int n : nums) { 
            if (!map.containsKey(n)) {
                int left = map.containsKey(n - 1) ? map.get(n - 1) : 0;
                int right = map.containsKey(n + 1) ? map.get(n + 1) : 0;
                
                int len = left + right + 1;     // 1 is self
                map.put(n, len);
                max = Math.max(max, len);
                
                map.put(n - left, len);     //update leftmost boundary's length
                map.put(n + right, len);
            }    
        }
        return max;
    }
	
	
	/**
     * Longest Common Prefix
	 * @param strs
	 * @return
	 */
	public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0)
            return "";
        String prefix = strs[0];
        for (int i = 0; i < prefix.length(); i++) {		//用字母比
            for (int j = 1; j < strs.length; j++) {		// 再Word一个个比
            	//记得比长度，否则prefix长的话会越界。i从0开始，所以i==length时就超过一个。刚好能substring
                if (i == strs[j].length() || 		
                        prefix.charAt(i) != strs[j].charAt(i))
                    return prefix.substring(0, i);
            }
        }
        return prefix;
    }


    /**
     * 541. Reverse String II
     * reverse the first k characters for every 2k characters counting from the start of the string.
     * 每2k个，就reverse
     * @param s
     * @param k
     * @return
     */
    public String reverseStrII(String s, int k) {
        char[] ch = s.toCharArray();
        for (int i = 0; i < ch.length; i += 2 * k) {        // i += 2 * k
            reverse(ch, i, Math.min(i + k - 1, ch.length - 1));
        }
        return String.valueOf(ch);
    }


	/**
     * 151. Reverse Words in a String
	 * Given s = "the sky is blue", return "blue is sky the".
	 * @param s
	 * @return
	 * 或者用.split("\\s+") 去掉中间多余空格
	 */
	public String reverseWords(String s) {
        if (s.length() == 0 || s == null) {
            return "";
        }
        String[] arr = s.split(" ");
        String result = "";
        for (int i = arr.length - 1; i >= 0; i--) {         //start from end
            if (!arr[i].equals("") && !arr[i].equals(" ")) {       //in case of " 1"
                result += arr[i] + " ";
            }
        }
        // check if it's "".  remove the last" "
        return result.length() == 0 ? "" : result.substring(0, result.length() - 1);
    }
	
	
	// in place, reverse整个string，再reverse Word
	public void reverseWordsII (char[] s) {
        reverse(s, 0, s.length - 1);        //reverse整串
        
        int j = 0;
        while (j < s.length) {
            int i = j;
            while (j < s.length && s[j] != ' ') {
                j++;
            }
            reverse(s, i, j - 1);       //reverse单词
            j++;            //跳过空格
        }
    }
    
    private void reverse(char[] s, int i, int j) {
        while (i <= j) {
            char tmp = s[i];
            s[i] = s[j];
            s[j] = tmp;
            i++;
            j--;
        }
    }
    
    
    /**
     * 557. Reverse Words in a String III
     * 保留空格原先的位置
     * Input: "Let's take   LeetCode  contest"
	  Output: "s'teL ekat   edoCteeL  tsetnoc"
     * @param s
     * @return
     */
    public String reverseWordsIII(String s) {
        char[] ch = s.toCharArray();
        int start = 0;
        for (int i = 0; i <= s.length(); i++) {
            if (i == s.length() || ch[i] == ' ') {
                reverse(ch, start, i - 1);
                start = i + 1;
            }
        }
        return String.valueOf(ch);
    }
	
	
	/**
     * 189. Rotate array
     * rotate the array to the right by k steps
     *
     * copy到另一个数组  时间和空间复杂度O(n)
     *
     * @param nums
     * @param k
     */
    public void rotateA(int[] nums, int k) {
        int[] a = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            a[(i + k) % nums.length] = nums[i];		//这个arr[]弄完就是rotate完的
        }
        for (int i = 0; i < nums.length; i++) {
            nums[i] = a[i];
        }
    }
    
    
    /**
     * 189. Rotate array
     *  Original List                   : 1 2 3 4 5 6 7
		After reversing all numbers     : 7 6 5 4 3 2 1
		After reversing first k numbers : 5 6 7 4 3 2 1
		After revering last n-k numbers : 5 6 7 1 2 3 4 --> Result
     *
     * 若判断string A 和 B 是否rotate，直接查
     * return A.length() == B.length() && (A + A).contains(B);
     */
    public void rotate(int[] nums, int k) {
        if (nums == null || nums.length == 1 || k == 0) {
            return;
        }
        
        int len = nums.length;
        k = k % len;		//记得取模
        
        reverse(nums, 0, len - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, len - 1);
        
        /*	//太复杂
        if (k == 0) {
            return;
        }
        int[] tmp = new int[k];
        int index = len - k;
        int i = 0;
        while (index < len) {
            tmp[i++] = nums[index++];
        }
        
        i = len - k - 1;	// i是正常序列的最后index
        for (int j = len - 1; j >= 0; j--) {
            if (i >= 0 && i < len) {	// 记得i<len
                nums[j] = nums[i--];
            } else {
                nums[j] = tmp[j];
            }
        }  */
    }
    
    private void reverse(int[] n, int s, int e) {
        while (s < e) {
            int tmp = n[s];
            n[s] = n[e];
            n[e] = tmp;
            s++;
            e--;
        }
    }


    /**
     * 把 ABCDE12345 -> A1B2C3D4E5
     * 逆着merge sort来
     * 中间 C1 C2 C3 C4 变成 C1 C3 C2 C4
     * DE123 -> 123DE
     * @param arr
     * @param left
     * @param right
     */
    public void convert(char[] arr, int left, int right) {
        if (right - left <= 1)	 	// 1或2个数时不用reverse。4个数才要
        return;

        int size = right - left + 1;

        int mid = left + size / 2;
        int leftmid = left + size / 4;
        int rightmid = left + size * 3 / 4;

        // C1 C2 C3 C4 → C1 C3 C2 C4  DE123 -> 123DE
        reverse(arr, leftmid, mid - 1);		// C2 reverse
        reverse(arr, mid, rightmid - 1);		// C3 reverse
        reverse(arr, leftmid, rightmid - 1);		// C2+C3 reverse

        //左边是2组lm-left的长度
        convert(arr, left, left + 2*(leftmid - left) - 1);
        convert(arr, left + 2*(leftmid - left), right);
    }



    /**
     * 242. Valid Anagram
	 * 判断2个String是不是anagram  字母一样但打乱顺序
	 * @param s
	 * @param t
	 * @return
	 */
	public boolean isAnagram(String s, String t) {
        if (s == null || t == null || s.length() != t.length()) {
            return false;
        }
        
        int[] arr = new int[256];
        for (char c : s.toCharArray()) {
            arr[c - 'a']++;
        }
        for (char c : t.toCharArray()) {
            if (--arr[c - 'a'] < 0) {
                return false;
            }
        }
        return true;
        
	}
	
	
	/**
     * 49. Group Anagram
	 * Given an array of strings, group anagrams together. 且要排序
		For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"], 
		Return:		
		[
		  ["ate", "eat","tea"],
		  ["nat","tan"],
		  ["bat"]
		]
	 * 可以把string sort，当成map的key
	 */
	public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> result = new ArrayList<>();  
        if (strs == null || strs.length == 0) {
            return result;
        }
        
        Map<String, List<String>> map = new HashMap<>();
        
        for (String s : strs) {
            String sortedStr = orderStr(s);    // sort str OR get anagram
            if (!map.containsKey(sortedStr)) {
                map.put(sortedStr, new ArrayList<String>());
            }
            map.get(sortedStr).add(s);
        }
        
        return new ArrayList<List<String>>(map.values());
    }
    
    public String orderStr(String s) {
        // 常规做法 sort 稍微慢点
       // char[] chArr = s.toCharArray();
      //  Arrays.sort(chArr);
      //  return String.valueOf(chArr);  // faster than new String(chArr)
        
        
        // 放到arr里记录字母出现的次数来判断是否anagram。 快点
        int[] keyArr = new int[26];			
        //记得用int[]. 虽然char[]也行，但返回来是ASCII的第一位++等，返回来可以是些奇怪的char，反正不是数字
        
        for (int i = 0; i < s.length(); i++) {
            keyArr[s.charAt(i) - 'a']++;
        }
        		
    //    return String.valueOf(keyArr);		//不能用这个，否则打的是地址
        return Arrays.toString(keyArr);
    }
    
    

    /**
     * 249. Group Shifted Strings
     * we can "shift" each of its letter to its successive letter, for example: "abc" -> "bcd".
     * 每个list代表算是shift得到的。
     * given: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"], 
     * 返回 [
			  ["abc","bcd","xyz"],
			  ["az","ba"],
			  ["acef"],
			  ["a","z"]
			]
     * @param strings
     * @return
     * 主要就是算那offset。
     * 每个单词的char都减掉offset使它从a开始，组成的str放map里看是否能有同样的shift list
     * key.append((str.charAt(i) + 26 - shift) % 26);
     * 
     * 不能用int存diff作为可以,可能会不同的长度比如abc, tz这种
     */
    public List<List<String>> groupStrings(String[] strings) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strings) {
            String key = getKey(str);
            
            if (!map.containsKey(key)) {
                map.put(key, new ArrayList<String>());
            }
            map.get(key).add(str);
        }
        return new ArrayList<List<String>>(map.values());
    }
    
    public String getKey(String str) {
        StringBuilder key = new StringBuilder();
        // 算出减掉shift的str. 比如abc. 不能用int存diff作为可以,可能会不同的长度比如abc, tz这种
        int shift = str.charAt(0) - 'a';
        for (int i = 0; i < str.length(); i++) {
         /*   char c = (char) (str.charAt(i) - shift);		//这个可能好理解
            if (c < 'a')    // 可能 hj & bd
                c += 26;
            key.append(c);
            */
            key.append((str.charAt(i) + 26 - shift) % 26);
        }
        return key.toString();
    }
	
	
	/**
     * Anagrams
	 * Given an array of strings, return all groups of strings that are anagrams.
	 * 只打出所有anagram 不用排序
	 * 
	 * 用hashmap看每个单词word是否出现过，但是要先排序
	 * 没出现，就放进map，把i 记录当前index
	 * 出现1次，就放进result，用之前的i 得到strs[i]，然后再i = -1，标成出现2次
	 * 出现2次以上，i == -1 时直接放进result
	 * @param strs
	 * @return
	 */
	public ArrayList<String> anagrams(String[] strs) {
		ArrayList<String> result = new ArrayList<String>();
		HashMap<String, Integer> map= new HashMap<String, Integer>();
		
		for (int i = 0 ; i < strs.length; i++) {
			char[] word = strs[i].toCharArray();		// 每个word放在char[]
			Arrays.sort(word);							// 转成array才能sort
			String sortedW = new String(word);			// 转回string跟map里的word相比
			if (!map.containsKey(sortedW)) {		// 没有的话就加，可以加sorted反正只是对比而已
				map.put(sortedW, i);					// i-只出现一次就是i,方便加到result。2次就标成-1
			} else {
				if (map.get(sortedW) != -1) {			//之前只出现过1次，加这次是2了
					result.add(strs[map.get(sortedW)]);		//把只出现1次的加到结果
					map.put(sortedW, -1);				// 出现2次了, i = -1
				}
				result.add(strs[i]);					//无论出现几次都要加进result，因为已出现过
			}
		}
		return result;
	}
	
	
	/**
     * 跟Group Anagram差不多
	 * Write a method to sort an array of strings so that all the anagrams are next to each other
	 * 用HashMap<String, ArrayList<String>>的结构。String为排序后的str, AL装各种anagrams的变种s
	 * @param arr   
	 */
	public void sortAnagram(String[] arr) {
		HashMap<String, ArrayList<String>> hash = new HashMap<String, ArrayList<String>>();
		// group word by anagram
		for (String s : arr) {
			String key = sortChar(s);
			if(!hash.containsKey(key)) {
				hash.put(key, new ArrayList<String>()); 	// 没出现过就在hash里新建这类anagram
			}
			ArrayList<String> anagrams = hash.get(key);		//有的话先得到这类anagrams
			anagrams.add(s);								//然后添加进去
		}
		
		// convert hashmap to array
		int i = 0;
		for (String key : hash.keySet()) {
			ArrayList<String> anagrams = hash.get(key);	
			if (anagrams.size() < 2) {
				continue;
			} else {						// 怎么才不输出只有一个词但没有anagram的
				for (String str : anagrams) {
					arr[i] = str;
					i++;
				}
			}
			// 怎么确保 只有一个词没有其他anagrams的不出现？
			// 数组后面那些还是放原来的东西，怎么删掉----数组不能改变大小
		}
	}
	public String sortChar(String s) {
		char[] ch = s.toCharArray();
		Arrays.sort(ch);
		return new String(ch);
	}
	
	/** 实现Comparator接口的compare方法，
	 * 在里面先把每个String都排好序sortChar，然后比较compareTo
	 */
	class AnagramComparator implements Comparator<String> {
		// 这插入上面那个sortChar(s)方法
		
		public int compare(String s1, String s2) {
			return sortChar(s1).compareTo(sortChar(s2));
		}
	}
	
	
	/**
     * 268. Missing Number
	 * an array containing n distinct numbers taken from 0, 1, 2, ..., n, 找到missing的那一个数 from the array.
	 * @param nums
	 * @return
	 * 正常的 1,...n的总和是(n + 1) * n / 2
	 * for循环一遍，减去每个num, 剩下的sum就是那个missing number
	 */
	public int missingNumber(int[] nums) {
        int n = nums.length;
        int sum = (n + 1) * n / 2;
        for (int i : nums) {
            sum -= i;
        }
        return sum;
    }
	
	/** 用异或XOR, 不同为1，相同为0.任何数XOR 0都是本身。所以 a^b^b = a
	 *  要想到其实坐标i跟nums[i]相等，除了一个会不同，那么就XOR找到这个不同的
	 *  比如 0,1,3, 坐标为0,1,2. 刚开始res=3,最后找missing 2时，3^2^3 = 2就找到了
	 *  
	 *  另外的思路也是一样，只不过用 i - nums[i]表示。通常情况下减完为0，这就能找到missing one
	 *  还是上面的例子，刚开始res=3. 前面都是0，最后一个数时，3 += 2 - 3 -> 2
	 *  	刚开始res=n, 因为会减掉n(nums[i]), 这样就能找到对应的i，也就是坐标对应缺失的那个数
	 */
	public int missingNumber2(int[] nums) {
        int res = nums.length;
        for (int i = 0; i < nums.length; i++) {
       //     res = res ^ i ^ nums[i];
            res += i - nums[i];			// i在前面，因为要减掉nums[i]。找到i就是找到missing one
        }
        return res;
    }
	
	
	/**
     * 41. First Missing Positive
	 * 找出第一个 没在array中出现的 正数
	 * Given [1,2,0] return 3, and [3,4,-1,1] return 2.
	 *  should run in O(n) time and uses constant space. 所以不能用hashSet
	 * @param A
	 * @return
	 * 用swap。让1,2,3,4按顺序排列。如果找到5,那就放在A[4]那里
	 * 主要判断条件，当***** A[i] != i + 1时，就要swap  *******
	 * 	不过这里还有其他条件要check
	 * 	 a. 负数或0的话 不管
	 * 	 b. 因为要swap(i, A[i] - 1)，所以要确保 A[i] <= n
	 * 	 c. 如果有重复的也不管，所以要判断当前是否跟前一个一样，否则死循环
	 * 
	 * 放完以后就是1,2,3,4按顺序，直接找第一个没出现的整数就行
	 */
	public int firstMissingPositive(int[] A) {
        if (A == null) {
            return 1;
        }
        
        int n = A.length;
        int i = 0;
        while (i < n) {
        	// 正数swap, 负数或0无需swap。且在数组范围内. A[i] == i+1证明已经是对的位置，不用swap
            if (A[i] > 0 && A[i] <= n && A[i] != i + 1 && A[i] != A[A[i] - 1]) {
                swap(A, i, A[i] - 1);			// 要交换的i, A[i]-1位置的不能相等，否则死循环
            } else {
                i++;
            }
        }
   /*  对的。 跟上面一样，不是for里面有while就一定O(n^2).这个while只是换完继续在原位置换，maybe2次. 所以其实每个点只遍历一次
        for (i = 0; i < A.length; i++) { 
        	 while (A[i] > 0 && A[i] <= n && A[i] != (i+1) && A[i] != A[A[i] - 1]) {
                 swap(A, i, A[i] - 1);
             }
        }
   */     
        for (i = 0; i < n; i++) {        	//从头开始遍历，找到第一个不在应在位置上的元素
            if (A[i] != i + 1) {
                return i + 1;
            }
        }

        return n + 1;                // 说明所有元素都在正确的位置，那么只能返回数组长度后的第一个元素了  
    }
	
	
	
	/**
     * 448. Find All Numbers Disappeared in an Array
	 *  integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.
	 *  找出所有缺失的
	 *  Input:[4,3,2,7,8,2,3,1]， Output: [5,6]
	 * @param nums
	 * @return
	 * 解放跟上面一样.. 
	 * 也是把nums[i]和坐标对应变成12345..那么nums[i]跟nums[nums[i] - 1]来调换
	 * 使得[4,3,2,7,8,2,3,1]变成正确顺序[1, 2, 3, 4, 3, 2, 7, 8] 
	 * != i+1的就是missing number
	 * 
	 * 不过这个要换挺多次.. 下面有个换成负数negative的比较快
	 */
	public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> result = new ArrayList<>();
        int n = nums.length;
        int i = 0;
        while (i < n) {
            if (nums[i] <= n && nums[i] != nums[nums[i] - 1]) {
                swap(nums, i, nums[i] - 1);
            } else {
                i++;
            }
        }
        
        for (i = 0; i < n; i++) {        	//从头开始遍历，找到第一个不在应在位置上的元素
            if (nums[i] != i + 1) {
                result.add(i + 1);
            }
        }
        return result;
    }
	
	
	/**
     * 448. Find All Numbers Disappeared in an Array
     * 一个数组  1 ≤ a[i] ≤ n (n = size of array),有的数出现1或2次，有的没出现。找出没出现的
     * @param nums
     * @return
     * 需要in-place，所以不能额外新开一个arr
     * 需要O(n), 所以不能sort
     * 
     * 主要是，把出现的数变成负数negative。第二次扫时如果是正数那就说明没出现过..
     * 那怎么利用上面来实现呢？
     * 因为for (i), i是递增的，那么就以坐标index为 要比的对象，所以是 n[i] = -n[n[i] - 1]。这里的nums[i]-1为坐标
     * 
     * 之前的解法是变成1,2,3,4这么顺序来.. 这个只是把出现过的数n[i]用坐标index表示
     */
    public List<Integer> findDisappearedNumbers1(int[] nums) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
        	// n[i]对应的坐标
            int index = Math.abs(nums[i]) - 1;        //可能第一次变成负数，那2nd也还是要负的所以要abs
            if (nums[index] > 0) {				//之前变过的就不用再变了
                nums[index] = -nums[index];
            }
        }
        
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                list.add(i + 1);        //记得是坐标i.. 如果找duplicate那就是list.add(nums[i])
            }
        }
        return list;
    }
    
    
    /**
     * 442. Find All Duplicates in an Array
     * 一个数组  1 ≤ a[i] ≤ n (n = size of array),有的数出现1或2次. 找出出现2次的
     * @param nums
     * @return
     * 跟上面一样。出现一次是负数，第二次出现如果发现是负数，那就找到了
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
	
    
    private void swap(int[] arr, int a, int b) {
        int tmp = arr[a];
        arr[a] = arr[b];
        arr[b] = tmp;
    }
    
    
    /**
     * Count and Say
     * Sequence按照1, 11, 21, 1211, 111221, ...其实就是 count + num
     * 输入5，输出111221.. 输入8，返回"1113213211"，如果后面连续一个一样，那就count++. 
     * @param n
     * @return
     */
    public String countAndSay(int n) {	// 用char[]和StringBuilder会快点
    	String str = "1";
    	while (--n > 0) {			// --n
    		StringBuilder sb = new StringBuilder();
    		for (int i = 0; i < str.length(); i++) {
    			int count = 1;
    			while ((i + 1) < str.length() && str.charAt(i) == str.charAt(i + 1)) {
    				count++;
    				i++;				
    			}	
    			sb.append(count).append(str.charAt(i));
    		}
    		str = sb.toString();
    	}
    	return str;
    }
    
    // 跟上面差不多，只是把循环里的单独抽出来一个方法，recursive地call。会慢点
    public String countAndSay2(int n) {
        String s = "1";
        while (n > 1) {
            s = getValue(s.toCharArray());
            n--;
        }
        return s;
    }
    
    public String getValue(char[] chars) {
        String str = "";
        
        for (int i = 0; i < chars.length; i++) {
            int count = 1;
            while (i + 1 < chars.length && chars[i] == chars[i + 1]) {
                count++;
                i++;
            } 
            str += Integer.toString(count) + chars[i];
            
        }
        return str;
    }
    
    

    /**
     * 271. Encode and Decode Strings
     * 给串List<String>, encode成一个string. 再通过decode来返回之前的List<String>
     * @param strs
     * @return
     * 跟count and say类似，都是count + str
     * 每个单词变成   length + '/'或者 '*' 比较好 + s   ---> "5/abcde"
     * 
     * 如果不能用 '/' '#'之类的分隔符，那么找出max length 作为固定长度digits
     * "abcd"      ---> "00004abcd"
	 * "aaa.....a" ---> "99999aaa....a" if the string length is 99999
     */
    public String encode(List<String> strs) {
        StringBuilder sb = new StringBuilder();
        for (String s : strs) {
            sb.append(s.length()).append('/').append(s);   // "5/abcde"
        }
        return sb.toString();
    }

    // Decodes a single string to a list of strings.
    public List<String> decode(String s) {
        List<String> list = new ArrayList<>();
        int i = 0;
        while (i < s.length()) {
            int slash = s.indexOf('/', i);      //从i开始的'/''
            int len = Integer.valueOf(s.substring(i, slash));    // str's length
            list.add(s.substring(slash + 1, slash + 1 + len));
            i = slash + 1 + len;
        }
        return list;
    }


    /**
     * 443. String Compression - in place
     * ["a","b","b","b", ... 多个 "b", "c"] => ["a","b","1","1", "c"]  假设10个b
     * 如果只有一个字母的话，不compress
     * @param chars
     * @return
     */
    public int compress(char[] chars) {
        int len = chars.length;
        int idx = 0;
        int i = 0;

        while (i < len) {
            int count = 0;
            char cur = chars[i];
            while (i < len && chars[i] == cur) {
                i++;
                count++;
            }
            chars[idx++] = cur;
            if (count > 1) {
                for (char c : Integer.toString(count).toCharArray()) {
                    chars[idx++] = c;
                }
            }
        }
        return idx;
    }
    
    
	
    /**
     * 169. Majority Element
     * Given an array of size n, find the majority element. 出现超过 n/2次. 假设一定存在
     * @param nums
     * @return
     *
     * Moore voting  http://www.cs.utexas.edu/~moore/best-ideas/mjrty/example.html
     * 相当于 2个不同的数相互抵消.。 最后剩下的就是结果了
     *
     * candidate表示最终num.. 最后是result
     * 每次跟candidate一样的num，那么count++, 否则 不等就count--.. 当count == 0 时，把candidate换成当前的num
     */
    public int majorityElement(int[] nums) {
    	//Moore voting  http://www.cs.utexas.edu/~moore/best-ideas/mjrty/example.html
    	int count = 0;	        // count表示candidate的出现次数
        Integer candidate = null;

        for (int num : nums) {
            if (count == 0) {
                candidate = num;
                count = 1;
            } else {
                if (candidate == num)   count++;
                else                    count--;
            }
        }
        return candidate;
    	
    	 /*
    	 naive 先排序，找中间的元素

        Arrays.sort(nums);
        return nums[nums.length / 2];
        */
    	
    	/*
    	naive   Hashmap, break when次数 > 1/2

        Map<Integer, Integer> map = new HashMap<>();

        for (int val : nums) {
            map.put(val, map.getOrDefault(val, 0) + 1);

            if (map.get(val) > nums.length / 2) {
                return val;
            }
        }        
        return -1;
        */
    }
    
    
    /**
     * 229. Majority Element II   找出现超过 n/3次的
     * 只有0，1，2个 的结果, 超过 n/3的数 最多只能有2个
     * @param nums
     * @return
     */
    public List<Integer> majorityElementII(int[] nums) {
        List<Integer> list = new ArrayList<>();
        // moore voting. use 2 num to track 
        int n1 = 0, n2 = 0, count1 = 0, count2 = 0;

        for (int num : nums) {
            if (num == n1) {                    //记得这个在判断count==0之前
                count1++;
            } else if (num == n2) {
                count2++;
            } else if (count1 == 0) {
                n1 = num;
                count1 = 1;
            } else if (count2 == 0) {
                n2 = num;
                count2 = 1;
            } else {
                count1--;           // 都不是候选人，而是others。但是n1,n2有票，所以 - 1
                count2--;
            }
        }

        // 再次double check看这2个候选人是不是对的..再遍历一次
        count1 = 0;
        count2 = 0;
        for (int num : nums) {
            if (num == n1) {
                count1++;
            } else if (num == n2) {
                count2++;
            }
        }

        List<Integer> result = new ArrayList<>();

        if (count1 > nums.length / 3)   result.add(n1);
        if (count2 > nums.length / 3)   result.add(n2);

        return result;
    }
    
    
    
    
    /**
     * 163. Missing Ranges - easy
     * 输出缺失的ranges。 数组是sorted的
     * [0, 1, 3, 50, 75], lower = 0 and upper = 99, return ["2", "4->49", "51->74", "76->99"].
     * @param nums
     * @param lower
     * @param upper
     * @return
     * 简单.. 注意几点
     * 1. 有可能越界，所以用long
     * 2. pre = lower - 1, 这样如果arr为空时，也可以进入for循环里加结果
     */
    // 这个避免了越界.. 因为无需low-1
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> result = new ArrayList<>();

        for (int num : nums) {
            if (num > lower) {
                if (num > lower + 1) {
                    result.add(lower + "->" + (num - 1));
                } else {
                    result.add(String.valueOf(lower));
                }
            }
            if (num == upper)
                return result;

            lower = num + 1;        // 更新lower
        }

        if (lower <= upper) {
            result.add(lower + (upper > lower ? "->" + upper : ""));
        }
        return result;
    }

    public List<String> findMissingRanges1(int[] nums, int lower, int upper) {
        List<String> result = new ArrayList<>();
        for (int n : nums) {
            if (n == Integer.MIN_VALUE) {
                lower = n + 1;
                continue;
            }
            
            if (lower == n - 1) {
                result.add(String.valueOf(lower));
            } else if (lower < n - 1) {
                result.add(lower + "->" + (n - 1));
            }
            if (n == Integer.MAX_VALUE)     return result;
            
            lower = n + 1;
        }
        
        if (lower == upper)     result.add(String.valueOf(lower));
        else if (lower < upper) result.add(lower + "->" + upper);
        return result;
    }
    
    
    //1. 有可能越界，所以用long
    // 2. pre = lower - 1, 这样如果arr为空时，也可以进入for循环里加结果
    public List<String> findMissingRanges2(int[] nums, int lower, int upper) {
        List<String> list = new ArrayList<>();
        long pre = lower - 1;        // to add corner case when arr empty
        long cur = 0;
        for (int i = 0; i <= nums.length; i++) {
            cur = i < nums.length ? nums[i] : upper + 1;
            if (cur - pre == 2) {
                list.add(String.valueOf(cur - 1));
            } else if (cur - pre > 2) {
                list.add((pre + 1) + "->" + (cur - 1));
            }
            pre = cur;
        }
        return list;
    }
    
    
    /**
     * 228. Summary Ranges
     * Given a sorted integer array without duplicates, return the summary of its ranges.
     * given [0,1,2,4,5,7], return ["0->2","4->5","7"].
     * @param nums
     * @return
     */
    public List<String> summaryRanges(int[] nums) {
        List<String> list = new ArrayList<>();
        if (nums == null || nums.length == 0)   return list;
        if (nums.length == 1) {
            list.add(String.valueOf(nums[0]));
            return list;
        }
        
        for (int i = 0; i < nums.length; i++) {
            int low = nums[i];
            while (i < nums.length - 1 && nums[i] + 1 == nums[i + 1]) {
                i++;
            }
            if (low == nums[i]) { 
                list.add(String.valueOf(low));
            } else {
                list.add(low + "->" + nums[i]);
            }
        }
        return list;
    }
    
    
    
    /**
     * 303. Range Sum Query - Immutable
     * 简单的算prefix sum
     */
    public int sumRange(int[] nums, int x, int y) {
    	int[] sums = new int[nums.length + 1];
        
        for (int i = 0; i < nums.length; i++) {
            sums[i + 1] = sums[i] + nums[i];
        }

        return sums[y + 1] - sums[x];
    }
    
    
    
    /**
     * 370. Range Addition
     * 长度为length的数组，初始都是0.
     * 有k次update，需要在[i,j]区间加val。看最后这个数组变成怎样. [startIndex, endIndex, inc]
     * @param length
     * @param updates
     * @return
     * naive方法大概是 O(n*k). 太慢.. 需要改进
     * 
     * 这个trick在于，每次update只更新start地方和end..来区分要+val还是-val..
     * 最后循环一遍这个数组，后面的加前面的就行，像prefix sum一样arr[i] += arr[i - 1];
     * 
     * 在start位置+val,那后面的也都会+val.
     * 在a[end + 1]位置 -val, 证明结束，从这开始后面都-val,也就是跟之前的+val抵消成0
     */
    public int[] getModifiedArray(int length, int[][] updates) {
        int[] arr = new int[length];
        
        // 只在start处+val, end+1的地方-val
        for (int[] u : updates) {
            int start = u[0];
            int end = u[1];
            int val = u[2];
            arr[start] += val;
            if (end < length - 1)
                arr[end + 1] -= val;    // a[end+1]的位置减去val，说明结束
        }
        
        //后面的加前面的sum就行 (range sum那样)
        for (int i = 1; i < length; i++) {
            arr[i] += arr[i - 1];
        }
        
        return arr;
    }


    /**
     * 598. Range Addition II - easy
     * 每次给的 operation是 以 operation比如[2,2]为结束点的矩形面积+1.
     * count and return the number of maximum integers in the matrix after performing all the operations.
     * m = 3, n = 3
     * operations = [[2,2],[3,3]]
     *
     * 简单的问题，就是找所有操作矩形的重叠部分。因为矩形的起点一致，所以分别找最小的宽高就行了。
     */
    public int maxCount(int m, int n, int[][] ops) {
        // all ops are rectangle, overlap.. just to find the min
        for (int[] op : ops) {
            m = Math.min(m, op[0]);
            n = Math.min(n, op[1]);
        }
        return m * n;
    }
    
     

    /**
     * Count of Range Sum - 都是正数
     * 给nums[], 找出 lower <= subarray sum <= upper 符合的个数
     *
     * 重点是prefix sum，用二分法找range
     * lower <= S[i] - S[k] <= upper 已知l, u, S[i]，要求S[k] 
     * 所以 sum[i] - upper <= S[k] 是left
     * sum[i] - lower >= S[k] 是right
     * 用find()方法找到S[k]值对应的index k
     */
    public int countRangeSumSimpler(int[] nums, int lower, int upper) {
    	int len = nums.length;
    	int[] sum = new int[len];
    	sum[0] = nums[0];
    	for (int i = 1; i < len; i++) {
    		sum[i] = sum[i-1] + nums[i];
    	}
    	
    	int count = 0;
    	for (int i = 0; i <len; ++i) {
    		if (lower <= sum[i] && sum[i] <= upper) {
    			count++;
    		}
    		int left = find(sum, len, sum[i] - upper);
    		int right = find(sum, len, sum[i] - lower + 1);
    		count += right - left;
    	}
    	
    	return count;
    }
    
    public int find(int[] A, int len, int val) {
    	if (A[len - 1] < val) {
    		return len;
    	}
    	if (val < A[0]) {
    		return 0;
    	}
    	
    	int index = 0;
    	int l = 0, r = len - 1;
    	while (l <= r) {
    		int mid = (l + r) / 2;
    		if (A[mid] >= val) {
    			index = mid;
    			r = mid - 1;
    		} else {
    			l = mid + 1;
    		}
    	}
    	
    	return index;
    }
    
    
    /**
     * Count of Range Sum - 都是正数
     *
     * 用two pointer来找range （取代上面的二分法）
     * 这种是O(n)
     */
    public int countRangeSumSimpler2(int[] nums, int lower, int upper) {
    	int len = nums.length;
    	int[] sum = new int[len];
    	sum[0] = nums[0];
    	for (int i = 1; i < len; i++) {
    		sum[i] = sum[i-1] + nums[i];
    	}
    	
    	int count = 0;
    	int j = 0, k = 0;
    	for (int i = 0; i < len; ++i) {
    		if (lower <= sum[i] && sum[i] <= upper) {		//not sure if need this/?
    			count++;
    		}
    		//用two pointer来找range （取代上面的二分法）
    		while (j < len && sum[j] - sum[i] < lower) {
    			j++;		// 找left
    		}
    		while (k < len && sum[k] - sum[i] < upper) {
    			k++;
    		}
    		count += k - j;
    	}
    	
    	return count;
    }
    
    
    
    /**
     * 327. Count of Range Sum - 有正负数 - naive O(n^2)
     * 求prefix sum，然后for 2层。。 
     */
    public int countRangeSumNaive(int[] nums, int lower, int upper) {
        int n = nums.length;
        long[] sums = new long[n + 1];
        for (int i = 0; i < n; ++i)
            sums[i + 1] = sums[i] + nums[i];
        
        int ans = 0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j <= n; ++j)
                if (sums[j] - sums[i] >= lower && sums[j] - sums[i] <= upper)
                    ans++;
        return ans;
    }
    
    
    
    /**
     * 327. Count of Range Sum - 有正负数
     * 给nums[], 找出 lower <= subarray sum <= upper 符合的个数
     *
     * 利用merge sort方法，因为左右半段都是sorted，那么就能用prefix sum, 右边sum[j] - 左sum[i] 判断跟lower, upper情况
     * 就变成上面 都是正数时，双指针的类似解法
     * 归并排序中有一步调用 merge 函数，将有序的左数组和右数组进行合并，而这时的右数组中的任一元素在 sum 数组中的位置正是在左数组任一元素之后！
     * 
     * 普通的sort会改变index，但是merge sort还保留着原先的index
     * 先一直二分，直到只剩1个数，就可以返回0
     * 
     * 1. 找mid, 然后recursion算(start, mid) + (mid+1, end) 的count，作为之前得到的结果
     * 
     * 2. 对于当前，要开始merge，并且算这次的count
     * 	  2.1 考虑了 右边s[j]-s[i](左) 的情况。count += 右边 k - j (符合lower，upper）
     *      因为merge时，左右半段(4, 4)已经sorted了，已经在上一轮找完 左半段(2,2)的s[j]-s[i]的情况。
     *      所以在这一轮 找的是 i在左边，j/k在右边 的sum差。
     *      
     *    2.2 普通的merge sort.. 记得要收尾，并且copy merged[]回 sum.
     *    
     * 3. return count
     */
    public int countRangeSum(int[] nums, int lower, int upper) {
        if (nums == null || nums.length == 0 || lower > upper) return 0;

        long[] sums = new long[nums.length + 1];	// len + 1 	
      //  sums[0] = nums[0];
        for (int i = 1; i < sums.length; i++) {		//记得是sums.len
            sums[i] = sums[i-1] + nums[i - 1];
        }
        
        return countWhileMergeSort(sums, lower, upper, 0, sums.length - 1);
    }
    
    // 如果想要sums.lengh = nums.len, 不加1 的话，下面这方法最开始的判断改成
    // if (start == end)   return lower <= sums[start] && sums[start] <= upper ? 1 : 0;
    private int countWhileMergeSort(long[] sums, int lower, int upper, int start, int end) {
        if (start == end) 
            return 0;
 //       if (start == end)   return lower <= sums[start] && sums[start] <= upper ? 1 : 0;
        
        int mid = start + (end - start) / 2;
        int count = countWhileMergeSort(sums, lower, upper, start, mid)
                  + countWhileMergeSort(sums, lower, upper, mid + 1, end);
        
        long[] merged = new long[end - start + 1];
        int j = mid + 1, k = mid + 1;		// 大的右半段, 用于count
        
        int index = 0;      // index for merged[], 注意这里的index从0开始，因为是新建的数组。而不是从start开始
        int right = mid + 1;    //用于普通的merge sort, compare left & right to sort
        
        // compare 左半段 VS 右半段, 算符合区间的count++ & sort 
        for (int i = start; i <= mid; i++) {        	//i是比较小的左半段
            while (j <= end && sums[j] - sums[i] < lower)   j++;
            while (k <= end && sums[k] - sums[i] <= upper)   k++;       
            count += k - j;         //upper & lower proceed one step, fine for next i
            
            // 普通的merge sort里的 compare left & right to sort
            while (right <= end && sums[i] > sums[right]) 	merged[index++] = sums[right++];
            merged[index++] = sums[i];      //remember to add left(i). no need i++ cause for{} will +1
            
        }
        while (right <= end)    merged[index++] = sums[right++];	// deal with rest of right part （普通merge sort需要的收尾部分）
        
        System.arraycopy(merged, 0, sums, start, merged.length);	//把排好序的merged数组copy回sums,给下一步用
        //这个arraycopy相当于下面这个
//        for (int t = 0; t < merged.length; t++) {
//            sums[start + t] = merged[t];
//        }
        
        return count;
    }
    
    
    
    /** 493. Reverse Pairs
     *  important reverse pair if i < j and nums[i] > 2*nums[j]. 返回pairs的对数
     *  Input: [1,3,2,3,1]，返回2. 因为那两个3和最后的1
     * @param nums
     * @return
     * 跟上面很像，也是mergesort
     * 
     * 不过要先处理count，再merge排序.. 这样前面还有可能 n[i] > 2*n[j].. 否则排序后就没可能了因为排完就从小到大
     * 
     * trick 判断nums[i] / 2.0 > nums[tmp]不会越界。否则要long. -> n[i] > 2L * n[j]
     */
    public int reversePairs(int[] nums) {
        return pairsMergeSort(nums, 0, nums.length - 1);
    }
    
    public int pairsMergeSort(int[] nums, int start, int end) {
        if (start >= end)   return 0;
        
        int mid = (start + end) / 2;
        int count = pairsMergeSort(nums, start, mid) + pairsMergeSort(nums, mid + 1, end);
        
        int[] merged = new int[end - start + 1];
        int j = mid + 1, tmp = mid + 1;
        int idx = 0;
        for (int i = start; i <= mid; i++) {
            while (tmp <= end && nums[i] / 2.0 > nums[tmp]) {       //这样不会越界，否则就n[i] > 2L * n[j]
                tmp++;
            } 
            count += tmp - (mid + 1);
            
            // 普通merge
            while (j <= end && nums[i] > nums[j])      merged[idx++] = nums[j++];
            merged[idx++] = nums[i];
        }
        while (j <= end) {
            merged[idx++] = nums[j++];
        }
        
        System.arraycopy(merged, 0, nums, start, merged.length);
        
        return count;
    }
    
    
    /** 315. Count of Smaller Numbers After Self
     * 找出右边小于自己的个数 。Given nums = [5, 2, 6, 1]，返回 Return the array [2, 1, 1, 0].
     * @param nums
     * @return
     * 这题也是用merge sort，跟上面的类似
     * 每次merge时，右边 > 左边的话，需要放在左前面，这时顺便count++ 就行了!!
     * The smaller numbers on the right of a number are exactly those that jump from its right to its left during a stable sort. 
     *
     * 这里要额外用pos[i] = i 来记录index位置, 否则如果2个5重复就分不清是哪个.
     *  PS: 这里需要得出每个n[i]的smaller count,所以需要每个pos... 
     *  	而range count和reverse pair只需要总的count，所以不需要pos[]来表示每个
     *  
     * 然后merge后也是copy到pos[]数组。
     * 
     * 比如		 5,2,6,5,1. 
     * 
     * 排序后pos  4,1,0,3,2  \  这两个就是 排完后的结果.. 但是都以pos[i]表示
     * num排好序  1,2,5,5,6  /
     * 
     */
    public List<Integer> countSmaller(int[] nums) {  
        int len = nums.length;
        int[] smaller = new int[len];  
        int[] pos = new int[len];  			//要记录index, 否则如果2个5重复就分不清是哪个.
        for (int i = 0; i< len; i++) {		//因为需要写出 每个n[i] 的smaller count，而不是总的，所以需要每个pos
            pos[i] = i;  
        }
        mergeSort(nums, smaller, pos, 0, len - 1); 
        
        List<Integer> result = new ArrayList<>();  
        for (int i = 0; i < len; i++) {
            result.add(smaller[i]);  
        }
        
        return result;  
    }  
    
    private void mergeSort(int[] nums, int[] smaller, int[] pos, int start, int end) {
        if (start >= end)   return;
        
        int mid = start + (end - start) / 2;
        mergeSort(nums, smaller, pos, start, mid); 
        mergeSort(nums, smaller, pos, mid + 1, end);
        
        int[] merged = new int[end - start + 1];	//注意这个merged[]放的是排好序的index, pos[]的i
        int j = mid + 1;
        int index = 0;
        int jump = 0;   //count the smaller number that jump from right to left
        
        for (int i = start; i <= mid; i++) {        //for i前半段
            while (j <= end && nums[pos[i]] > nums[pos[j]]) {
                jump++;                    					 //后半段小的话
                merged[index++] = pos[j++];
            }
            smaller[pos[i]] += jump;			//else n[i]小，那就把前面算的这次jump放进smaller
            merged[index++] = pos[i];		//i会自己++
        }
        while (j <= end) {						//记得 剩下的j段
            merged[index++] = pos[j++];
        }
        
        
        // copy merged to pos   也可以用System.arraycopy(merged, 0, pos, start, merged.length);
        for (int t = 0; t < merged.length; t++) {
            pos[start + t] = merged[t];
        }
    }
    
    
    
    /**
     * 315. Count of Smaller Numbers After Self
     *
     * 用一个sorted list, 表示已经sort好的..
     * nums每个num在这个sorted里找应该插入的值，那就知道前面有多少个小于自己的值
     * 找插入位置时，因为list已经是sorted的了，那么binary search
     *
     * 这题找 smaller after self, 那么从后面开始
     *
     * 从最后面往前扫，另外维护一个新数组sorted来放 扫过了的数 排好序的样子
     * 比如 5,2,6,5,1.，先从1开始，然后放sorted里，没有比他更小的.. 
     * 后面那个5，想办法插入sorted里，变成1,5. 这样index是1， 说明前面有几个数比他小（1个）
     * 2的时候，前面只有一个1比它小，所以也是1个..
     * 
     * 在插入sorted时，要排好序，才能知道前面有几个比它小。然后可以binary search变成log(n)的查找方式
     *
     *
     * 有点像LIS 或者 俄罗斯套娃 russian dolls.. 也是 在排好序的list里插入num
     */
    public List<Integer> countSmallerBS(int[] nums) {
        int len = nums.length;
        List<Integer> sorted = new ArrayList<>();
        Integer[] smaller = new Integer[len];       // 用Integer[]最后才能Arrays.asList(..)

        for (int i = len - 1; i >= 0; i--) {
            int index = findIndex(sorted, nums[i]);
//           int index = Collections.binarySearch(sorted, nums[i]);			//有些test case没通过
//           if (index < 0)	index = ~ index;
            smaller[i] = index;     //后面扫过的有多少个在自己前面比较小的
            sorted.add(index, nums[i]);
        }

        return Arrays.asList(smaller);
    }
	
    
    public int findIndex(List<Integer> sorted, int target) {
        int i = 0, j = sorted.size() - 1;
        while (i <= j) {
            int mid = (i + j) / 2;
            if (sorted.get(mid) < target) {
                i = mid + 1;
            } else {
                j = mid - 1;
            }
        }
        return i;
    }
    


	/**
     * 88. Merge Sorted Array  2 arrays
	 * 合并。假设A有enough space >= m+n
     *
	 * 从后往前，先放大的.. 这样能先把empty的填好
	 */
	public void merge(int nums1[], int m, int nums2[], int n) {
		int i = m - 1, j = n - 1;
        while (i >= 0 && j >= 0) {
            if (nums1[i] >= nums2[j]) {
                nums1[i + j + 1] = nums1[i];
                i--;
            } else {
                nums1[i + j + 1] = nums2[j];
                j--;
            }
        }
        while (j >= 0) {			// 最后检查B[n].. A空了就不用会自动有
            nums1[j] = nums2[j];
            j--;
        }
    }


    /**
     * Merge sort
     * @param nums
     * @return
     */
    public int[] sortArray(int[] nums) {
        mergeSort(nums, 0, nums.length - 1);

        return nums;
    }

    private void mergeSort(int[] nums, int start, int end) {
        if (start >= end)
            return;

        int mid = (start + end) / 2;

        mergeSort(nums, start, mid);            // 分成两段
        mergeSort(nums, mid + 1, end);

        merge(nums, start, mid, mid + 1, end);      // merge
    }

    private void merge(int[] nums, int start1, int end1, int start2, int end2) {
        int lo = start1;
        int hi = end2;

        int[] newArr = new int[end1 - start1 + 1 + end2 - start2 + 1];
        int idx = 0;

        while (start1 <= end1 && start2 <= end2) {
            if (nums[start1] < nums[start2]) {
                newArr[idx++] = nums[start1++];
            } else {
                newArr[idx++] = nums[start2++];
            }
        }

        // 最后可能 前部分 或者 后部分没排完，那就补上
        while (start1 <= end1) {
            newArr[idx++] = nums[start1++];
        }

        while (start2 <= end2) {
            newArr[idx++] = nums[start2++];
        }

        // 把排序好的copy回原先的num[]
        idx = 0;
        while (lo <= hi) {
            nums[lo++] = newArr[idx++];
        }
    }
	
	/**
     * Merge K Sorted Arrays
	 * Divide & Conquer 分到最后两个arrays相比较，然后merge（上面的）
	 * @return
	 */
	public List<Integer> mergeKSortedArrays(List<int[]> chunks) {
		List<Integer> result = new ArrayList<>();
		
		mergeSort(result, 0, chunks.size());	//divide arrays
		
		return result;
	}
	
	public void mergeSort(List<Integer> result, int start, int end) {
		if (start == end) {
			return;
		}
		
		int mid = start + (end - start) / 2;
		//TODO
//		int[] left = mergeSort(result, start, mid);
//		int[] right = mergeSort(result, mid+1, end);
//		
//		merge(left, right);
	}
	
	
	 
	/** Merge K Sorted Arrays
	 * 用Priority Queue
	 * @param chunks
	 * @return
	 */
	public int[] mergeKSortedArraysHeap(List<int[]> chunks) {
		int total = 0;
		Queue<Element> heap = new PriorityQueue<Element>(chunks.size(), eleComparator);
		for (int i = 0; i < chunks.size(); i++) {	//add 1st of every arr[]
			if (chunks.get(i).length > 0) {
				Element e = new Element(i, 0, chunks.get(i)[0]);
				heap.add(e);
				total += chunks.get(i).length;		//add every arrs' length
			}
		}
		int[] result = new int[total];
		int index = 0;
		while (!heap.isEmpty()) {
			Element ele = heap.poll();		//每次poll出来一个，就add进去那个数组的下一个数
			result[index++] = ele.val;
			if (ele.col + 1 < chunks.get(ele.row).length) {		//that arr has more to compare
				ele.col += 1;
				ele.val = chunks.get(ele.row)[ele.col];
				heap.add(ele);
			}
		}
		
		return result;
	}
	
	private Comparator<Element> eleComparator = new Comparator<Element>() {
		public int compare(Element e1, Element e2) {
			return e1.val - e2.val;
		}
	};
	
	private class Element {
		int row;
		int col;
		int val;
		
		public Element(int row, int col, int val) {
			this.row = row;
			this.col = col;
			this.val = val;
		}
	}


    
    /**
     * 215. Kth Largest Element in an Unsorted Array
     *
     * 703. Kth Largest Element in a Stream -  有stream也用它
     *
     * Given [3,2,1,5,6,4] and k = 2, return 5.
     *
     * 用minHeap，size为K。把前K个放进去。
     * 之后(k~n)个数跟heap里的root比，大于就放进去。这样得到K里面的最小值。
     * (不是用maxHeap, 除非放n个数进去。浪费！)
     * 时间 O(nlogK) + O(k)空间(heap)
     */
    public int findKthLargestMinHeap(int[] nums, int k) {
        Queue<Integer> minHeap = new PriorityQueue<>(k, (n1, n2) -> n1 - n2);
        
        for (int i = 0; i < k; i++) {
            minHeap.add(nums[i]);
        }
        
        for (int i = k; i < nums.length; i++) {
            if (nums[i] > minHeap.peek()) {
                minHeap.poll();
                minHeap.add(nums[i]);
            }
        }
        return minHeap.peek();
        
        // 可以都放进去，直到size > k 就弹出来。但这样slower，因为每个点都放进去，要重新排
        /*
        Queue<Integer> minHeap = new PriorityQueue<>(k+1);
        for (int i : nums) {
            minHeap.add(i);
            if (minHeap.size() > k) {
                minHeap.poll();
            }
        }
         */
    }
    
    
    /**
     * 215. Kth Largest Element in an Unsorted Array
     * @param nums
     * @param k
     * @return
     * 用quickSelect方法。 最快是O(n) -> O(n) + O(n/2) + O(n/4) + ...= O(2n)
     * 但是最坏情况是O(n^2)
     */
    public int findKthLargest(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k <= 0) {
            return 0;
        }
        int start = 0, end = nums.length - 1;
        int index = nums.length - k;    // rec是找Kth smallest的. and index -1 (0 ~ k-1)

        while (true) {
            int pos = partition(nums, start, end);

            if (pos == index) {
                return nums[pos];           // 找到
            } else if (pos < index) {
                start = pos + 1;
            } else {
                end = pos - 1;
            }
        }

        // 也可以不用while，直接if-else 调用 recursion，看quickSelect()那

    }

    /**
     * 最后while以后，记得 start(那个作为pivot的index) 与 right(j) 来 swap
     * 比如最后一步 变成  12, ....., 11, 15, 14, ...
     *          (pivot)start ...   i , j, ..
     *
     * 因为 11 < 12, 所以 i + 1 到 15的位置
     * 而 15 > 12,  所以 j - 1  到 11的位置
     *
     * 所以变成  (pivot)start ...   j , i, ..
     *
     * 然后 start需要和j swap，这样最后 j (end) 的地方 是12，代表smallest in right
     */
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

        swap(nums, start, j);   // j的数会大于pivot，所以 j - 1, should at the last smaller one. need to swap with start

        return j;           // j is the smallest of right part 这时 j 在 i 左边
    }



    /**
     * 跟1 的区别是，把findKthLargest() 主函数里  while( ... ) 换成 直接 if-else调用 recursion  quickSelect()
     */
    public int findKthLargest1(int[] nums, int k) {

        if (nums == null || nums.length == 0 || k <= 0) {
            return 0;
        }											//helper里找第K小，所以这要反过来
        return quickSelect(nums, 0, nums.length - 1, nums.length - k);
    }

    // 跟下面的quickSort() 很像.. 先 partition.. 然后 recursively call quickSelect()
    public int quickSelect(int[] nums, int start, int end, int k) {
        if (start == end) {
            return nums[start];
        }

        int pos = partition(nums, start, end);              // 调用上面的partition

        if (pos == k)     return nums[pos];
        if (pos < k)      return quickSelect(nums, pos + 1, end, k);
        else              return quickSelect(nums, start, pos - 1, k);
    }



    /**
     * Quick Sort  -  best O(n*logn) , worst O(n^2)
     * @param arr
     *
     * quickSort()这里recursion 调用，有 logN 次
     * partition那是 O(n)...  所以整个quick sort是 O( N * logN )
     */
    public void quickSort(int[] arr) {
        quickSort(arr, 0, arr.length - 1);
    }

    public void quickSort(int[] arr, int start, int end) {
        if (start >= end)
            return;

        int mid = partition(arr, start, end);           // 上面那个partition

        // 接下来再 recursively sort left和right半段
        quickSort(arr, start, mid - 1);
        quickSort(arr, mid + 1, end);
    }



    /**
     * 主要区别就是：把 partition() 和 recursion quickSelect2() 都放到一起
     */
    public int quickSelect2(int[] nums, int start, int end, int k) {
        if (start == end) {
            return nums[start];
        }

        int pivot = nums[start];	//或者找nums[mid]

        int left = start;
        int right = end;

        while (left <= right) {
            while (left <= right && nums[left] < pivot)   left++;
            while (left <= right && nums[right] > pivot)  right--;

            if (left <= right) {
                swap(nums, left, right);
                left++;
                right--;
            }
        }

        if (k <= right)     return quickSelect2(nums, start, right, k);
        if (k >= left)      return quickSelect2(nums, left, end, k);

        return pivot;

        /** 这种会把start和end的index设为从0开始，所以后面的k要减去前部分 （麻烦）
         // start ~ j, j+1 ~ i-1(=p), i ~ end
         if (k <= j - start + 1) {
         return quickSelect(nums, start, j, k);    // cause contains k, still find Kth smallest
         } else if (k >= i - start + 1) {
         return quickSelect(nums, i, end, k - (i - start));    // partly contains k, so find (k - (start~i))th number
         }
         return nums[j+1];   //else this part ==p
         */
    }


    /**
     * 973. K Closest Points to Origin
     * 用quick select方法更快
     * @param points
     * @param K
     * @return
     */
    public int[][] kClosest(int[][] points, int K) {
        int left = 0;
        int right = points.length - 1;

        while (left <= right) {
            int mid = quickSelect(points, left, right);

            if (mid == K) {
                break;
            } else if (mid < K) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return Arrays.copyOf(points, K);
    }

    private int quickSelect(int[][] points, int left, int right) {
        int start = left;
        int[] pivot = points[start];

        while (left <= right) {
            while (left <= right && isSmallerOrEqual(points[left], pivot)) {
                left++;
            }
            while (left <= right && !isSmallerOrEqual(points[right], pivot)) {
                right--;
            }

            if (left <= right) {
                swap(points, left, right);
            }
        }

        swap(points, start, right);     // remember to swap !!

        return right;
    }

    private boolean isSmallerOrEqual(int[] a, int[] b) {
        return a[0] * a[0] + a[1] * a[1] <= b[0] * b[0] + b[1] * b[1];
    }

    private void swap(int[][] points, int i1, int i2) {
        int[] temp = points[i1];
        points[i1] = points[i2];
        points[i2] = temp;
    }



    /**
     * 347. Top K Frequent Elements
     * @param nums
     * @param k
     * @return
     * 用HashMap存frequency
     * 然后根据freq存minHeap..里面放的是整个entry, 这样才能比较q.peek().freq.
     * 同时 minHeap大小为k, 每次peek就是最少的freq，如果比当前的更少，那就换掉peek
     */
    public List<Integer> topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int n : nums) {
            map.put(n, map.getOrDefault(n, 0) + 1);
        }
        
        PriorityQueue<Map.Entry<Integer, Integer>> minHeap = new PriorityQueue<>((a, b) -> a.getValue()-b.getValue());

        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
        	if (minHeap.size() < k) {
        		minHeap.offer(entry);
        	} else if (minHeap.peek().getValue() < entry.getValue()) {
        		minHeap.poll();
        		minHeap.offer(entry);
        	}
        }
        
        List<Integer> list = new LinkedList<Integer>();
        while (!minHeap.isEmpty()) {
            list.add(0, minHeap.poll().getKey());
        }
        
        return list;
    }
    
    
        
        /**
         * 347. Top K Frequent Elements - bucket sort O(n)
         * 用Array放bucket
         * frequency当Arr的index
         */
    public List<Integer> topKFrequentBucket(int[] nums, int k) {
    	Map<Integer, Integer> map = new HashMap<Integer, Integer>();
    	//也是存frequency
    	for (int n : nums) {
    		if (!map.containsKey(n))	map.put(n, 1);
            else		map.put(n, map.get(n) + 1);
    	}

     // if only 1num in nums, should have index 1, cause freq = 1
        List<Integer>[] bucket = new List[nums.length + 1];
        for (int n : map.keySet()) {
            int freq = map.get(n);
            if (bucket[freq] == null) {
                bucket[freq] = new ArrayList<>();
            }
            bucket[freq].add(n);
        }
        
        List<Integer> list = new ArrayList<>();
        // 记得是len-1开始 , 因为最后面的freq才大
        for (int i = nums.length; i > 0 && list.size() < k; i--) {
            if (bucket[i] != null) {

                // 692. Top K Frequent Words 这里需要相同freq的单词sort 那就加 Collections.sort(bucket[i])

                list.addAll(bucket[i]);
           //     k -= bucket[i].size();
            }
        }
        
        return list.subList(0, k);	 //如果有n个符合的，但>k, 那就subList
    }
    
    
    
    /**
     * 451. Sort Characters By Frequency   - bucket sort O(n)
     * 给tree, 输出 eetr. 其中tr顺序没所谓
     * @param s
     * @return
     * 跟上面的top k frequent 很像
     */
    public String frequencySort(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        
        //最多长度是都不重复。如果只有1个元素，需要len+1,因为freq=1不是0
        StringBuilder[] bucket = new StringBuilder[s.length() + 1];    
        
        //也可以用 List<Character>[] bucket.. 最后加result时再for (char c:bucket[i]  for (i)加sb
        
        for (Character c : map.keySet()) {
            int freq = map.get(c);
            if (bucket[freq] == null) {
                bucket[freq] = new StringBuilder();
            }
            for (int i = freq; i > 0; i--) {
                bucket[freq].append(c);
            }
        }
        
        StringBuilder sb = new StringBuilder();
        for (int i = bucket.length - 1; i > 0; i--) {
            if (bucket[i] != null) {		//记得非null
                sb.append(bucket[i]);
            }
        }
        return sb.toString();
    }
    
    // O(nlogn)  在for n时，每次poll也需要logn时间
    public String frequencySort2(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray()) {
            if (!map.containsKey(c)) {
                map.put(c, 1);
            } else {
                map.put(c, map.get(c) + 1);
            }
        }
        
        int n = s.length();
        Queue<Map.Entry<Character, Integer>> heap = new PriorityQueue<>(n, new Comparator<Map.Entry<Character, Integer>>() {
                public int compare(Map.Entry<Character, Integer> e1, Map.Entry<Character, Integer> e2) {
                    return e2.getValue() - e1.getValue();
                }
            });
        
        heap.addAll(map.entrySet());        //add all map to heap
        
        StringBuilder sb = new StringBuilder();
        while (!heap.isEmpty()) {
            Map.Entry e = heap.poll();
            for (int i = 0; i < (int) e.getValue(); i++) {
                sb.append(e.getKey());
            }
        }
        return sb.toString();
    }
    
    
    
    /**
     * 373. Find K Pairs with Smallest Sums
     * 给2个sorted arr和k。每个数组挑一个数组成pair，找出第1~k个最小的pairs
     * @param nums1
     * @param nums2
     * @param k
     * @return
     * 跟378. Kth Smallest Element in a Sorted Matrix 很类似
     * 
     * 1. 先把nums2 (或nums1)里的数 放到minHeap里，顺便<k
     * 2. 然后 每次poll一个后，再加多一个进去，这个是nums1的后一个数 (或nums2，如果1里放n1的话)
     * 
     * k * log(k)
     */
    public List<int[]> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<int[]> result = new ArrayList<>();
        int m = nums1.length;
        int n = nums2.length;
        
        Queue<NumType> heap = new PriorityQueue<>();
        for (int j = 0; j < n && j < k; j++) {   				//add nums2 in heap,相当于列
            heap.add(new NumType(0, j, nums1[0] + nums2[j]));
        }
        
        while (k-- > 0 && !heap.isEmpty()) {
            NumType t = heap.poll();
            result.add(new int[]{nums1[t.x], nums2[t.y]});
            if (t.x == m - 1)    continue; 				
            heap.add(new NumType(t.x + 1, t.y, nums1[t.x + 1] + nums2[t.y]));
        }                       // 把nums1 后面的(row)放进去
        return result;
    }
    
    class NumType implements Comparable<NumType>{
        int x, y, val;
        public NumType (int x, int y, int val) {
            this.x = x;
            this.y = y;
            this.val = val;
        }
        
        @Override	
        public int compareTo(NumType that) {	//compareTo()
            return this.val - that.val;
        }
    }
    
    
    /** 这方法比较容易想.. 但是慢
     * for n1, for n2, 都加进去，维持k大小的max heap就行
     */
    public List<int[]> kSmallestPairsSlower(int[] nums1, int[] nums2, int k) {
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>(k, new Comparator<int[]>() {
            public int compare (int[] a, int[] b) {
                return (b[0] + b[1]) - (a[0] + a[1]);
            }
        });
        
        for (int i = 0; i < nums1.length && i < k; i++) {
            for (int j = 0; j < nums2.length && j < k; j++) {
                if (maxHeap.size() < k) {
                    maxHeap.add(new int[]{nums1[i], nums2[j]});
                } else {
                    int[] peek = maxHeap.peek();
                    if (nums1[i] + nums2[j] < peek[0] + peek[1]) {
                        maxHeap.add(new int[]{nums1[i], nums2[j]});
                        maxHeap.poll();
                    }
                }
            }
        }
        
        List<int[]> result = new ArrayList<>();
        while (!maxHeap.isEmpty()) {
            result.add(maxHeap.poll());
        }
        return result;        
    }
    
    
    /**
     * Kth Largest in N Arrays
     * @param arrays a list of array
     * @param k an integer
     * @return an integer, K-th largest element in N arrays
     */
    public int KthInArrays(List<int[]> arrays, int k) {
    	Queue<Node> heap = new PriorityQueue<>();
    	int n = arrays.size();
    	for (int i = 0; i < n; i++) {
    		int index = arrays.get(i).length - 1;	//put the biggest
    		heap.add(new Node(i, index, arrays.get(i)[index]));
    	}
    	
    	for (int i = 0; i < k - 1; i++) {
    		Node t = heap.poll();
    		if (t.index > 0) {
    			heap.add(new Node(t.row, t.index - 1, arrays.get(t.row)[t.index-1]));
    		}
    	}
    	
    	return heap.peek().val;
    }
    
    class Node implements Comparable<Node>{
        int row, index, val;
        public Node (int row, int index, int val) {
            this.row = row;
            this.index = index;
            this.val = val;
        }
        
        @Override	
        public int compareTo(Node that) {	//compareTo()
            return that.val - this.val;
        }
    }
    
    
    
    
    /**
     * 4. Median of Two Sorted Arrays
     * http://www.jiuzhang.com/qa/2079/
     * @param A
     * @param B
     * @return
     * 假设要找第k(len/2)个元素。
     * 在A和B中分别寻找各自数组的第k/2个元素，比较两个找到的元素的大小。
     * 		若A中元素小，则抛弃掉A中的前k／2个元素，
     * 		反之抛弃掉B中的前	k／2个元素，并继续寻找A和B中的剩下的 第k（这里k＝k－k／2）个元素。
     * 
     * 几个边界条件：
     * 1）当其中一个数组元素全部被抛弃时(A_start >= A.length)，直接返回另一个数组中的第k个元素。
     * 2）如果k＝＝1，则直接返回两个数组中第一个元素较小的那一个。
     * 3）如果一个数组剩余元素不足k／2个，则抛弃另一个数组的k／2个元素
     * （肯定不会将要找的第k个元素抛弃掉，因为就算不足的那个数组的元素也一起被抛弃掉，抛掉的元素还是不足k个）
     */
    public double findMedianSortedArrays(int[] A, int[] B) {
        int len = A.length + B.length;
        if (len % 2 == 1) 
            return findKth(A, 0, B, 0, len/2 + 1);  // 因为index本身-1，所以要找第中间个+1
        return (findKth(A, 0, B, 0, len/2) + findKth(A, 0, B, 0, len/2 + 1)) / 2.0;
    }
    
    public static int findKth(int[] A, int A_start, int[] B, int B_start, int k) {
        if (A_start >= A.length)    return B[B_start + k - 1];
        if (B_start >= B.length)    return A[A_start + k - 1];
        
        //  A和B里找从小到大的第1的数 (就是k)
        if (k == 1) return Math.min(A[A_start], B[B_start]);
        
        //当A数组短于要找的，那么就设为max，这样能包含A。因为B的前面k/4已经知道很小，不会取到。但是A的大小不确定
        int midA = A_start + k / 2 - 1 < A.length ? A[A_start + k / 2 - 1] : Integer.MAX_VALUE;
        int midB = B_start + k / 2 - 1 < B.length ? B[B_start + k / 2 - 1] : Integer.MAX_VALUE;
        
        if (midA < midB) {		// A的中点 < B, 那么 startA ~ 中点的前半部分都抛弃。从中点mid开始找
            return findKth(A, A_start + k / 2, B, B_start, k - k / 2);
        } else {							// 抛弃了前 k/2的nums们，所以要找的只剩 k - k/2
            return findKth(A, A_start, B, B_start + k / 2, k - k / 2);
        }
    }


    /**
     * Find Median in large file
     * 在很大的文件里找median
     *
     * Assumption:
     * 1. file很大无法存内存memory
     * 2. file里存integer
     * 3. file里的Integer范围是INT_MIN和INT_MAX
     *
     * Approach:
     * 二分查找
     * 思路就是：
     * 1. 先找在INT_MIN和INT_MAX的median，
     * 2. for循环扫所有integers，找出比这个数小的个数是否有一半，然后调整二分的边界
     *
     * Time: O(32n) = O(n) where n is the size of array
     * we at most iterate the num 32 times
     * 奇数遍历32次，偶数因为查2次所以是64次
     *
     * 因为范围是 Integer..那么就是 2^32 - 1 .. 每次二分那就是最多32次
     * Space: O(1)
     */
    public double findMedian(int[] nums) {
        long len = nums.length;

        if (len % 2 == 1) {
            return guessKthSmallest(nums, len / 2 + 1);
        } else {
            return (guessKthSmallest(nums, len / 2) + guessKthSmallest(nums, len / 2 + 1)) / 2.0;
        }
    }

    private double guessKthSmallest(int[] nums, long k) {
        long left = Integer.MIN_VALUE;
        long right = Integer.MAX_VALUE;

        while (left <= right) {
            int count = 0;
            long result = left;
            long guessMid = left + (right - left) / 2;

            // 找出count个小于guess的数，并且持续更新result成为最大的作为median candidate
            for (int num : nums) {
                if (num <= guessMid) {
                    count++;
                    result = Math.max(result, num);     // 找到最大的smaller，这样是median
                }
            }

            if (count == k) {
                return result;
            } else if (count < k) {
                left = guessMid + 1;
            } else {
                right = guessMid - 1;
            }
        }
        return left;
    }


    /**
     * 349. Intersection of Two Arrays  - easy
     * 结果是unique的，去重
     * @param nums1
     * @param nums2
     * @return
     *
     * O(n). 用2个set，一个存某个数组，另一个放相交的result
     *
     * 或者 O(logn) sort，two pointer.. 因为要de-dupe所以还需要多一个hash set来存结果
     */
    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums1) {
            set.add(num);
        }

        Set<Integer> result = new HashSet<>();
        for (int num : nums2) {
            if (set.contains(num)) {
                result.add(num);
            }
        }

        // 放result
        int[] output = new int[result.size()];
        int i = 0;
        for (int n : result) {
            output[i++] = n;
        }
        return output;
    }


    /**
     * 350. Intersection of Two Arrays II - easy
     * Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2, 2].
     * 其实是找重复出现的..
     * 所以map里存的是出现的次数
     * @param nums1
     * @param nums2
     * @return
     *
     * Follow-up
     * What if elements of nums2 are stored on disk, and the memory is
     * limited such that you cannot load all elements into the memory at
     * once?
     *
     * If only nums2 cannot fit in memory, put all elements of nums1 into a HashMap,
     * read chunks of array that fit into the memory, and record the intersections.
     *
     * If both nums1 and nums2 are so huge that neither fit into the memory,
     * sort them individually (external sort), then read 2 elements from each array at a time in memory, record intersections.
     *
     * https://leetcode.com/problems/intersection-of-two-arrays-ii/discuss/82243/Solution-to-3rd-follow-up-question
     */
    public int[] intersect(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map = new HashMap<>();
        List<Integer> intersect = new ArrayList<>();
        for (int i = 0; i < nums1.length; i++) {
            if (map.containsKey(nums1[i])) 
                map.put(nums1[i], map.get(nums1[i]) + 1);
            else map.put(nums1[i], 1);
        }
        
        for (int i = 0; i < nums2.length; i++) {
            if (map.containsKey(nums2[i]) && map.get(nums2[i]) > 0) {
                intersect.add(nums2[i]);
                map.put(nums2[i], map.get(nums2[i]) - 1);	// -1 更新
            }
        }
        int[] result = new int[intersect.size()];
        int k = 0;
        for (int num : intersect) {
            result[k++] = num;
        }
        return result;
    }
    
    
    /** 
     * 350. Intersection of Two Arrays II - if sorted - easy
     * sorted的话就用 双指针
     */
    public int[] intersectionII(int[] nums1, int[] nums2) {
        List<Integer> intersect = new ArrayList<>();
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int i = 0;
        int j = 0;
        while (i < nums1.length && j < nums2.length) {
            if (nums1[i] < nums2[j]) {
                i++;
            } else if (nums1[i] > nums2[j]) {
                j++;
            } else {
                intersect.add(nums1[i]);
                i++;
                j++;
            }
        }
        
        int[] result = new int[intersect.size()];
        int k = 0;
        for (int num : intersect) {
            result[k++] = num;
        }
        return result;
    }


    /**
     * 844. Backspace String Compare - easy
     * 删掉空格，看2个string是否一样..
     * @param S "ab#c"
     * @param T "ad#c"
     * @return true，因为只剩下c
     */
    public boolean backspaceCompare(String S, String T) {
        return deleteBackSpace(S).equals(deleteBackSpace(T));
    }

    // 额外空间
    private String deleteBackSpace(String s) {
        char[] ch = s.toCharArray();
        int i = 0;
        for (int j = 0; j < s.length(); j++) {
            if (ch[j] == '#') {
                i--;
                if (i < 0) {
                    i = 0;
                }
            } else {
                ch[i] = ch[j];
                i++;
            }
        }
        return i <= 0 ? "" : String.valueOf(ch).substring(0, i);
    }

    // 没有额外空间  从后面比较更方便，而且一起比较   快一点
    public boolean backspaceCompare1(String S, String T) {
        int i = S.length() - 1, j = T.length() - 1;
        int skipS = 0, skipT = 0;

        while (i >= 0 || j >= 0) {
            while (i >= 0) {
                if (S.charAt(i) == '#') {
                    skipS++;
                    i--;
                } else if (skipS > 0) {
                    skipS--;
                    i--;
                } else {
                    break;
                }
            }

            while (j >= 0) {
                if (T.charAt(j) == '#') {
                    skipT++;
                    j--;
                } else if (skipT > 0) {
                    skipT--;
                    j--;
                } else {
                    break;
                }
            }

            if (i >= 0 && j >= 0 && S.charAt(i) != T.charAt(j))
                return false;

            if ((i >= 0) != (j >= 0))
                return false;

            i--;
            j--;
        }
        return true;
    }


    /**
     * 833. Find And Replace in String - google
     * S = "abcd", indexes = [0,2], sources = ["a","cd"], targets = ["eee","ffff"]
     * 表示 在idx 0的位置，match了sources的 "a"的话，那就replace成targets "eee".
     * idx 2时，match了"cd" 的话就replace成 "ffff"
     *
     * Output: "eeebffff"
     * 假设 All these operations occur simultaneously. 这个不会有overlap的问题
     *
     * indexes[] 不是sorted的，所以需要用HashMap来存下
     */
    public String findReplaceString(String S, int[] indexes, String[] sources, String[] targets) {

        StringBuilder sb = new StringBuilder();
        Map<Integer, Integer> match = new HashMap<>();  // key index[i] pos, value: index

        for (int i = 0; i < indexes.length; i++) {
            if (S.startsWith(sources[i], indexes[i])) {
                match.put(indexes[i], i);
            }
        }

        for (int i = 0; i < S.length(); i++) {
            if (match.containsKey(i)) {
                sb.append(targets[match.get(i)]);
                i += sources[match.get(i)].length() - 1;
            } else {
                sb.append(S.charAt(i));
            }
        }
        return sb.toString();
    }

    /**
     * 833. Find And Replace in String
     * 如果indexes这些是 sorted有序的话，可以下面这么做..
     */
    public String findReplaceString1(String S, int[] indexes, String[] sources, String[] targets) {

        StringBuilder sb = new StringBuilder();

        int idx = 0;      // loop through indexes
        for (int i = 0; i < S.length(); i++) {
            if (idx == indexes.length || i < indexes[idx]) {        // no change
                sb.append(S.charAt(i));
            } else {
                String source = sources[idx];
                if (S.substring(i, i + source.length()).equals(source)) {
                    sb.append(targets[idx]);
                    i += source.length() - 1;
                    idx++;
                } else {
                    sb.append(S.charAt(i));
                }
                /*
                int j = i;
                int k = 0;
                while (j < i + source.length() && j < S.length() && k < source.length()) {
                    if (S.charAt(j) == source.charAt(k)) {
                        j++;
                        k++;
                    } else {
                        idx++;
                        sb.append(S.charAt(i));
                        break;
                    }
                }
                if (j == i + source.length() && k == source.length()) {
                    sb.append(targets[idx]);
                    i = j - 1;  // 因为后面i++ 这样会skip掉一位
                    idx++;
                }
                */
            }
        }
        return sb.toString();
    }


    /**
     * 412. Fizz Buzz - Easy
     * 遇到被3整除就Fizz，被5整除就Buzz
     * @param n
     * @return
     *
     * 扩展：如果有更多mapping  被7整除就pizza 之类的
     * 那就需要map来存  comment out的那些
     */
    public List<String> fizzBuzz(int n) {

        List<String> ans = new ArrayList<String>();

        // 如果mapping太多的话
        Map<Integer, String> map = new HashMap<>();
        map.put(3, "Fizz");
        map.put(5, "Buzz");

        for (int num = 1; num <= n; num++) {
            String s = "";

            if (num % 3 == 0) {
                s += "Fizz";
            }
            if (num % 5 == 0) {
                s += "Buzz";
            }

            // 用map的话，上面改成
//             for (Integer key : map.keySet()) {
//                 if (num % key == 0) {
//                     s += map.get(key);
//                 }
//             }


            if (s == "") {
                s = Integer.toString(num);
            }

            ans.add(s);
        }

        return ans;
    }

    // 不用 % 的话，用 int fizz, buzz ++  来表示
    public List<String> fizzBuzz1(int n) {
        List<String> result = new ArrayList<String>();
        int fizz = 0;
        int buzz = 0;

        for (int num = 1; num <= n; num++) {
            fizz++;
            buzz++;
            if (fizz == 3 && buzz == 5) {
                result.add("FizzBuzz");
                fizz = 0;
                buzz = 0;
            } else if (fizz == 3) {
                result.add("Fizz");
                fizz = 0;
            } else if (buzz == 5) {
                result.add("Buzz");
                buzz = 0;
            } else {
                result.add(String.valueOf(num));
            }
        }
        return result;
    }


    /**
     * 605. Can Place Flowers
     * 0的时候可以放flower，但是2朵花之间至少有一个0相隔
     * @param flowerbed
     * @param n
     * @return
     */
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int len = flowerbed.length;
        int count = 0;

        for (int i = 0; i < len; i++) {
            if (flowerbed[i] == 0) {
                if ((i == 0 || flowerbed[i - 1] == 0) && (i == len - 1 || flowerbed[i + 1] == 0)) {
                    flowerbed[i] = 1;
                    count++;
                }
            }
            // 记得在if 最外面.. 不一定要为0时才判断
            if (count >= n) {       // break earlier
                return true;
            }
        }
        return false;           // 也可以最后才判断 count >= n
    }

    // 不改变input，直接算 几个0可以放多少花..能放 (zeros - 1) / 2 朵花
    public boolean canPlaceFlowers1(int[] flowerbed, int n) {
        int len = flowerbed.length;
        int zeros = 1;              // corner case, 第一个数是0时
        int flowers = 0;

        for (int i = 0; i < len; i++) {
            if (flowerbed[i] == 0) {
                zeros++;
            } else {
                flowers += (zeros - 1) / 2;     // 连续几个zeros可以放(zeros - 1) / 2 朵花
                zeros = 0;                      // 断开以后，要reset下zero
            }
        }

        // 最后element是0， 这时不需要 zeros - 1再除2
        if (zeros > 0) {
            flowers += zeros / 2;
        }

        return flowers >= n;
    }


    /**
     * 599. Minimum Index Sum of Two Lists - easy
     *  find out their common interest with the least list index sum.
     * @param list1
     * @param list2
     * @return
     */
    public String[] findRestaurant(String[] list1, String[] list2) {
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < list1.length; i++) {
            map.put(list1[i], i);
        }

        List<String> result = new ArrayList<>();
        int minSum = Integer.MAX_VALUE;
        int sum = 0;

        for (int i = 0; i < list2.length; i++) {
            if (map.containsKey(list2[i])) {
                sum = i + map.get(list2[i]);
                if (sum < minSum) {
                    result.clear();
                    result.add(list2[i]);
                    minSum = sum;
                } else if (sum == minSum) {
                    result.add(list2[i]);
                }
            }
        }
        return result.toArray(new String[result.size()]);
    }

    // no space, 慢 O( (len1 + len2)^2 * strLength)
    public String[] findRestaurantSlow(String[] list1, String[] list2) {
        int len1 = list1.length;
        int len2 = list2.length;

        List<String> result = new ArrayList<>();

        for (int sum = 0; sum < len1 + len2 - 1; sum++) {
            for (int i = 0; i <= sum; i++) {
                if (i < len1 && sum - i < len2 && list1[i].equals(list2[sum - i])) {
                    result.add(list1[i]);
                }
            }
            if (result.size() > 0)
                break;
        }
        return result.toArray(new String[result.size()]);
    }


    /**
     * 722. Remove Comments
     * remove掉 // 或者 /*...
     * @param source
     * @return
     */
    public List<String> removeComments(String[] source) {
        List<String> result = new ArrayList<>();

        StringBuilder sb = new StringBuilder();
        boolean isBlock = false;

        for (String line : source) {
            int len = line.length();
            for (int i = 0; i < len; i++) {
                if (isBlock) {
                    if (line.charAt(i) == '*' && i + 1 < len && line.charAt(i + 1) == '/') {
                        isBlock = false;
                        i++;                // skip '/'
                    }
                } else {
                    if (line.charAt(i) == '/' && i + 1 < len && line.charAt(i + 1) == '/') {
                        break;              // ignore remaining
                    } else if (line.charAt(i) == '/' && i + 1 < len && line.charAt(i + 1) == '*') {
                        isBlock = true;
                        i++;                // skip '*'
                    } else {
                        sb.append(line.charAt(i));          // normal char
                    }
                }
            }

            if (!isBlock && sb.length() > 0) {
                result.add(sb.toString());
                sb.setLength(0);
            }
        }

        return result;
    }
    
	
	public static void main(String[] args) {
		ArrStrSolution as = new ArrStrSolution();
		String test = "       ".trim();
		System.out.println(as.checkSubarraySum(new int[]{2, 34},6));
		String[] strs = {"worde", "eordw","abc", "cab", "erw", "eword", "aaa", "cba"};
//		System.out.println("worde, eordw,abc, cab, erw, eword, aaa, cba");
		List<List<String>> result = as.groupAnagrams(strs);
		System.out.println(result.toString());
		
		List<Interval> list = new ArrayList<>();
		Interval a = new ArrStrSolution().new Interval(1,4);
		list.add(a);
		list.add(new ArrStrSolution().new Interval(0, 2));
		list.add(new ArrStrSolution().new Interval(3, 5));
		
		int[] arr = {1,2,1,0,0,2,1};
		
		
		int arr1[] = {1,5,6,9,21};
		int arr2[] = {4,6,11,14};
		int arr3[] = {-1,0,7};
		int arr4[] = {-4,-2,11,14,18};
		int arr5[] = {2,6};
		int arr6[] = {-5,-2,1,5,7,11,14};
		int arr7[] = {-6,-1,0,15,17,22,24};
		List<int[]> chunks = new ArrayList<int[]>();
		chunks.add(arr1);
		chunks.add(arr2);
		chunks.add(arr3);
		chunks.add(arr4);
		chunks.add(arr5);
		chunks.add(arr6);
		chunks.add(arr7);
		
		System.out.println(as.KthInArrays(chunks, 4));
		
		System.out.println("=============");
		System.out.println("result : "+as.countSmaller(new int[]{5,2,6,1}));
		
		
	}

}

