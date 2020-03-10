
import java.util.*;

/**
 * @author Yan
 *
 */
public class DFS_Subset {
	
	/**78. Subsets - recursive
	 * return all possible subsets. 不能重复
	 * @param nums
	 * @return
	 */
	public static List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (nums == null || nums.length == 0)   return result;
        
        Arrays.sort(nums);	//有重复时要sort
        ArrayList<Integer> subset = new ArrayList<Integer>();
        helper(result, subset, nums, 0);
        
        return result;
    }
    
    public static void helper(List<List<Integer>> result, ArrayList<Integer> subset, int[] nums, int pos) {
    	// 如果result.add(subset)，会每次都加同样的指针to 上面方法里那个空的subset，所以output的结果都是一样的.. 
    	// for里面操作虽然每次会改变subset, 但是最终所有recursion弹完以后subset会变为空
    	// 所以这么多指针指的subset也都是空
    	result.add(new ArrayList<Integer>(subset));
        
        for (int i = pos; i < nums.length; i++) {
        	// 如果不允许重复的话  针对subsetII
        	if (i != pos && nums[i] == nums[i-1]) {
                continue;
            }
            subset.add(nums[i]);
            helper(result, subset, nums, i+1);
            subset.remove(subset.size() - 1);
        }
    }
    
    
    /** 78. Subsets - Iteration
     * @param nums
     * @return
     * 循环nums, 每次在新的subset list里加num, 导致每次for的result都double size
     * 如果上一次的result size是4，那就里面那个for循环4次，把num加到新复制的sub里
     * 这样result最后就被8的size
     *
     * 比如1, 2, 3的情况
     * 第一次是1，subset为空，加完后result变2个size {}, {1}.
     * 轮到2，那就是现有result基础上 size * 2..变成 {}. {1}, {2}, {1,2}.. 根据之前的来
     * 轮到3，那就是 4 * 2的size..在之前基础上加3
     *
     * http://bangbingsyb.blogspot.com/2014/11/leetcode-subsets-i-ii.html
     */
    public static List<List<Integer>> subsetsIter(int[] nums) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (nums == null || nums.length == 0)   return result;
        
        Arrays.sort(nums);	//有重复时要sort

        List<Integer> subset = new ArrayList<Integer>();
        result.add(subset);

        // 每个num在之前result里的subset里加自己num，每次都double size append到result里
        for (int num : nums) {
            int n = result.size();			//记得先记录这个
            for (int j = 0; j < n; j++) {			//result每次都double翻倍
                subset = new ArrayList<Integer>(result.get(j));		//每次复制一个已有的subset
                subset.add(num);			 //在之前的sub里加num   
                result.add(subset);
            }
        }
        
        /*
        //用iteration && 不允许重复的话
        int begin = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i == 0 || nums[i] != nums[i-1]) {
                begin = 0;
            }
            int n = result.size();
            // 要从begin开始，否则如果从0开始的话，会重复往空集里加一个重复的数，也会在其他小集里重复加
            for (int j = begin; j < n; j++) {
                subset = new ArrayList(result.get(j));
                subset.add(nums[i]);
                result.add(subset);
            }
            begin = n;	//这样之后能从刚刚新加的subset开始加num，避免重复之前的
        } */
        
        return result;
    }
    
	
	/** Permutations - recursion
	 * 模板
	 * @param num
	 * @return
	 */
	public ArrayList<ArrayList<Integer>> permutation(int[] num) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        ArrayList<Integer> per = new ArrayList<Integer>();
        permuteHelper(result, per, num);    //call
        return result;
    }
    
    private void permuteHelper(ArrayList<ArrayList<Integer>> result, 
                                ArrayList<Integer> per, int[] num) {
        if (per.size() == num.length) {
            result.add(new ArrayList<Integer>(per));     //should new AL, or will change everytime
            return;                 //return, or it'll go through following and size++
        }
        
        for (int i = 0; i < num.length; i++) {
        //start from 0 everytime. if contains, then next ele.    
            if (per.contains(num[i])) {			//这里O(n)比较慢
                continue;                       
            }
            per.add(num[i]);
            permuteHelper(result, per, num);    
            per.remove(per.size() - 1);           //clear subset 1by1 to store new subset
        }
    }
    
    // swap 更快的.. 初始POS为0. 通过swap改变数组，每次再转成list加到result里
    public void helper(List<List<Integer>> result, int pos, int[] nums) {
        if (pos == nums.length) {
            // 把调换过位子的nums转成ArrayList加到result里
            List<Integer> list = new ArrayList<Integer>();
            for(int i = 0; i < nums.length; i++){
                list.add(nums[i]);
            }
            result.add(list);    // 可以直接result.add(Arrays.asList(nums)) 但是nums需要时Integer[]

            return;
        }
        
        for (int i = pos; i < nums.length; i++) {
            swap(pos, i, nums);     // 记住是 pos + 1  !!! 因为可能性在逐步减少.. 但不是i + 1 否则就更少不对了
            helper(result, pos + 1, nums);  // 第一个位置可以有三种可能, 第二个位置就只能在剩下的两个数里面选一个
            swap(pos, i, nums);
        }
        
        
        // 如果是不能重复的 II
        Set<Integer> usedSet = new HashSet<>();
        for (int i = pos; i < nums.length; i++) {
        	if (usedSet.add(nums[i])) {			//如果set没有n[i]，那就加他并TRUE。否则不加且FALSE
	            swap(pos, i, nums);
	            helper(result, pos + 1, nums);
	            swap(pos, i, nums);
        	}
        }
    }
    
    public void swap (int x, int y,int[] num){
        int temp = num[x];
        num[x]=num[y];
        num[y]=temp;
    }
    
    
    
    /** 46. Permutations - Iterative
     * @param nums
     * @return
     * 每次在当前有的(nums[i-1]组成的)list里，把nums[i]插入任意位置，组成排列
     * 1. 刚开始把空list放result
     * 2. 然后放后面的数n
     * 3. 这nums[i]个数要插在 j<=i的位置
     * 4. copy当前result的nums[i-1]组成的perm，在不同 j 位置插入nums[i]
     * 
     * !! newResult是新建一个，newPerm是copy之前的，这样newResult.add(newPerm). 最后把result = newResult
     * 
     * 另外for(j)可以跟for(perm : result)调换位置
     * 	a. for(j)在外面，那就每个perm加完第j位以后，再j++
     * 	b. for(perm)在外面，每个perm插入很多个j位，这个perm结束后，再下一个perm
     * 		b1 !! j在里面的话，也要每次copy perm，因为会生成不同的新perm
     * 		for (List<Integer> perm : result) {
                    for (int j = i; j >= 0; j--) {      //insert nums[i] to j position
                    List<Integer> newPerm = new ArrayList<>(perm);        //copy a perm
     * 
     * 同时j除了从0位置开始插，也可以j=i, j>=0 从后往前插
     */
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
//        List<Integer> perm0 = new ArrayList<>();
//        perm0.add(nums[0]);      //add the 1st num.
//        result.add(perm0);		////如果这里放了第0个，for i 从 1 开始
        result.add(new ArrayList<Integer>());
        
        for (int i = 0; i < nums.length; i++) {			
            List<List<Integer>> newResult = new ArrayList<>();		//新的，不是copy
            for (int j = 0; j <= i; j++) {      //insert nums[i] to j position
                for (List<Integer> perm : result) {
                    List<Integer> copyPerm = new ArrayList<>(perm);        //copy a perm
                    copyPerm.add(j, nums[i]);
                    newResult.add(copyPerm);
                }
            }
            result = newResult;
        }
                
        /*
       // 也可以用Linkedlist, 这样可以Pollfirst，只要keep住res.size()就行。 不过好像慢一点
        LinkedList<List<Integer>> res = new LinkedList<>();
        result.add(new ArrayList<Integer>());
        
        for (int num : nums) {
            int size = result.size();
            for (; size > 0; size--) {
                List<Integer> perm = res.pollFirst();
                for (int j = 0; j <= perm.size(); j++) {
                    List<Integer> copyp = new ArrayList<>(perm);
                    copyp.add(j, num);
                    res.add(copyp);
                }
            }
        }
        */
        
        return result;
    }
    
    
    /** 47. Permutation II - 有duplicate
     * might contain duplicates, return all possible unique permutations. 
     * [1,1,2] have the following unique permutations: [1,1,2], [1,2,1], and [2,1,1].
     * @param num
     * @return
     */
    public List<List<Integer>> permuteUnique(int[] num) {
        List<List<Integer>> result = new ArrayList<>();
        if(num == null || num.length == 0)
            return result;
        List<Integer> list = new ArrayList<Integer>();
        boolean[] visited = new boolean[num.length];
        
        Arrays.sort(num);
        helper(result, list, visited, num);
        return result;
    }
    
    public void helper(List<List<Integer>> result, List<Integer> list, boolean[] used, int[] num) {
        if(list.size() == num.length) {
            result.add(new ArrayList<>(list));
            return;
        }
        
        for(int i = 0; i < num.length; i++) {
            if (used[i] ||
            		(i != 0 && num[i] == num[i - 1] && !used[i - 1])){
                continue;
        /*
            上面的判断主要是为了去除重复元素影响。
            比如，给出一个排好序的数组，[1,2,2]，那么第一个2和第二2如果在结果中互换位置，
            我们也认为是同一种方案，所以我们强制要求相同的数字，原来排在前面的，在结果
            当中也应该排在前面，这样就保证了唯一性。所以当前面的2还没有使用的时候，就
            不应该让后面的2使用。
        */
            }
            used[i] = true;
            list.add(num[i]);
            helper(result, list, used, num);
            list.remove(list.size() - 1);
            used[i] = false;
        }
    }   
    
    
    /** 47. Permutation II - 有duplicate  - iterative
     * @param nums
     * @return
     * 之前那方法 + hashset.. 尽管这样比较慢，耗空间
     */
    public List<List<Integer>> permuteUniqueIterative(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        result.add(new ArrayList<Integer>());
        Arrays.sort(nums);
        
        for (int i = 0; i < nums.length; i++) {
            Set<String> set = new HashSet<>();		//每个新nums[i]就new一个set
            List<List<Integer>> newRes = new ArrayList<>();
            for (List<Integer> perm : result) {
                for (int j = 0; j <= i; j++) {
                    List<Integer> copyp = new ArrayList<>(perm);
                    copyp.add(j, nums[i]);
                    if (set.add(copyp.toString())) {		//没重复才加newresult里
                        newRes.add(copyp);
                    }
                }
            }
            result = newRes;
        }
        return result;
    }
    
    
	
	/** 77. Combinations
	 * return all possible combinations of k numbers out of 1 ... n.
	 * @param n
	 * @param k
	 * @return [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
	 */
	public ArrayList<ArrayList<Integer>> combine(int n, int k) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        ArrayList<Integer> comb = new ArrayList<Integer>();
        recHelper(result, comb, n, k, 1);
        return result;
    }
    
    private void recHelper(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> comb, int n, int k, int pos) {
        if (comb.size() == k) {
            result.add(new ArrayList<Integer>(comb));
            return;
        }
        //要用pos, 否则变成permutation，包括不同顺序
        for (int i = pos; i <= n; i++) {
            comb.add(i);
            recHelper(result, comb, n, k, i + 1);
            comb.remove(comb.size() - 1);
        }
        
        /*// 如果给n=4, k=2
    	// 返回[[1,2],[1,3],[1,4],[2,1],[2,3],[2,4],[3,1],[3,2],[3,4],[4,1],[4,2],[4,3]]
          for (int i = 1; i <= n; i++) {
            if (comb.contains(i))
                continue;
             ........
         */
    }
    
    
    //iterative方法，跟permutation类似
    public List<List<Integer>> combineIterative(int n, int k) {
        List<List<Integer>> result = new ArrayList<>();
        result.add(new ArrayList<Integer>());
        
        for (int t = 1; t <= k; t++) {		//外层的K
            List<List<Integer>> newres = new ArrayList<>();
            for (List<Integer> comb : result) {
                int i = comb.size() == 0 ? 1 : comb.get(comb.size() - 1) + 1;      // get the prev num
                for (; i <= n - (k - t); i++) {
                    List<Integer> newcom = new ArrayList<>(comb);
                    newcom.add(i);
                    newres.add(newcom);
                }
            }
            result = newres;
        }
        return result;
    }
    
    
    
    /** Combination Sum I
     *  可以重复算自己, 且num没有duplicate
     * given candidate set 2,3,6,7 and target 7
     * solution：[7] [2, 2, 3] 
     * 
     * given candidate set 10,1,2,7,6,1,5 and target 8
     * A solution set is: [1, 7] [1, 2, 5] [2, 6] [1, 1, 6] 
     * @param num
     * @param target
     */
    public ArrayList<ArrayList<Integer>> combinationSum(int[] num, int target) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
        ArrayList<Integer> comb = new ArrayList<Integer>();
        Arrays.sort(num);                   //have duplicate,so need to sort first
        recHelperI(result, comb, num, target, 0);
        return result;
    }
    
    // combination sum I
    private void recHelperI(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> comb, int[] num, int target, int pos) {
        if (target < 0) {
            return;
        }
        if (target == 0) {
            result.add(new ArrayList<>(comb));
            return;
        }
        
        for (int i = pos; i < num.length; i++) {

        /*
        // combination sum II 不能重复用的话。 只用for里面开始加上，跟subset一样
        if (i > pos && nums[i-1] == nums[i])
            continue;
            下面dfs就是 i + 1
        */
            comb.add(num[i]);     
            						//  直接传i.因为可以重复计算. 用pos 防止再次算回前面的
            recHelperI(result, comb, num, target - num[i], i);	//rec时sum减，这样返回上一层sum还是不变

            comb.remove(comb.size() - 1);
        }
        

    }
    
    
    /** combination sum I - DP + Iterative
     * cache里存 和为target所对应的结果List<List<Integer>>
     * 记得要先for num, 不能先target，否则会有重复的结果（顺序不一样）
     * 比如[2,3,6] 在算target为5时，for到2可以组成[2,3], 后面for到3时可以组成[3,2]，那么就重复了
     */
    public List<List<Integer>> combinationSumDP(int[] nums, int target) {
        if (nums == null || nums.length == 0)
            return new ArrayList<>();
        Arrays.sort(nums);
        
        // nextNum, 每个target sum对应的组合
        Map<Integer, List<List<Integer>>> map = new HashMap<>();
        map.put(0, new ArrayList<List<Integer>>());          //初始记得放0，后面单独的6可以用
        map.get(0).add(new ArrayList<Integer>());
        
        // 记得要先for num, 不能先target，否则会有重复
        for (int num : nums) {
            for (int t = num; t <= target; t++) {
                if (map.containsKey(t - num)) {     // 之前算过 t - num的comb结果
                    for (List<Integer> comb : map.get(t - num)) {
                        List<Integer> tmp = new ArrayList<>(comb);
                        tmp.add(num);
                        
                        // 加到这次和为t里面
                        if (!map.containsKey(t)) {
                            map.put(t, new ArrayList<List<Integer>>());    //每个target有一个comb list
                        }
                        map.get(t).add(tmp);
                    }
                }
            }
        }
        
        if (map.get(target) == null)
            return new ArrayList<>();
        
        return map.get(target);
    }
    
    // combination sum II   给的candidate[]有可能有duplicate的，每个数只能用1次
    private void recHelperII(ArrayList<ArrayList<Integer>> result, ArrayList<Integer> comb, int[] num, int target, int pos) {
    	if (target < 0)        return;
        
    	if (target == 0) {
            result.add(new ArrayList<Integer>(comb));
            return;
        }
        
        for (int i = pos; i < num.length; i++) {
        	if (i > pos && num[i-1] == num[i])
                continue;
            comb.add(num[i]);
            recHelperII(result, comb, num, target - num[i], i + 1);
            comb.remove(comb.size()-1);
        }
    }
    
    
    
    /** 216. Combination Sum III
     * Find all possible combinations of k numbers that add up to a number n
     * 给k, 从1 ~ k找出加起来=n 的组合.. 其中k为 1 ~ 9
     * @param k
     * @param n
     * @return
     */
    public List<List<Integer>> combinationSumIII(int k, int n) {
        List<List<Integer>> result = new ArrayList<>();
        sumdfsIII(result, new ArrayList<Integer>(), n, k, 0, 1);
        return result;
    }
    
    public void sumdfsIII(List<List<Integer>> result, List<Integer> comb, int target, int k, int sum, int start) {
        if (k == 0 && sum == target) {
            result.add(new ArrayList<Integer>(comb));
            return;
        }
        
        if (k < 0 || sum > target)
            return;
        
        for (int i = start; i <= 9; i++) {
            if (sum + i > target)
                return;
            comb.add(i);
            sumdfsIII(result, comb, target, k - 1, sum + i, i + 1);
            comb.remove(comb.size() - 1);
        }
    }
    
    
    /** 377. Combination Sum IV
     * Given an integer array with all positive numbers and no duplicates, 
     * find the number of possible combinations that add up to a positive integer target.
     * 求所有组合 加起来为target的 组合个数.. 
     * 可以重复用num，如[1,2,3], target=4, 可以(1,1,1,1), (1,1,2), (2,1,1)
     * @param nums
     * @param target
     * @return
     * 一看到个数，就可以想到DP
     * dp[i]为 如果target总和为i时，有多少组合个数
     *
     * 1. 初始化dp[0]=1
     * 2. for (i=1, <= target). 这代表dp[i]..因为dp都是基于之前的结果，所有要当前dp[i]算完以后，再i++
     * 3. 对于每个target i, 看是否>= 数组里的num.. 这样能拆成更小的.. d[i] += d[i - num]
     *
     * 因为这题可以 重复用num.. 是个完全背包 问题
     */
    public int combinationSumIV(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;          //若nums里有1，那d[1] += d[0], 所以需要d[0]为1
        
        for (int i = 1; i <= target; i++) {     //dp[i]表示target为i. 需要在外层
            for(int num : nums) {   
                if (i >= num) {         //只有当target i有大于nums[]里的数，才能拆分成更小的
                    dp[i] += dp[i - num];
                }
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
    
    
    /** 254. Factor Combinations
     * 8 = 2 x 2 x 2; 以及  = 2 x 4. 不能包含1 或 本身
     * @param n
     * @return
     * 正常的backtracking..要记录start，这样不会有重复..
     * 另外n==1时，需要查 size>1, 否则只有 {8} 是不对的
     */
    public List<List<Integer>> getFactors(int n) {
        List<List<Integer>> result = new ArrayList<>();
        if (n <= 3)     return result;
        
        dfsFactor(result, new ArrayList<Integer>(), n, 2);
        return result;
    }
    
    private void dfsFactor(List<List<Integer>> result, List<Integer> path, int n, int start) {
        if (n == 1) {
            if (path.size() > 1) {
                result.add(new ArrayList<>(path));
            }
            return;
        }
        
        // 正常是 i <= n  但是比较慢，看dfs2.
//        for (int i = start; i <= n; i++) {
        for (int i = start; i <= (int) Math.sqrt(n); i++) {
            if (n % i == 0) {

                // 正常慢的情况
//                path.add(i);
//                dfs(result, path, n / i, i);
//                path.remove(path.size() - 1);

                path.add(i);

                // 加这个更快.. 这样直接加上 n / i
                path.add(n / i);
                result.add(new ArrayList<>(path));      // 把 .., i 和 num / i 直接加到result里
                path.remove(path.size() - 1);

                dfsFactor(result, path, n / i, i);
                path.remove(path.size() - 1);

            }
        }
    }



    /**
     * 266. Palindrome Permutation
     * 给string看是否能组成palindrome
     * Input: "code"
     * Output: false
     *
     * @param s
     * @return
     * 遇到一次就++，存在了的就--
     * 也可以用char[].
     */
    public boolean canPermutePalindrome(String s) {
    	Set<Character>set = new HashSet<Character>();
    	for (char c : s.toCharArray())  
    		if (set.contains(c)) set.remove(c);// If char already exists in set, then remove it from set
    		else set.add(c);// If char doesn't exists in set, then add it to set
    	return set.size() <= 1;
    }

    private boolean canPermutePalindrome(String s, Map<Character, Integer> map) {
        int odd = 0;

        for (char c : s.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
            if (map.get(c) % 2 == 0) {
                odd--;
            } else {
                odd++;
            }
//            odd += map.get(c) % 2 != 0 ? 1 : -1;      // 或者直接一句
        }

        return odd <= 1;
    }
    
    
    /** 267. Palindrome Permutation II
     * s = "aabb", return ["abba", "baab"].   s = "abc", return [].
     * @param s
     * @return
     * 在list里重复的都先除以2，这样加result时再把后半段加起来就行
     * 记得dfs里的sb要reverse回来
     * 
     * 下面有更快更短的方法，用int[]
     */
    public List<String> generatePalindromes(String s) {
        List<String> result = new ArrayList<>();
        List<Character> list = new ArrayList<>();
        Map<Character, Integer> map = new HashMap<>();

        if (!canPermutePalindrome(s, map))
            return result;
        
        String mid = "";
        // 得到一半的char, 并且相同的也排在一起
        for (Map.Entry<Character, Integer> entry : map.entrySet()) {
            if (entry.getValue() % 2 != 0) {    //odd.
                mid += entry.getKey();
            }
            for (int i = 0; i < entry.getValue() / 2; i++) {
                list.add(entry.getKey());
            }
        }

        palinHelper(result, list, new boolean[list.size()], new StringBuilder(), mid);
        return result;
    }
    
    private void palinHelper(List<String> result, List<Character> list, boolean[] used, StringBuilder perm, String mid) {
        if (perm.length() == list.size()) {
            result.add(perm.toString() + mid + perm.reverse().toString());
            perm.reverse();       //remember to reset !!!!!!!!!
            return;
        }
        
        for (int i = 0; i < list.size(); i++) {
            if (used[i])    continue;
            if (i > 0 && list.get(i-1) == list.get(i) && !used[i-1])
                continue;
                
            used[i] = true;
            perm.append(list.get(i));
            palinHelper(result, list, used, perm, mid);
            perm.deleteCharAt(perm.length()-1);
            used[i] = false;
        }
    }

    // 或者用 swap 更快..  result用hashSet
    private void dfsPermutation(Set<String>result, char[] chars, int pos, String mid) {
        if (pos == chars.length) {
            String s = new String(chars);
            result.add(s + mid + new StringBuilder(s).reverse());
            return;
        }

        for (int i = pos; i < chars.length; i++) {
            if (i > pos && chars[i - 1] == chars[i])
                continue;

            swap(chars, i, pos);
            dfsPermutation(result, chars, pos + 1, mid);
            swap(chars, i, pos);
        }
    }

    
    // 这个跟上面一样，但是用int[]来存，更快！！ 而且不需要另外的list来排序输出前半段
    public List<String> generatePalindromesFaster(String s) {
        List<String> list = new ArrayList<>();
        int numOdds = 0; // How many characters that have odd number of count
        int[] map = new int[128]; // Map from character to its frequency
        for (char c: s.toCharArray()) {
            map[c]++;
            numOdds += (map[c] & 1) == 1 ? 1 : -1;
        }
        if (numOdds > 1)   return list;
        
        String mid = "";
        int length = 0;
        for (int i = 0; i < map.length; i++) {
            if (map[i] > 0) {
                if ((map[i] & 1) == 1) { // Char with odd count will be in the middle
                    mid = "" + (char)i;
                }
                map[i] /= 2; // Cut in half since we only generate half string
                length += map[i]; // The length of half string
            }
        }
        generatePalindromesHelper(list, map, length, "", mid);
        return list;
    }
    private void generatePalindromesHelper(List<String> list, int[] map, int length, String s, String mid) {
        if (s.length() == length) {
            list.add(s + mid + new StringBuilder(s).reverse().toString());
            return;
        }
        for (int i = 0; i < map.length; i++) { // backtracking just like permutation
            if (map[i] > 0) {
                map[i]--;			// 这里相当于index.略过了重复的
                generatePalindromesHelper(list, map, length, s + (char)i, mid);
                map[i]++;
            } 
        }
    }

    private void swap(char[]ss ,int i, int j) {
        char temp = ss[i];
        ss[i] = ss[j];
        ss[j] = temp;
    }
    
    
    /**
     * 131. Palindrome Partitioning
     * given s = "aab", 返回 ["aa","b"], ["a","a","b"]  里面的字母顺序不能变
     * @param s
     * @return
     * 因为字母顺序不能变，所以跟subset很像..用个pos来存位置
     */
    public List<List<String>> partition(String s) {
        List<List<String>> result = new ArrayList<>();
        List<String> part = new ArrayList<String>();
        if (s == null) {
            return result;
        }
        partRecur(result, part, s, 0);
        return result;
    }
    
    private void partRecur(List<List<String>> result, List<String> part, String s, int start) {
        if (start == s.length()) {
            result.add(new ArrayList<String>(part));
            return;
        }
        
        for (int i = start; i < s.length(); i++) {
            if (!isPalindrome(s, start, i)) {
                continue;
            }
        
	        part.add(s.substring(start, i + 1));
	        partRecur(result, part, s, i + 1);
	        part.remove(part.size() - 1);
        }
    }
    
    private boolean isPalindrome(String s, int i, int j) {
    	while (i < j) {
            if (s.charAt(i++) != s.charAt(j--))
                return false;
        }
        return true;
    }
    
    
    /**
     * 131. Palindrome Partition
     * 先用Boolean[][] 存是否为palindrome. 
     * 然后在dfs里，if (dp[pos][i]为true, 就dfs后面
     * @param s
     * @return
     */
    public List<List<String>> partitionDP(String s) {
        List<List<String>> res = new ArrayList<>();
        boolean[][] dp = new boolean[s.length()][s.length()];
        for(int i = 0; i < s.length(); i++) {
            for(int j = 0; j <= i; j++) {
                if(s.charAt(i) == s.charAt(j) && (i - j <= 2 || dp[j+1][i-1])) {
                    dp[j][i] = true;
                }
            }
        }
        helperPal(res, new ArrayList<String>(), dp, s, 0);
        return res;
    }
    
    private void helperPal(List<List<String>> res, List<String> path, boolean[][] dp, String s, int pos) {
        if(pos == s.length()) {
            res.add(new ArrayList<>(path));
            return;
        }
        
        for(int i = pos; i < s.length(); i++) {
            if(dp[pos][i]) {
                path.add(s.substring(pos,i+1));
                helperPal(res, path, dp, s, i+1);
                path.remove(path.size()-1);
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
        int[] minC = new int[len];

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
    
    

	/** 51. N Queen
	 * 返回所有可行的solution
     * The n-queens puzzle is the problem of placing n queens on an n×n chessboard such that no two queens attack each other.
	 * @param n
	 * @return
	 * 每次只要增加Q在哪一列就行，加到List<Integer> solution/rows
	 */
	public ArrayList<String[]> solveNQueens(int n) {
		ArrayList<String[]> result = new ArrayList<String[]>();
		if (n <= 0) {
			return result;
		 }
		 search(n, new ArrayList<Integer>(), result);
		 return result;
	}
	
	private void search(int n, ArrayList<Integer> rows, ArrayList<String[]> result) {
		if (rows.size() == n) {
			 result.add(drawChessboard(rows));
			 return;
		 }
		//每行的列（格）, 所以是col(c)
		 for (int col = 0; col < n; col++) {		
			 if (!isValid(rows, col)) {
				 continue;
			 }
			 rows.add(col);
			 search(n, rows, result);			//定在某一行
			 rows.remove(rows.size() - 1);
		 }
	}
	
	private boolean isValid(ArrayList<Integer> rows, int col) {
		 int row = rows.size();
		 for (int i = 0; i < row; i++) {
	     	 // same column 。因为一个row 只会 插一列 所以一定不会same row
			//因为不同循环row是不同的所以不用判断
			 if (rows.get(i)== col) {
				 return false;
			 }
			// |列-列| == |行-行| 
			 if (Math.abs(rows.get(i) - col) == Math.abs(row - i)) {
				 return false;
			 }
/*			下面这种也行
			 // left-top to right-bottom
			 if (i - rows.get(i) == row - col) {
				 return false;
			 }
			 // right-top to left-bottom
			 if (i + rows.get(i) == row + col) {
				 return false;
			 }
*/		
		 }
		 return true;
	}
	
	private String[] drawChessboard(ArrayList<Integer> rows) {
		 String[] chessboard = new String[rows.size()];
		 for (int i = 0; i < rows.size(); i++) {
			 chessboard[i] = "";
			 for (int j = 0; j < rows.size(); j++) {
				 if (j == rows.get(i)) {
					 chessboard[i] += "Q";
				 } else {
					 chessboard[i] += ".";
				 }
			 }
		 }
		 return chessboard;
	}
	
	public List<String> drawChessBoard(List<Integer> rows) {
        List<String> sol = new ArrayList<>();
        int n = rows.size();
        for (int col : rows) {
            char[] row = new char[n];
            Arrays.fill(row, '.');
            row[col] = 'Q';
            sol.add(new String(row));
        }
        return sol;
    }
	
	
    
    /** N-Queens II 返回结果个数
     * @param n
     * @return
     */
	int result;
    public int totalNQueens(int n) {
        result = 0;
		if (n <= 0) {
			return 0;
		 }
		 ArrayList<Integer> rows = new ArrayList<Integer>();
		 search(n, rows);               //no need to pass result.
		 return result;
	}
	
	private void search(int n, ArrayList<Integer> rows) {
		if (rows.size() == n) {
		    result ++;
		    return;
		}
		
		for (int col = 0; col < n; col++) {          // column of a certain row
		    if (!isValid(rows, col)) {
		        continue;
		    }
		    rows.add(col);
		    search(n, rows);
		    rows.remove(rows.size() - 1);           //clear rows for another solution
		}
	}
    
    
    
    /**
     * 17. Letter Combinations of a Phone Number
     * Input: "23"
     * Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
     */
    public ArrayList<String> letterCombinations(String digits) {
        ArrayList<String> result = new ArrayList<String>();
        StringBuilder sb = new StringBuilder();
        
        if (digits == null) {
            return result;
        }
        
        Map<Character, char[]> map = new HashMap<Character, char[]>();
        map.put('0', new char[] {});
        map.put('1', new char[] {});
        map.put('2', new char[] { 'a', 'b', 'c' });
        map.put('3', new char[] { 'd', 'e', 'f' });
        map.put('4', new char[] { 'g', 'h', 'i' });
        map.put('5', new char[] { 'j', 'k', 'l' });
        map.put('6', new char[] { 'm', 'n', 'o' });
        map.put('7', new char[] { 'p', 'q', 'r', 's' });
        map.put('8', new char[] { 't', 'u', 'v'});
        map.put('9', new char[] { 'w', 'x', 'y', 'z' });
        
        // 也可以用 String[] 表示map
        String[] keys = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
        /*  dfs里这样
        String s = keys[digits.charAt(sb.length()) - '0'];		// str.charAt(0)-'0' 是把char格式的数字转成int
        for (int i = 0; i < s.length(); i++) {
        */
        
        letterHelper(result, map, digits, sb);
        return result;
    }
    
    private void letterHelper(ArrayList<String> result, Map<Character, char[]> map, String digits, StringBuilder sb) {
        if (sb.length() == digits.length()) {
            result.add(sb.toString());			//要toString()
            return;
        }   
        
        for (char c : map.get(digits.charAt(sb.length()))) {
            sb.append(c);                           
            letterHelper(result, map, digits, sb);     
            sb.deleteCharAt(sb.length() - 1);			//记住是deleteCharAt
        }
    }
    
    
    // 正常的iterative
    public List<String> letterCombinationsItera(String digits) {
        List<String> result = new ArrayList<String>();
        Map<Character, char[]> map = new HashMap<Character, char[]>();
        map.put('0', new char[] {});
        map.put('1', new char[] {});
        map.put('2', new char[] { 'a', 'b', 'c' });
        map.put('3', new char[] { 'd', 'e', 'f' });
        map.put('4', new char[] { 'g', 'h', 'i' });
        map.put('5', new char[] { 'j', 'k', 'l' });
        map.put('6', new char[] { 'm', 'n', 'o' });
        map.put('7', new char[] { 'p', 'q', 'r', 's' });
        map.put('8', new char[] { 't', 'u', 'v'});
        map.put('9', new char[] { 'w', 'x', 'y', 'z' });
        
        for (char c : digits.toCharArray()) {
        	List<String> newRes = new ArrayList<>();
        	for (char alpha : map.get(c)) {
        		for (String s : result) {
        			s += alpha;
        			newRes.add(s);
        		}
        	}
        	result = newRes;
        }
        
        return result;
    }
    
    
    // 用FIFO的queue来iterative
    public List<String> letterCombinationsIterative(String digits) {
        LinkedList<String> queue = new LinkedList<String>();
        if (digits == null || digits.length() == 0)     return queue;
        
        queue.add("");      //first length is 0, 否则刚开始会empty，peek出错
        
        String[] keys = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
        
        for (int i = 0; i < digits.length(); i++) {
            int x = digits.charAt(i) - '0';
            while (queue.peek().length() == i) {      //means the previous length, not after insertion
                String s = queue.remove();
                for (char c : keys[x].toCharArray()) {
                    queue.add(s + c);
                }
            }
        }
        return queue;
    }
    
    
    
    /**
     * 93. Restore IP Addresses
     * Given "25525511135",
     * return ["255.255.11.135", "255.255.111.35"]	..	可以1位或者3位
     * @param s
     * @return
     * 1. 在验证字符串是否是数字的时候，要注意0的情况，001，010，03都是非法的。所以，如果第一位取出来是0，那么我们就判断字符串是否是"0"，不是的情况都是非法的
     * 2. 取字符串的时候，注意位数不够的问题，不仅<4, 而且<s.length()
     * 3. 注意substring的范围
     * 4. 字符串转换成数字 Integer.parseInt(); 
     * 5. 别忘了IP 地址里面的 "."
     * 6. 到第4个Part的时候我们就可以整体验证剩下的所有字符串（因为第4个Part最后一定要取到结尾才是正确的字符串）
     */
    public ArrayList<String> restoreIpAddresses(String s) {
        ArrayList<String> result = new ArrayList<String>();
        if (s.length() > 12) {
            return result;
        }
        ipRecur(result, s, "", 1);
        return result;
    }
    
    private void ipRecur(ArrayList<String> result, String s, String tmp, int count) {
        if (count == 4 && isValid(s)) {		//有4段就可以
            result.add(tmp + s);		// 记住位置不能换
            return;
        }
     // 从1开始才能substring因为exclusive.  长度小于剩下的s 长
        for (int i = 1; i < 4 && i < s.length(); i++) {
            String substr = s.substring(0, i);
            if (isValid2(substr)) {
                ipRecur(result, s.substring(i), tmp + substr + '.', count + 1);
            }
        }
    }

    private void dfsIP(List<String> result, String input, String str, int pos, int count) {

        // 如果在dfsIP里判断 count==3的情况，就可以用这个 无需再判断valid
        // if (count == 4 && pos == input.length()) {

        if (count == 3 && pos < input.length()) {
            String sub = input.substring(pos);
            if (isValid2(sub)) {
                result.add(str + sub);
            }

            return;
        }

        for (int i = 1; i <= 3 && pos + i <= input.length(); i++) {
            String sub = input.substring(pos, pos + i);
            if (isValid2(sub)) {
                // dfsIP(result, input, str + sub + (count == 3 ? "" : "."), pos + i, count + 1);
                dfsIP(result, input, str + sub + ".", pos + i, count + 1);
            }
        }
    }
    
    private boolean isValid2(String s) {
        if (s.charAt(0) == '0' && s.length() > 1)
            return false;

        return Integer.parseInt(s) <= 255;
    }
    
    // 更快
    public List<String> restoreIpAddresses2(String s) {
        List<String> list = new ArrayList<>();
        int len = s.length();
        
        // 记得每个是 <= x + 3 && i < len - 2
        for (int i = 1; i <= 3 && i < len - 2; i++) {
            for (int j = i + 1; j <= i + 3 && j < len - 1; j++) {
                for (int k = j + 1; k <= j + 3 && k < len; k++) {
                    String s1 = s.substring(0, i);
                    String s2 = s.substring(i, j);
                    String s3 = s.substring(j, k);
                    String s4 = s.substring(k, len);
                    if (isValid(s1) && isValid(s2) && isValid(s3) && isValid(s4)) {
                        list.add(s1 + '.' + s2 + '.' + s3 + '.' + s4);   
                    }
                }
            }
        }
        return list;
    }
    
    public boolean isValid(String s) {
        if  (s.length() == 0 || s.length() > 3 || Integer.parseInt(s) > 255 ||
                (s.charAt(0) == '0' && s.length() > 1)) {
                    return false;
        }
        return true;
    }
    
    
    /**
     * 22. Generate Parentheses
     * given n = 3。 n pairs 
     * solution："((()))", "(()())", "(())()", "()(())", "()()()"
     * @param n
     * @return
     * 第一种，像之前的程序一样，我们把String tmp写在dfs里面，每回Return删去刚刚添加的字符
     * 第二种，直接把tmp ＋（添加的字符）当参数传入下一层调用函数，这样返回后在上一层是之前传入的参数，不用删字符
     * 先放左，再放右(要判断left < right)
     */
    public ArrayList<String> generateParenthesis(int n) {
        ArrayList<String> result = new ArrayList<String>();
        if (n <= 0) {
            return result;
        }
        parenthRecur(result, "", n, n);
        return result;
    }
    
    private void parenthRecur(ArrayList<String> result, String paren, int left, int right) {
        if (left == 0 && right == 0) {
            result.add(paren);
            return;
        }
        
        if (left > 0) {
            parenthRecur(result, paren + "(", left-1, right);       //can't left--
        }
/*        if (left > 0) {
            String ntmp = paren + "(";
            parenthRecur(result, ntmp, left-1, right);       //can't left--
            paren = ntmp.substring(0, ntmp.length() - 1);      //no remove/delete方法所以只能substring
        }
*/
        if (left < right) {		// !!! 不是只right>0就行
            parenthRecur(result, paren + ")", left, right-1);
        }
    }
    
    
    // iterative
    // 又是一样的模板.. 最里面在之前(n-1)生成的结果里，for k每个地方插入().
    // 但有可能重复所以用hashset去重
    public List<String> generateParenthesisIterative(int n) {
        List<String> result = new ArrayList<String>();
        if (n <= 0) {
            return result;
        }
        
        result.add("()");

        for (int i = 1; i < n; i++) {           // i=1 开始，因为初始化已经用掉一个'('了
            Set<String> set = new HashSet<>();          // 为了防止重复
            for (String s : result) {
                for (int k = 0; k < s.length(); k++) {      // 在已有的string里不同位置插入()  一对对地加
                    set.add(s.substring(0, k) + "()" + s.substring(k));
                }
            }
            result.clear();
            result.addAll(set);
        }
   
        return result;
    }
	
    
    /** 241. Different Ways to Add Parentheses
     * 在算式中加括号，看所有结果是怎样的。只有 + - *
     * Input: "2*3-4*5"

		(2*(3-(4*5))) = -34
		((2*3)-(4*5)) = -14
		((2*(3-4))*5) = -10
		(2*((3-4)*5)) = -10
		(((2*3)-4)*5) = 10
		Output: [-34, -14, -10, -10, 10]
     * @param input
     * @return
     * 这题是要用dfs, 但是思想是 divide & conquerer
     * 1. 按照当前运算符 + - *来划分左右str
     * 2. 调用dfs得到 左右 两部分的result
     * 3. 再用当前 运算符来+-*  左右两部分
     * 		这里需要 2层for. for(left for(right)) 因为left跟right各自有不同结果，所以要都遍历
     * 
     * 这里挺多input.substring会有重复运算，所以放在cache里 memorize.
     * 记得最后要把结果放进cache里
     * 
     * !! 注意要有返回条件，是在for循环以后.. 
     * 		如果result为空就返回int, 这时传进来的只可能1位的数字。(-+*不会传进来）
     */
    public List<Integer> diffWaysToCompute(String input) {
        Map<String, List<Integer>> cache = new HashMap<>();
        return dfsCache(input, cache);
    }
    
    private List<Integer> dfsCache(String input, Map<String, List<Integer>> cache){
        if (cache.containsKey(input)) {
            return cache.get(input);
        }
        
        List<Integer> result = new ArrayList<>();
        
        for (int i = 0; i < input.length(); i++) {
            char c = input.charAt(i);
            if (c == '+' || c == '-' || c == '*') {
                String s1 = input.substring(0, i);
                String s2 = input.substring(i + 1);
                List<Integer> left = dfsCache(s1, cache);		//这里 dfs left & right
                List<Integer> right = dfsCache(s2, cache);

                for (int x : left) {			//这里也要 2层 for循环，因为left跟right各自有不同结果，所以要都遍历
                    for (int y : right) {
                        if (c == '+') {
                            result.add(x + y);
                        } else if (c == '-') {
                            result.add(x - y);
                        } else if (c == '*') {
                            result.add(x * y);
                        }
                    }
                }
            }
        }
        // 这是返回条件.. 只剩下数字，没有运算符
        if (result.size() == 0) {
            result.add(Integer.parseInt(input));
        }
        
        cache.put(input, result);   //!!!!记得加cache里
        return result;
    }
    
    
    
    /**
     * 494. Target Sum
     * 在每个数字前后任意加 + 或 -，使得和为target。看总共有几种ways
     * Input: nums is [1, 1, 1, 1, 1], S is 3.
     * Output: 5
     * Explanation:
     *
     * -1+1+1+1+1 = 3
     * +1-1+1+1+1 = 3
     * +1+1-1+1+1 = 3
     * +1+1+1-1+1 = 3
     * +1+1+1+1-1 = 3
     *
     * There are 5 ways to assign symbols to make the sum of nums be target 3.
     *
     * 最简单的就DFS.. a + b = target, 或者 -a + b = target.. 所以只用求2种可能
     */
    public int findTargetSumWays(int[] nums, int target) {
   //     int[] res = new int[1];
        Map<String, Integer> cache = new HashMap<>();       // pos->num, value是有几种ways
        return helper(nums, cache, target, 0);
   //     return res[0];
    }
    
    // 最naive的方法, 也可以return int...跟下面memorize一样
    public void helper(int[] nums, int target, int pos){
        if (pos == nums.length) {
            if (target == 0)
                result++;
            return;
        }
        helper(nums, target + nums[pos], pos + 1);
        helper(nums, target - nums[pos], pos + 1);
    }
    
    
    // 加了Cache.. 不过也挺慢
    public int helper(int[] nums, Map<String, Integer> cache, int sum, int pos) {
        String str = pos + "->" + sum;
        if (cache.containsKey(sum)) {
            return cache.get(str);
        }
        
        if (pos == nums.length) {
            if (sum == 0)   return 1;
            else            return 0;
        }
        
        int add = helper(nums, cache, sum - nums[pos], pos + 1);
        int minus = helper(nums,  cache, sum + nums[pos], pos + 1);
        
        cache.put(str, add + minus);
        return add + minus;
    }


    /** 494. Target Sum
     * Given nums = [1, 2, 3, 4, 5] and target = 3 then one possible solution is +1-2+3-4+5 = 3
     *
     * Here positive subset is P = [1, 3, 5] and negative subset is N = [2, 4]
     * sum(P) - sum(N) = target
     * sum(P) + sum(N) + sum(P) - sum(N) = target + sum(P) + sum(N)
     *  2 * sum(P) = target + sum(nums)
     * 那么问题就变成 Find a subset P of nums such that sum(P) = (target + sum(nums)) / 2
     */
    public int findTargetSumWaysDP(int[] nums, int target) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        
        // 如果 sum+target 不是 subset sum的2倍，那就无解
        if (target > sum || (sum + target) % 2 == 1)    return 0;
        
        // 要找到sum(P) = (target + sum(nums)) / 2
        return subsetSum(nums, (sum + target) / 2);
    }
    
    private int subsetSum(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int num : nums) {
             for (int t = target; t >= num; t--) {      //记得从后往前
                dp[t] += dp[t - num];
            }
        }
        return dp[target];
    }
    
    
    /** 494. Target Sum
     * dp[i][j]表示前i位的target sum为j, 有几种方式
     * 跟背包很像
     * @param nums
     * @param target
     * @return
     */
    public int findTargetSumWaysDP1(int[] nums, int target) {
        int sum = 0;
        for (int n : nums) {
            sum += n;
        }
        if (sum < Math.abs(target))      
            return 0;
        
        int len = nums.length;
        int[][] dp = new int[len][2 * sum + 1];
        
        // 初始化
        if (nums[0] == 0) {
            dp[0][sum] = 2;
        } else {
            dp[0][sum - nums[0]] = 1;
            dp[0][sum + nums[0]] = 1;
        }
        
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j <= 2 * sum; j++) {		// sum*2. 
                if (j - nums[i] >= 0)       dp[i][j] += dp[i - 1][j - nums[i]];
                if (j + nums[i] <= 2*sum)   dp[i][j] += dp[i - 1][j + nums[i]];
            }
        }
  //      return dp[len - 1][target + sum];
        
        // 或者用一维数组
        int[] dp1 = new int[2*sum+1];
        dp1[0+sum] = 1;
        for(int i = 0; i<nums.length; i++){
            int[] next = new int[2*sum+1];
            for(int k = 0; k<2*sum+1; k++){
                if(dp1[k]!=0){
                    next[k + nums[i]] += dp1[k];
                    next[k - nums[i]] += dp1[k];
                }
            }
            dp1 = next;
        }
        return dp1[sum + target];
    }
    
    
    /** 282. Expression Add Operators
     * 0-9 add binary operators (not unary) +, -, or *
     * "105", 5 -> ["1*0+5","10-5"]   可以组成不同长度的digit，比如10..
     * @param num
     * @param target
     * @return
     *
     * 正常的dfs.. 要注意几点
     * 1. overflow，所以要long cur
     * 2. 0的出现，可以 10，101， 也可以0自己，但不能有0开头的 01。
     * 		一旦最后那样 num.charAt(pos) == '0' && i != pos，就要停。
     * 3. 算乘数时，记得另外一个变量存multiply表示 之前pre要乘的数。total - pre + pre * cur
     * 		if you have a sequence of 12345 and you have proceeded to 1 + 2 + 3, now your eval is 6 right? 
     * 		If you want to add a * between 3 and 4, you would take 3 as the digit to be multiplied, 
     * 			so you want to take it out from the existing eval. 
     * 		You have 1 + 2 + 3 * 4 and the eval now is (1 + 2 + 3) - 3 + (3 * 4)
     * 4. 用char[] 来代替str.substring, 用stringBuilder都更快 （看dfs_solution的）
     * 		long cur = Long.parseLong(new String(num, pos, i - pos + 1));  //最后是count个数，中间是从哪开始
     */
    public List<String> addOperators(String num, int target) {
        List<String> result = new ArrayList<>();
        helper(result, num, target, "", 0, 0, 0);
        return result;
    }
    
    public void helper (List<String> result, String num, int target, String path, int pos, long total, long pre) {
        if (pos == num.length()) {
            if (total == target) {
                result.add(path);
            }
            return;
        }
        
        for (int i = pos; i < num.length(); i++) {
            //  starts with 0 && i在pos后面，就是说 01的情况
        	if (num.charAt(pos) == '0' && i != pos)
                break;      // don't want to proceed if  "01", but ok if just "0", then i==pos
        	
            Long cur = Long.parseLong(num.substring(pos, i + 1));    // substring(pos, i+1)
            
            if (pos == 0) {		//只是第一个数.. 其他情况是后面else的
                helper(result, num, target, path + cur, i + 1, cur, cur);   
            } else {
                helper(result, num, target, path + "+" + cur, i + 1, total + cur, cur);
                helper(result, num, target, path + "-" + cur, i + 1, total - cur, -cur);    //pre = -cur
                helper(result, num, target, path + "*" + cur, i + 1, total - pre + pre * cur, cur * pre);
            }
        }
    }
    
    //用char[] 和 StringBuilder更快
    public void helper (List<String> result, char[] num, int target, StringBuilder path, int pos, long total, long multiply) {
        if (pos == num.length) {
            if (total == target) {
                result.add(path.toString());
            }
            return;
        }
        
        long cur = 0;
        for (int i = pos; i < num.length; i++) {
        //  starts with 0 && i在pos后面，就是说 01的情况
            if (num[pos] == '0' && i != pos)
                break;      // don't want to proceed if  "01", but ok if just "0", then i==pos
            cur = cur * 10 + (num[i] - '0');    // substring(pos, i+1)
            
            int len = path.length();
            if (pos == 0) {
                helper(result, num, target, path.append(cur), i + 1, cur, cur);  
                path.setLength(len);
            } else {
                helper(result, num, target, path.append("+").append(cur), i + 1, total + cur, cur);
                path.setLength(len);
                helper(result, num, target, path.append("-").append(cur), i + 1, total - cur, -cur);    //mult = -cur
                path.setLength(len);
                helper(result, num, target, path.append("*").append(cur), i + 1, total - multiply + multiply * cur, cur * multiply);
                path.setLength(len);		//sb记得要删掉新加的，回到之前的length
            }
        }
    }
   
    
    
    /**
     * 246. Strobogrammatic Number
     * A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).
     * 判断num是否为那种
     * 比如 "69"  "88"
     */
    public boolean isStrobogrammatic(String num) {
        int i = 0, j = num.length() - 1;
        while (i <= j) {
            if (num.charAt(i) == num.charAt(j) && (num.charAt(i) == '0' || num.charAt(i) == '1' || num.charAt(i) == '8')) {
                i++;
                j--;
            } else if ((num.charAt(i) == '6' && num.charAt(j) == '9') || (num.charAt(i) == '9' && num.charAt(j) == '6')) {
                i++;
                j--;
            } else {
                return false;
            }
        }
        
        // shorter 
        for (int m=0, n=num.length()-1; i <= j; i++, j--)
            if (!"00 11 88 696".contains(num.charAt(m) + "" + num.charAt(n)))
                return false;
        return true;
    }
    
    
    
    /** 247. Strobogrammatic Number II
     * Given n = 2, return ["11","69","88","96"]
     * @param n
     * @return
     * 记得判断 如果在最外面 i = n 时，不要用0
     * 
     * 1. 先根据n的奇偶来初始化result，从中间开始，往外扩散
     * 2. for循环result，并且建一个新的tmp, 每次在外面加2个数给中间的string
     * 3. 加完tmp后再赋值给result，这样可以下一次再for result加数
     */
    public List<String> findStrobogrammatic(int n) {
        List<String> odd = Arrays.asList("0", "1", "8");
        List<String> even = Arrays.asList("");
        List<String> result = n % 2 == 0 ? even : odd;		// 从中间开始往外扩散
        
        for (int i = n % 2 + 2; i <= n; i += 2) {		//i从2或3开始，因为下面要每次加2个数
            List<String> list = new ArrayList<>();
            for (String s : result) {
                if (i != n)     list.add("0" + s + "0");
                
                list.add("1" + s + "1");
                list.add("6" + s + "9");
                list.add("8" + s + "8");
                list.add("9" + s + "6");
            }
            result = list;
        }
        return result;
        
   //    return recHelper(n, n);		recursion
    }
    
    
    // recursion也比较好理解
    public List<String> recHelper(int n, int outLayer) {
        if (n == 0)     return new ArrayList<String>(Arrays.asList(""));
        if (n == 1)     return new ArrayList<String>(Arrays.asList("0", "1", "8"));

        // 递归拿到 中间的list结果
        List<String> list = recHelper(n - 2, outLayer);		//当是list已经拿到中间的string了


        // 当前n 创建新result，在之前得到的list外层加上新的数字
        List<String> result = new ArrayList<>();
        
        for (String s : list) {						//在里层string的外层加2个数
            if (n != outLayer)
                result.add("0" + s + "0");      // 0 cannot at the begining, so n != m can avoid
            
            result.add("1" + s + "1");
            result.add("6" + s + "9");
            result.add("8" + s + "8");
            result.add("9" + s + "6");
        }
        return result;
    }
    
    
    /**
     * 248. Strobogrammatic Number III
     * 返回符合范围的个数  Given low = "50", high = "100", return 3. 
     * Because 69, 88, and 96 are three strobogrammatic numbers.
     *
     * naive做法就是 基于II的，根据length作为n来dfs生成多种结果.. 最后去掉<low, >high的情况
     */
    public int strobogrammaticInRangeNaive(String low, String high) {
        List<String> list = new ArrayList<>();
        for (int n = low.length(); n <= high.length(); n++) {
            list.addAll(recHelper(n, n));
        }
        
        int count = 0;
        for (String num : list) {
            if ((num.length() == low.length() && num.compareTo(low) < 0) || 
                (num.length() == high.length() && num.compareTo(high) > 0)) {
                    continue;
                }
            count++;
        }
        return count;
    }


    /**
     * 248. Strobogrammatic Number III
     *
     * 这个快点 。
     * 其实差不多，也是根据不同length长度  call dfs来生成不同number..
     * 这里用char[]比较快
     */
    private static final char[][] PAIRS = {{'0','0'},{'1','1'},{'8','8'},{'6','9'},{'9','6'}};

    public int strobogrammaticInRange(String low, String high) {
        if (low == null || high == null || low.length() > high.length() ||
                (low.length() == high.length() && low.compareTo(high) > 0))
            return 0;

        int count = 0;

        // 每种length长度 call 一次dfs来生成number
        for(int len = low.length(); len <= high.length(); len++) {
            count += dfsStrobogrammatic(low, high, new char[len], 0, len - 1);
        }

        return count;
    }

    private int dfsStrobogrammatic(String low, String high, char[] ch, int left, int right) {
        // 表示已经从外往中间填好值了
        if (left > right) {
            String s = String.valueOf(ch);
            if ((s.length() == low.length() && s.compareTo(low) < 0 ) ||
                    (s.length() == high.length() && s.compareTo(high) > 0)) {
                return 0;
            } else {
                return 1;
            }
        }

        int count = 0;

        // 从外往里填可能的值
        for (char[] p : PAIRS) {
            ch[left] = p[0];
            ch[right] = p[1];

            if (ch.length > 1 && ch[0] == '0')  continue;       // 最外层不能是0
            if (left == right && p[0] != p[1])  continue;       // 中间的奇数

            count += dfsStrobogrammatic(low, high, ch, left + 1, right - 1);
        }

        return count;
    }
    
    
    /**
     * 0,1,? 其中?代表0或1  --- 简单
     * @param str
     * @return
     */
    public static ArrayList<String> findPermutation (String str) {
		ArrayList<String> res = new ArrayList<String>();
		StringBuilder tmp = new StringBuilder();
		findHelper(res, tmp, str, 0);
		return res;
	}
	
	public static void findHelper (ArrayList<String> res, StringBuilder tmp, String str, int pos) {	
		if(pos == str.length()) {
			res.add(tmp.toString());
			return;
		}
		
		if(str.charAt(pos) == '?') {
			tmp.append('0');
			findHelper(res, tmp, str, pos + 1);
			tmp.deleteCharAt(tmp.length() - 1);
			
			tmp.append('1');
			findHelper(res, tmp, str, pos + 1);
			tmp.deleteCharAt(tmp.length() - 1);
		} else {
			tmp.append(str.charAt(pos));
			findHelper(res, tmp, str, pos + 1);
			tmp.deleteCharAt(tmp.length() - 1);
		}
	}
	
	
	

    /**
     * 79. Word Search
     * Given a 2D board and a word, find if the word exists in the grid.
	  可以上下左右相邻的找，但是每次最多use一次，所以要设'#'来标记used （代替额外的visited[])
     * @param board
     * @param word
     * @return
     * !!!!!记得回溯！ 记得pos==length时返回TRUE结束
     * 设'#'来标记used （代替额外的visited[])
     */
    public boolean exist(char[][] board, String word) {
        int rows = board.length;
        int cols = board[0].length;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (word.charAt(0) == board[i][j]) {
                    if (existDFS(board, i, j, word, 0))
                        return true;
                }
            }
        }
        return false;
    }
        
	int[] d = {0, 1, 0, -1, 0};
    private boolean existDFS(char[][] board, int i, int j, String word, int pos) {
        if (pos == word.length()) {		//记得！！！！
            return true;
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || word.charAt(pos) != board[i][j]) {
            return false;
        }
        
        board[i][j] = '#';  //mark as used
        for (int l = 0; l < 4; l++) {
            if (existDFS(board, i + d[l], j + d[l+1], word, pos + 1))
                return true;				//这里一有TRUE就可以提前return..因为这样就visited过了，不用再Reset回来
        }
        /*//也行，上面的简单点
        if (existDFS(board, i + 1, j, word, pos + 1) ||
            existDFS(board, i, j + 1, word, pos + 1) ||
            existDFS(board, i - 1, j, word, pos + 1) ||
            existDFS(board, i, j - 1, word, pos + 1)) {
                return true;
        }*/
        
     // reset , backtracking if that's not used 能到下面这步说明上面都是false，所以这个i,j没用到
        board[i][j] = word.charAt(pos);
        
        return false;
    }
    
        
    /** 212. Word Search II
     * words = ["oath","pea","eat","rain"] and board =["oaan","etae","ihkr","iflv"]矩阵
     * 返回["eat","oath"]
     * @param board
     * @param words
     * @return
     * 1. 普通dfs会超时
     * 2. 记得加word到list时，不能重复加，因为dfs会调用4次，所以要去重
     * 3. How do we instantly know the current character is invalid? HashMap?
		  How do we instantly know what's the next valid character? LinkedList?
		  But the next character can be chosen from a list of characters. "Mutil-LinkedList"?
		Combing them, Trie is the natural choice
	 * 为何用Trie树? 普通DFS每次都重新搜后面的单词，慢。
	 * 如果prefix一样的话，比如"aa", "aab", "aac"，那Trie树就能继续往下搜，而不用从头"a"开始
	 * 
	 * 4. 重复4次加list的话，用Trie里面一个word来存，如果还没isEnd, 那就默认null
     */
    public List<String> findWords(char[][] board, String[] words) {
        List<String> list = new ArrayList<>();
        if (board == null || board.length == 0 || words == null || words.length == 0)
            return list;
        
        TrieNode root = buildTrie(words);		//建树
        
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                dfs(board, list, root, i, j);
            }
        }
        return list;
    }
    //跟word search I 很像.. 另一个版本稍有改动，看dataStructure.java
    public void dfs(char[][] board, List<String> list, TrieNode node, int i, int j) {
        if (node.word != null) {
            list.add(node.word);
            node.word = null;       //prevent duplicate去重
        }			                     //不用return，可能还有下一个词符合，是前缀相同
        
        if (i < 0 || j < 0 || i >= board.length || j >= board[0].length)
            return;
        
        char c = board[i][j];
        //  visisted ||   not in dict, doesn't match
        if (c == '#' || node.children[c - 'a'] == null)
            return;
        
        board[i][j] = '#';
        for (int t = 0; t < 4; t++) {	//记得传下一个node
            dfs(board, list, node.children[c-'a'], i + d[t], j + d[t+1]);
        }
        board[i][j] = c;
    }
    
    public TrieNode buildTrie(String[] words) {
        TrieNode root = new TrieNode();
        for (String w : words) {
            TrieNode node = root;				//每个单词都从root开始走
            for (char c : w.toCharArray()) {
                int i = c - 'a';
                if (node.children[i] == null) {			
                    node.children[i] = new TrieNode();
                }
                node = node.children[i];		//往下走
            }
            node.word = w;      //at the end(leaf), add whole word instead of just isEnd
        }
        return root;
    }
    
    class TrieNode {
        String word;
        TrieNode[] children = new TrieNode[26];
    }
    
    public List<String> findWordsTLE(char[][] board, String[] words) {
        List<String> list = new ArrayList<>();
        if (board == null || board.length == 0 || words == null || words.length == 0)
            return list;
        
        int m = board.length;
        int n = board[0].length;
        for (int k = 0; k < words.length; k++) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (board[i][j] == words[k].charAt(0)) {
                        dfs(board, list, words[k], i, j, 0);
                    }
                }
            }
        }
        return list;
    }
    
    public void dfs(char[][] board, List<String> list, String word, int i, int j, int k) {
        if (k == word.length()) {
            if (!list.contains(word)) {
                list.add(word);
            }
            return;
        }
        
        if (i < 0 || j < 0 || i >= board.length || j >= board[0].length || word.charAt(k) != board[i][j])
            return;
        
        board[i][j] = '#';
        for (int t = 0; t < 4; t++) {
            dfs(board, list, word, i + d[t], j + d[t+1], k + 1);
        }
        board[i][j] = word.charAt(k);
    }

    
    
    /**
     * 279. Perfect Squares
     * Given a 正整数 n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.
     * For example, given n = 12, return 3 because 12 = 4 + 4 + 4; given n = 13, return 2 because 13 = 4 + 9.
     *
     * DP, 容易理解 dp[j] = Math.min(dp[j], dp[j - i * i] + 1)   [i~j]
     */
    public int numSquaresDP(int n) {
        int[] dp = new int[n + 1];

        for (int j = 1; j <= n; j++) {
            dp[j] = j;
            for (int i = 1; i * i <= j; i++) {
                dp[j] = Math.min(dp[j], dp[j - i * i] + 1);     // +1 说明这次j*j刚好能行
            }
        }
        return dp[n];
    }


    /**
     * 279. Perfect Squares
     * bfs - 因为是算 least number, 也就是求最短路径，那就用BFS
     *
     * queue里面存的是 完全平方数组成的sum。
     * 1. 所以每次Poll完以后，再for循环找可能的perfect squares, 记得 for (int i = 1; i * i <= n; i++) 这么找
     * 2. 当 poll + i * i == n 表示找到了，否则小于的话，就放到q里
     *
     * !!!! 放进q里前要确保没visit过，比如 1->4, 和4->1这种组合的情况，sum都是5，不要重复
     */
    public int numSquares(int n) {
        Queue<Integer> q = new LinkedList<>();
        boolean[] visited = new boolean[n + 1];
        q.add(0);       //dummy to start
        int level = 0;

        while (!q.isEmpty()) {
            int size = q.size();
            level++;
            while (size-- > 0) {
                int cur = q.poll();
                for (int i = 1; i * i <= n; i++) {
                    int sum = cur + i * i;
                    if (sum == n) {
                        return level;
                    } else if (sum > n) {
                        break;
                    }
                    // sum < n
                    if (!visited[sum]) {    	// 避免重复的组合比如 1->4, 4->1
                        q.add(sum);			//是加sum
                        visited[sum] = true;
                    }
                }
            }
        }
        return level;
    }


    // 用DFS就很慢
    public int numSquaresDFS(int n) {
        return dfsSquare(n, 0, n);
    }
    
    public int dfsSquare(int n, int level, int min) {
        if (n == 0 || level == min)		// 记得如果level == min也要提前return，说明找到了
            return level;
        
        for (int i = (int) Math.sqrt(n); i > 0; i--) {
            min = dfsSquare(n - i*i, level + 1, min);
        }
        return min;
    }


    /**
     * 489. Robot Room Cleaner
     * 只有robot这个object.. 不知道房间怎么样..
     * 需要clean整个房间.. 0为障碍，1为空地.. boolean move() 返回是否可以往前走
     *
     * DFS 遍历
     *
     * 因为我们不知道board.. 所以只能把(i,j)变成string来放set里看有没visited
     * 每次根据当前方向curDir来调整.. 而且dfs完后需要turnRight()来换方向
     *
     * 记得backtrack把robot reset 回original pos & direction
     *
     */
    public void cleanRoom(Robot robot) {
        Set<String> visited = new HashSet<>();
        cleanHelper(robot, visited, 0, 0, 0);
    }

    int[] dir = {0, 1, 0, -1, 0};

    private void cleanHelper(Robot robot, Set<String> visited, int x, int y, int curDir) {
        String pos = x + "," + y;
        if (visited.contains(pos))
            return;

        visited.add(pos);
        robot.clean();

        for (int i = 0; i < 4; i++) {           // 下面的方向跟i 无关，只跟curDir有关
            if (robot.move()) {
                int nx = x + dir[curDir];
                int ny = y + dir[curDir + 1];

                cleanHelper(robot, visited, nx, ny, curDir);            // 往前走，同样的方向move

                // backtrack and reset to original pos & face direction
                goBackWithOrigDirection(robot);
            }

            robot.turnRight();          // 换方向
            curDir = (curDir + 1) % 4;          // 怕越界
        }
    }

    private void goBackWithOrigDirection(Robot robot) {
        robot.turnRight();
        robot.turnRight();
        robot.move();
        robot.turnRight();
        robot.turnRight();
    }


    class Robot {
        int id;

        public Robot() {
            id = id;
        }

        public void turnRight() {}
        public void clean() {}

        public boolean move() {
            return true;
        }

    }
    
    
    
    
	public static void main(String[] args) {
		DFS_Subset sol = new DFS_Subset();
		char[][] board = {{'o','a','a','n'}, {'e','t','a','e'},{'i','h','k','r'},{'i','f','l','y'}};
		String[] words = {"oath","pea","eat","rain"};
		List<String> al = sol.findWords(board, words);
//		ArrayList<String> al = sol.generateParenthesis(3);
		System.out.println(sol.combinationSumIII(3, 7));
		
		
		
	}
}
