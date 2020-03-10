import java.util.*;


/**
 * @author Yan
 *
 */
/**
 * @author Yan
 *
 */
public class StackQueue {
	
	public StackQueue() {
		
	}

    /*341 Flattern NextedList Iterator 
    Input: [[1,1],2,[1,1]]
    Output: [1,1,2,1,1]
    Explanation: By calling next repeatedly until hasNext returns false, 
                 the order of elements returned by next should be: [1,1,2,1,1].*/
    /**
     * // This is the interface that allows for creating nested lists.
     * // You should not implement it, or speculate about its implementation
     * public interface NestedInteger {
     *
     *     // @return true if this NestedInteger holds a single integer, rather than a nested list.
     *     public boolean isInteger();
     *
     *     // @return the single integer that this NestedInteger holds, if it holds a single integer
     *     // Return null if this NestedInteger holds a nested list
     *     public Integer getInteger();
     *
     *     // @return the nested list that this NestedInteger holds, if it holds a nested list
     *     // Return null if this NestedInteger holds a single integer
     *     public List<NestedInteger> getList();
     * }
     */
    public class NestedIterator implements Iterator<Integer> {
        Stack<NestedInteger> stack = new Stack<>();
        public NestedInteger(List<NestedInteger> nestedList) {
            //start from the last one, last in first out
            for (int i = nestedList.length - 1; i >= 0 ; i--){
                stack.push(nestedList.get(i));
            }
        }

        @Override
        public Integer next(){
            return stack.pop().getInteger();
        }

        @Override
        public boolean hasNext(){
            while (!stack.isEmpty()){
                NestedInteger curr = stack.peek();
                if(curr.isInteger()){
                    return true;
                }
                //if it is an integer ,then find, otherwise pop, try to search more
                stack.pop();
                //again, add from the last item, so LIFO
                for(int i = curr.getList().size() - 1; i >= 0; i--) {
                    stack.push(curr.getList().get(i));
                }
            }
            return false;
        }

    }
	/** Valid Parentheses
	 * The brackets must close in the correct order
	 * "()" and "(){[]}" are all valid 对
	 * but "(]" and "([)]" 错
	 * @param s
	 * @return
	 * 用stack。因为能把最close的括号POP出来，看能否与外面的配对
	 */
	public boolean isValid(String s) {
        Stack<Character> stack = new Stack<Character>();
        char c;
        for (int i = 0; i < s.length(); i++) {
            c = s.charAt(i);
            // faster
            if (c == '(') {
                stack.push(')');
            } else if (c == '[') {
                stack.push(']');
            } else if (c == '{') {
                stack.push('}');
            } else if (stack.isEmpty() || stack.pop() != c) {
                return false;
            }

            
            /*  //普通方法
            if (c == '(' || c == '[' || c == '{') {
         // if ("({[".contains(String.valueOf(c))) {  这样也行
                stack.push(c);					//前括号就放进去
            } else if (c == ')' || c == ']' || c == '}') {
                if (stack.isEmpty()) {          // 前括号pop完，说明match完了。
                    return false;				 // 现在外面还有)]的话，说明后括号多了
                }
                
                char pop = stack.pop();		// pop前括号出来match
                if (pop == '(' && c != ')' ||
                    pop == '[' && c != ']' ||
                    pop == '{' && c != '}') {
                        return false;
                }
            } */
        }
        return stack.isEmpty();               //if size=0,前括号都POP出来且 match
    }
	
	
	
	/** 32. Longest Valid Parentheses
	 * ")()())",  longest valid parentheses substring is "()()", 返回 4.
	 * @param s
	 * @return
	 * 用stack存index!!! 只有)跟peek是( match时才pop，max=(max, i-peek)。其他情况都push进去
	 * 技巧是：match时pop，算index substring相差值
	 */
	public int longestValidParentheses(String s) {
        Stack<Integer> stack = new Stack<Integer>();
        int max = 0;
        stack.push(-1);		//如果第一对match 那i=1, max=1-(-1) = 2
        
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {                    //match
                if (stack.peek() != -1 && s.charAt(stack.peek()) == '(') {
                    stack.pop();
                    max = Math.max(max, i - stack.peek());
                } else {
                    stack.push(i);
                }
            }   	//也可以吧上面的push都合并在else里，if判断多点+ charAt==')'
        }
        
        return max;
    }
	
	
	/** 32. Longest Valid Parentheses
	 * 用two pass 扫两遍。 每次算left和right的个数
	 * 1. 从左到右扫..如果left=right，那就算max，left*2. if right太多，那就不匹配，全Reset为0
	 * 2. 从right往前扫.. 也是一样.. left太多就不匹配，重置为0
	 * 
	 * 扫两遍是因为 如果从左开始扫，可能 ((). 这时是ok的，但是left多，所以会漏掉这情况
	 */
	public int longestValidParenthesesTwoPass(String s) {
        int max = 0;
        int left = 0, right = 0;        //存左右括号的个数
        // 从左 往右扫
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                left++;
            } else {
                right++;
            }
            // 判断left和right个数
            if (left == right) {
                max = Math.max(max, left * 2);
            } else if (right > left) {
                left = right = 0;
            }
        }
        
        // 从right边开始扫
        left = 0; right = 0;       
        for (int i = s.length() - 1; i >= 0; i--) {
            if (s.charAt(i) == '(') {
                left++;
            } else {
                right++;
            }
            if (left == right) {
                max = Math.max(max, left * 2);
            } else if (left > right) {
                left = right = 0;
            }
        }
        return max;
    }
	
	
	/** 32. Longest Valid Parentheses
	 * 用DP做法。找最长valid的前面一个，看是否能跟现在的i match。
	 * 前面那个就是 i - dp[i-1] - 1  (dp[i-1]是max valid长度）
	 * @param s
	 * @return
	 */
	public int longestValidParenthesesDP(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        
        int len = s.length();
        
        int[] dp = new int[len];
        int max = 0;
        
        for (int i = 1; i < len; i++) {
            if (s.charAt(i) == ')' ) {
                if (s.charAt(i-1) == '(') {		// 正好()匹配，就直接+2
                    dp[i] = i >= 2 ? dp[i-2] + 2 : 2;   //dp[i-2], not i-1
                    max = Math.max(max, dp[i]);
                } else {		// ))的情况，就要找最前面，看是否有 ( 来match
                    // dp[i-1] is valid longest, so see the pre of it to match
                    if (i-dp[i-1]-1 >= 0 && s.charAt(i - dp[i-1] - 1) == '(') {
                        int pre = i - dp[i-1] - 2 >= 0 ? dp[i - dp[i-1] - 2] : 0;
                        dp[i] = dp[i-1] + 2 + pre;
                        max = Math.max(max, dp[i]); //updae max
                    }
                }
            }
        }
        
        //合并版本
        s = ")" + s;    //add a prefix to avoid duplicate code & corner case
        //或者可以不加prefix，length不加1. 这样下面要判断是否 >= 0 
   //     int[] dp = new int[len + 1];
        
        for (int i = 1; i < len; i++) {
                // dp[i-1] is valid longest, so see the pre of it to match
            if (s.charAt(i) == ')' && s.charAt(i - dp[i-1] - 1) == '(') {
                dp[i] = dp[i-1] + 2 + dp[i - dp[i-1] - 2];
                dp[len] = Math.max(dp[len], dp[i]); //updae max
            }
        }
        
        return dp[len];
	}
	
	
	
	/** 只有'(' 和 ')'。看是否组成valid的括号对
	 * 那就只用count来记录'('就++ .. 是')'就--。 如果 count < 0, 说明太多')'.. 最后记得查count==0. 看有没多余的'('
	 * @param s
	 * @return
	 */
	public boolean isValidPar(String s) {
        int count = 0;
        for  (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                count++;
            } else {
                count--;
            }
            if (count < 0) {       //证明太多')'
                return false;
            }
        }
        return count == 0;			//看有没多余的'('
    }
	
	
	
	/** 301. Remove Invalid Parentheses
	 * 删掉invalid的，但是删的个数是minimum
	 * ["(a)())()" -> ["(a)()()", "(a())()"]
	 * @param s
	 * @return
	 * 基本思想：找到invalid的，for循环在前面不同位置remove多余的，然后再dfs剩下的..
	 * 
	 * 1. 怎么找invalid？用count记录，left的话++，right的话--。如果count < 0那就是invalid
	 * 
	 * 2. invalid的地方在 i，那么从上一次valid后的地方开始扫，也就是 j从[validPos, i]扫
	 * 		2.1 如果多出的')'. 那么remove时要注意，如果连续的")))",那只能删一个，不能重复
	 * 			所以可以删第一个，后面都跳过..
	 * 		2.2 删完后，dfs(newStr, i, j).. 记得更新i, j位置
	 * 
	 * 3. 加result
	 * 
	 * 4. 注意，除了从左->右扫，还需要 从右往前扫，跟Longest Valid Parentheses的two pass一样
	 *  为了避免重复代码，可以替换一下：
	 *   a. 用char[] 表示'(', ')'.. 从右往左时，调换顺序 
	 *   b. 从right扫时，要reverse str，这样才能也是从前往后
	 */
	public List<String> removeInvalidParentheses(String s) {
        List<String> result = new ArrayList<>();
        remove(result, s, 0,0, new char[]{'(', ')'});
        return result;
    }
    
    public void remove(List<String> result, String s, int pos, int validPos, char[] par) {
        int count = 0;
        
        for (int i = pos; i < s.length(); i++) {
            if (s.charAt(i) == par[0])         count++;
            else if (s.charAt(i) == par[1])    count--;        //也可以不用else.都一样
            
            // check if has invalid
            if (count < 0) {
                for (int j = validPos; j <= i; j++) {
                    if (s.charAt(j) == par[1]) {		//避免重复结果
                        if (j == validPos || s.charAt(j - 1) != par[1]) {
                            String newStr = new StringBuilder(s).deleteCharAt(j).toString();
                            remove(result, newStr, i, j, par);    //j删掉了，所以validPos更新..之前的都valid
                        }
                    }
                }
                return;
            }
        }
        
        String reverseStr = new StringBuilder(s).reverse().toString();
        if (par[0] == '(') {        //left to right结束了
            remove(result, reverseStr, 0, 0, new char[]{')', '('});  //左右括号要反过来
        } else {            //right to left结束. 这里把之前reversed的转回来变成正常顺序
            result.add(reverseStr);
        }
        
    }

	

    /** 71. Simplify Path 
     * path = "/home/", => "/home"；  path = "/a/./b/../../c/", => "/c" 因为. 是当前dir,  ..是parent
     * path = "/../" => "/";  多个/ 的话"/home//foo/" => "/home/foo".
     * 用Deque可以两头都poll
     * 其实也可以用arraylist, 这样pop时是list.remove(list.size() - 1); 但是比较慢
     */
    public String simplifyPath(String path) {
    	String[] paths = path.split("/");
        Deque<String> deque = new LinkedList<>();   //use deque can get from both sides
        StringBuilder sb = new StringBuilder();
        
        for (String s : paths) {
            if (s.equals("") || s.equals(".")) {	//记得跳过""
                continue;
            }
            if (s.equals("..")) {
                if (!deque.isEmpty()) {
                    deque.pollLast();
                }
            } else {
                deque.offerLast(s);
            }
        }
        
        if (deque.isEmpty())
            return "/";
            
        while (!deque.isEmpty()) {
            sb.append("/").append(deque.pollFirst());
        }
        return sb.toString();
    }
    
    
    
    /** 388. Longest Absolute File Path - 单调递增Stack
     * 给"dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2
     * \n回车 \t是tab，找到最长的文件path  dir\subdir1\file1.ext
     * @param input
     * @return
     * 跟下面那rectangle很像，也是用stack 维护increasing的目前为止的长度
     * 如果当前level < stack.size，就要弹出来，找parent
     * 
     * stack要存目前为止的length，而不是string..否则算长度时需要每次重新算很麻烦..
     */
    public int lengthLongestPath(String input) {
        Stack<Integer> stack = new Stack<>();		//记录目前为止的长度
        stack.push(0);      //dummy, 防止算第一个根目录时s.peek出错
        int maxLen = 0;
        
        for (String s : input.split("\n")) {		//方便切分文件
            int level = s.lastIndexOf("\t") + 1;        //start from 0,need +1。注意\t 代表一个字符 not2

            while (level + 1 < stack.size()) {      //has dummy 0,  so need + 1
                stack.pop();        //find parent
            }

            int len = stack.peek() + s.length() - level + 1;    //+1 to append '\'., 记得去掉\t
            stack.push(len);

            if (s.contains(".")) {                  //is file
                maxLen = Math.max(maxLen, len - 1);     //remove last '\'
            }
        }
        
     // ==================================================================
    	// 用int[] 比stack更快！！！
        String[] paths = input.split("\n");
        int[] arrStack = new int[paths.length + 1];
        for (String s : paths) {
            int level = s.lastIndexOf("\t") + 1; 
            
       // 相当于stack..  arrs[lev]相当于stack.peek(). 而且这里不用pop.直接更新arrs[lev+1]即可（覆盖之前的数）
            arrStack[level + 1] = arrStack[level] + s.length() - level + 1;
            len = arrStack[level+1];
            
            if (s.contains("."))
                maxLen = Math.max(maxLen, len - 1);     //remove last '\'
        }
        
        return maxLen;
    }


    /**
     * 496. Next Greater Element I
     * nums1是nums2的subset, 在nums1里的数，要找到对应的在nums2里的next greater element。没的话就-1
     * Input: nums1 = [4,1,2], nums2 = [1,3,4,2].
     * Output: [-1,3,-1]
     *
     * input [4,1,2], [2,3,4,1]
     * output [-1,-1,3]
     *
     * Naive..
     * 1. for()用hashmap存 nums2 里 num对应的index..
     * 2. for() 找map的index, 然后再for找到next greater
     *
     * O(m * n)
     *
     * 后面解法更好
     */
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        int len1 = nums1.length;
        int[] result = new int[len1];

        Map<Integer, Integer> map = new HashMap<>();    // <num, idx>
        for (int i = 0; i < nums2.length; i++) {
            map.put(nums2[i], i);
        }

        for (int i = 0; i < nums1.length; i++) {
            result[i] = -1;
            // 因为 nums1是nums2的子集，map里肯定contains
            for (int idx = map.get(nums1[i]) + 1; idx < nums2.length; idx++) {
                if (nums2[idx] > nums1[i]) {
                    result[i] = nums2[idx];
                    break;
                }
            }
        }

        return result;
    }

    /**
     * 496. Next Greater Element I
     *
     * 主要问题是，先想想  如何在一个array里找到 next greater element..并且优化复杂度
     *
     * 用单调递减 stack..
     *
     * 因为是 单调递减，只要 栈顶 < cur num 那么 当前num就是pop的next greater element..
     * 而且要pop出小的，放当前大的
     */
    public int[] nextGreaterElementBetter(int[] nums1, int[] nums2) {
        int len1 = nums1.length;
        int[] result = new int[len1];

        Map<Integer, Integer> map = new HashMap<>();    // <num, next greater>
        Stack<Integer> stack = new Stack<>();

        // 这题先想，如何在一个array里找到 next greater element..并且优化复杂度
        for (int num : nums2) {     // 大的集合
            while (!stack.isEmpty() && stack.peek() < num) {
                map.put(stack.pop(), num);      // 因为stack是单调递减，s.pop的next greater就是cur
            }
            stack.push(num);        // stack是单调递减
        }

        for (int i = 0; i < nums1.length; i++) {
            result[i] = map.getOrDefault(nums1[i], -1);
        }
        return result;
    }


    /**
     * 503. Next Greater Element II
     * 找数组里的next greater element.. 是circular array
     * @param nums
     * @return
     *
     * 也是跟上面一样，用单调递减stack.. 不过这次这里存的是index，这样result[] 才知道哪个index
     */
    public int[] nextGreaterElementsII(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];

        Arrays.fill(result, -1);

        Stack<Integer> stack = new Stack<>();       // 单调递减 存index

        for (int i = 0; i < n * 2; i++) {       // < n * 2
            int num = nums[i % n];
            while (!stack.isEmpty() && nums[stack.peek()] < num) {
                result[stack.pop()] = num;
            }

            if (i < n) {
                stack.push(i);
            }
        }
        return result;
    }
    
    
    /**
     * 84. Largest Rectangle in Histogram  - 单调递增stack
     * 给[2,1,5,6,2,3]，最大连续面积是5*2 = 10
     * @param height
     * @return
     *
     * 以当前i作为右边界，看左边那些的面积能到多大
     * 用stack存index，每次push进index。 stack保持单调递增  前面 < 后面，这样popLast才能确保中间这部分valid
     *
     * while (cur<=pre) 就开始pop 并算每次pop的高 * 宽的面积。---- pop直到stack里剩下的 < cur 停止。再push
     * 
     * 记得算w时 空就i, 否则是i - stack.peek() - 1
     * 		!!!!要peek而不是用之前的pop.因为stack里第一个和被pop出来的很可能隔着几个数
     */
    public int largestRectangleArea(int[] height) {
    	if (height == null || height.length == 0) {
    		return 0;
    	}
    	
    	// O(n)  每个数push和pop一次，都是O(1), 所以n个数加起来就是O(n)
        // 单调递增. 前面 < 后面，这样popLast才能确保中间这部分valid
    	Stack<Integer> stack = new Stack<Integer>();
    	int max = 0;

    	for (int i = 0; i <= height.length; i++) {
    		                // cur的高度。 直到最后超出数组范围=len. 才=-1，这样就能再进while里算面积
    		int cur = i < height.length ? height[i] : -1;
    		                // height[s.peek()]是previous高度. 当 cur <= pre时(<也可以)，就开始pop. 一个个算面积
    		while (!stack.isEmpty() && cur < height[stack.peek()]) {
    			int h = height[stack.pop()];	
    			int w = stack.isEmpty() ? i : i - stack.peek() - 1;		//已经pop一个，所以比原来的再-1
                            // empty说明是第一个，或者 cur i是[0,i]的min,所以长度是i也对

    			max = Math.max(max, h * w);		// 已pop的h为smallest高来算面积
    		}
    		stack.push(i);			//每次把index push进去.
    	}
    	
    	// ==================================================================
    	// 用int[] 比stack更快！！！
    	int n = height.length;
        int[] stackArr = new int[n + 1];
        int is = -1;
        for (int i = 0; i <= n; i++) {
            int curHeight = (i == n) ? 0 : height[i];
            while (is != -1 && curHeight < height[stackArr[is]]) {
                int hh = height[stackArr[is--]];
                int width = (is == -1) ? i : i - 1 - stackArr[is];
                max = Math.max(max, hh * width);
            }
            stackArr[++is] = i;
        }
    	
    	return max;
    }


    /**
     * 84. Largest Rectangle in Histogram  - 单调递增stack
     *
     * DP解法  https://discuss.leetcode.com/topic/39151/5ms-o-n-java-solution-explained-beats-96
     *
     * 核心思想：每个 cur height 乘以 宽 的面积，那么宽是 lessRight[i] - lessLeft[i] - 1 就是左右两边离自己最近的 closest smaller值
     *
     * index      0  1 2 3 4 5
     * heights    2  1 5 6 2 3
     * lessRight  1  6 4 4 6 6
     * lessLeft  -1 -1 1 2 1 4
     * width      1  6 2 1 4 3  (lessRight - lessLeft - 1)
     * area       2 6 10 6 8 3
     *
     * 那么怎么分别算 lessLeft 和 lessRight呢？
     * 通常要O(n^2) 循环2次来找，但是可以用之前存下来的结果，找得更快.. 其实就是DP
     *
     * naive 找 lessLeft
     * for (int i = 1; i < height.length; i++) {
     *     int p = i - 1;
     *     while (p >= 0 && height[p] >= height[i]) {
     *         p--;
     *     }
     *     lessFromLeft[i] = p;
     * }
     *
     * 有了DP存 lessLeft[]以后
     * while (p >= 0 && height[p] >= height[i]) {
     *       p = lessFromLeft[p];
     * }
     */
    public int largestRectangleArea2(int[] heights) {
        if (heights == null || heights.length == 0)
            return 0;

        int max = 0;
        int len = heights.length;
        int[] lessLeft = new int[len];      // closest left idx that less than cur height
        int[] lessRight = new int[len];     // closest right idx that less than cur height
        lessLeft[0] = -1;
        lessRight[len - 1] = len;

        // 找出比cur height 低的left idx   这里有点DP 的感觉，因为找的时候基于上次的结果
        for (int i = 1; i < len; i++) {
            int pre = i - 1;
            // 之前前面的已经算过它们之前的less left idx, 用之前算好的值即可
            while (pre >= 0 && heights[pre] >= heights[i]) {
                pre = lessLeft[pre];
            }

            lessLeft[i] = pre;
        }

        // 找出比cur height 低的right idx
        for (int i = len - 2; i >= 0; i--) {
            int next = i + 1;

            while (next < len && heights[next] >= heights[i]) {
                next = lessRight[next];
            }

            lessRight[i] = next;
        }

        // 最后 其实 width = lessRight[i] - lessLeft[i] - 1 只算中间 >= cur height的区域
        for (int i = 0; i < len; i++) {
            max = Math.max(max, heights[i] * (lessRight[i] - lessLeft[i] - 1));
        }

        return max;
    }
	
    
    /**
     * 85. Maximal Rectangle
     * 这道题的二维矩阵每一层向上都可以看做一个直方图，输入矩阵有多少行，就可以形成多少个直方图.
     * 对每个直方图都调用 Largest Rectangle in Histogram 中的方法，就可以得到最大的矩形面积。
     *
     * 主函数每行加上前一行累计的竖的面积。
     *
     * @param matrix
     * @return
     * m*n矩阵，每行O(n)复杂度，整个就是O(m*n)
     */
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)  
            return 0;
        int rows = matrix.length;
        int cols = matrix[0].length;
        int max = 0;
        int[] h = new int[cols + 1];		//声明每行的int[]。输入是char[][] 麻烦
        
        for (int i = 0; i < rows; i++) {
        	Stack<Integer> s = new Stack<>();
            for (int j = 0; j <= cols; j++) {		// 这样最后才能算最后
            	if (j < cols) {
            		h[j] = (matrix[i][j] == '1') ? h[j] + 1 : 0;  // 在原先arr[j]上+1就行
            	}

                // 算 以这一层向上 的直方图 max 矩形面积
            	while (!s.isEmpty() && h[s.peek()] >= h[j]) {
                    int height = h[s.pop()];
                    int wid = s.isEmpty() ? j : j - s.peek() - 1;
                    max = Math.max(max, height * wid);
                }
                s.push(j);
            }
        }
        return max;
    }


    /**
     * 85. Maximal Rectangle
     * 跟上一题的解法2类似，求left, right边界
     *
     * https://leetcode.com/problems/maximal-rectangle/discuss/29054/Share-my-DP-solution
     */
    public int maximalRectangle2(char[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return 0;

        int rows = matrix.length;
        int cols = matrix[0].length;
        int max = 0;
        int[] h = new int[cols];

        int[] left = new int[cols];      // closest left idx that less than cur height
        int[] right = new int[cols];     // closest right idx that less than cur height

        Arrays.fill(right, cols - 1);       // 记得初始化

        for (int i = 0; i < rows; i++) {
            int leftBound = 0;
            int rightBound = cols - 1;

            // 求当前height
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == '1') {
                    h[j]++;
                } else {
                    h[j] = 0;
                }
            }

            // 找出 都是1 的 left bound
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == '1') {
                    left[j] = Math.max(left[j], leftBound);
                } else {
                    left[j] = 0;
                    leftBound = j + 1;
                }
            }

            // 找出 都是1 的 right bound
            for (int j = cols - 1; j >= 0; j--) {
                if (matrix[i][j] == '1') {
                    right[j] = Math.min(right[j], rightBound);
                } else {
                    right[j] = cols - 1;
                    rightBound = j - 1;
                }
            }

            // 最后 其实 width = Right[i] - Left[i] + 1 只算中间 >= cur height的区域
            for (int j = 0; j < cols; j++) {
                max = Math.max(max, h[j] * (right[j] - left[j] + 1));
            }

            /*
            // 或者合起来变成2个for loop
            for (int j = cols - 1; j >= 0; j--) {
                if (matrix[i][j] == '1') {
                    right[j] = Math.min(right[j], rightBound);
                } else {
                    right[j] = cols - 1;
                    rightBound = j - 1;
                }
            }

            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == '1') {
                    h[j]++;
                    left[j] = Math.max(left[j], leftBound);
                    max = Math.max(max, h[j] * (right[j] - left[j] + 1));
                } else {
                    h[j] = 0;
                    left[j] = 0;
                    leftBound = j + 1;
                }
            }
            */
        }

        return max;
    }
    
    
    /**
     * 42. Trapping Rain Water - 比较复杂
     * Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6. 能灌多少水。每个bar为1的宽
     * @param height
     * @return
     * 类似上面的直方图。也是用stack存，但是decreasing.
     * 每次pop时代表H[i]大，所以找min(H[peek(), H[i]]来确定哪个low可以做boundary
     * 选出来bound后，要算出高，需要 bound-cur (cur是pop出来的高)，横向一层层叠加*w 算
     * 
     * 
     * 有个Two pointer更简单，看TwoPointer.java
     * 维护左右两边的高，挑low点的一格格往前走，并判断是否fill水。
     */
    public int trap(int[] height) {
        if (height == null || height.length <= 1)   return 0;
        
        int len = height.length;
        Stack<Integer> s = new Stack<>();
        int res = 0;
        int area = 0;
        int i = 0;
        
        while (i < len) {       // stack should be decreasing
            if (s.isEmpty() || height[s.peek()] >= height[i]) {
                s.push(i++);
            } else {
                int cur = height[s.pop()];
                if (s.isEmpty()) {        //empty means no left boundary
                    area = 0;
                } else {
                    int minBound = Math.min(height[s.peek()], height[i]);   //choose bound from peek or i (i>cur)
                    area = (minBound - cur) * (i - s.peek() - 1); 
                }
                res += area;
            }
        }
        return res;
    }
    
    
    
    /** 407. Trapping Rain Water II - BFS + minHeap
     * 给m * n的2D矩阵，看能装多少水
     * @param heights
     * @return
     * 先从四周外墙开始，每次先找min来BFS..看中间的邻居要不要fill water
     * 
     * 1. 建个Cell类存x, y, height
     * 2. 把外围墙cell加到min heap里，并记visited。heap其实放的是外围墙
     * 3. BFS循环 
     *  3.1 minQ.poll找出最小的cell
     *  3.2 找四周没遍历过的邻居 (若visit过就跳过，基本是外墙）
     *  3.3 设为visited，看邻居是否高于cur，还是加水变成cur的高度，并把max加到heap里（因为是外墙）
     *  		如果邻居fill水变成当前cell的高，那放进minHeap里要是更新过的高，也就是cell.h
     *  
     *  因为q是从外墙开始，如果邻居比现在的cell(外墙)高，那也没办法因为会流出去，放不了水..
     *  
     * !!!cell是visited的外墙里最小的，所以到时没访问的邻居若小于cell，可以直接fill到cell.h就行
     * 
     */
    public int trapRainWater(int[][] heights) {
        if (heights == null || heights.length == 0 || heights[0].length == 0)
            return 0;
        
        PriorityQueue<Cell> q = new PriorityQueue<Cell>(1, new Comparator<Cell>() {
            public int compare (Cell a, Cell b) {
                return a.h - b.h;
            }
        });
        
        int m = heights.length;
        int n = heights[0].length;
        boolean[][] visited = new boolean[m][n];
        // 要用visited[][], 不能再Cell类里加visited值，否则while里找4周时判断不了是否visited，因为没建Cell
        
      //left - right 建外围墙
        for (int i = 0; i < n; i++) {
            q.offer(new Cell(0, i, heights[0][i]));
            q.offer(new Cell(m-1, i, heights[m-1][i]));
            visited[0][i] = true;
            visited[m-1][i] = true;
        }
        
        //top - bottom 建外围墙
        for (int j = 0; j < m; j++) {
            q.offer(new Cell(j, 0, heights[j][0]));
            q.offer(new Cell(j, n-1, heights[j][n-1]));
            visited[j][0] = true;
            visited[j][n-1] = true;
        }
        
        int[] d = {0, 1, 0, -1, 0};
        int water = 0;
        
        // BFS, 每次Poll出来最low的墙
        while (!q.isEmpty()) {
            Cell cell = q.poll();   //cell作为visited的外墙里最小的，所以到时没访问的邻居若小于cell，可以直接fill到cell.h
            for (int i = 0; i < 4; i++) {
                int x = cell.x + d[i];
                int y = cell.y + d[i + 1];
                if (x >= 0 && x < m && y >= 0 && y < n && !visited[x][y]) {     //not visited
                    visited[x][y] = true;
                    water += Math.max(0, cell.h - heights[x][y]);	//若高于邻居就fill水
                    //put the new wall(higher) 更新外墙. 如果邻居矮的话要fill水，且fill到跟cell一样高。否则直接更新最高的
                    q.offer(new Cell(x, y, Math.max(cell.h, heights[x][y])));   
                    /*
                    if (cell.h > heights[x][y]) {			//这样比较清楚
                        water += cell.h - heights[x][y];
                        minQ.offer(new Cell(x, y, cell.h));     //加水到cell.h
                    } else {
                        minQ.offer(new Cell(x, y, heights[x][y]));
                    }
                    */
                }
            }
            
        }
        return water;
    }
    
    public class Cell {
        int x;
        int y;
        int h;
        
        public Cell(int row, int col, int height) {
            x = row;
            y = col;
            h = height;
        }
    }
    
    
    
    /** 417. Pacific Atlantic Water Flow
     * 左上角是Pacific，右下角是Atlantic。中间矩阵的cell代表高度，求 水能流到2个海洋的 pair数
     * 高的可以流到 相同或矮的那
     *
     * 跟上面的trapping rain water II类似
     * 要从外墙/ 海洋边缘开始搜。 否则一个个搜十字列才费劲了
     * 1. 分别处理2个海洋
     * 2. 从边缘搜起，如果neighbor比自己高，那就能流到自己，就标为TRUE
     * 3. 扫多一次，看是否2个boolean[][]在某点都为TRUE，就证明都能流
     * 
     * DFS跟BFS差不多，但是BFS要用queue，而且比dfs慢
     */
    public List<int[]> pacificAtlantic(int[][] matrix) {
        List<int[]> list = new ArrayList<>();
        if (matrix == null || matrix.length == 0)   return list;
        
        int m = matrix.length;
        int n = matrix[0].length;
        boolean[][] pacific = new boolean[m][n];    //if can flow to pacific, mark as visited
        boolean[][] atlantic = new boolean[m][n];
        Queue<int[]> pq = new LinkedList<>();
        Queue<int[]> aq = new LinkedList<>();
        
        for (int i = 0; i < m; i++) {   //vertical border
//            pq.add(new int[]{i, 0});	
//            aq.add(new int[]{i, n-1});		//bfs
            pacific[i][0] = true;
            atlantic[i][n-1] = true;
            
            dfs(matrix, pacific, i, 0);
            dfs(matrix, atlantic, i, n - 1);
        }
        
        for (int j = 0; j < n; j++) {   //horizonal border
//            pq.add(new int[]{0, j});
//            aq.add(new int[]{m-1, j});
            pacific[0][j] = true;
            atlantic[m-1][j] = true;
            
            dfs(matrix, pacific, 0, j);
            dfs(matrix, atlantic, m - 1, j);
        }
        
//        bfs(matrix, pq, pacific);
//        bfs(matrix, aq, atlantic);
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if (pacific[i][j] && atlantic[i][j]) {
                    list.add(new int[] {i, j});
                }
            }
        }
        return list;
    }
    
    int[] d = new int[] {0, -1, 0, 1, 0};
    
    public void dfs(int[][] matrix, boolean[][] visited, int i, int j) {
        int m = matrix.length;
        int n = matrix[0].length;
        visited[i][j] = true;
        for (int k = 0; k < 4; k++) {
            int x = i + d[k];
            int y = j + d[k+1];
            if (x >= 0 && x < m && y >= 0 && y < n && !visited[x][y] && matrix[x][y] >= matrix[i][j]) { 
                dfs(matrix, visited, x, y);
            }
        }
    }
    
    public void bfs(int[][] matrix, Queue<int[]> q, boolean[][] visited){
        int m = matrix.length;
        int n = matrix[0].length;
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int curi = cur[0];
            int curj = cur[1];
            for (int k = 0; k < 4; k++) {
                int x = curi + d[k];
                int y = curj + d[k+1];
                if (x >= 0 && x < m && y >= 0 && y < n && !visited[x][y] && matrix[x][y] >= matrix[curi][curj]) { 
                    q.add(new int[]{x, y});
                    visited[x][y] = true;
                }
            }
        }
    }
    
    
    /** 215. Kth Largest Element in an Unsorted Array
     * Given [3,2,1,5,6,4] and k = 2, return 5.
     * 另一种更快的解法是quick select，在Array那
     * @param nums
     * @param k
     * @return
     * 用minHeap，size为K。把前K个放进去。
     * 之后(k~n)个数跟heap里的root比，大于就放进去。这样得到K里面的最小值。
     * (不是用maxHeap, 除非放n个数进去。浪费！)
     * 时间 O(nlogK) + O(k)空间(heap)
     */
    public int findKthLargestMinHeap(int[] nums, int k) {
        Queue<Integer> minHeap = new PriorityQueue<>(k, Collections.reverseOrder());
        
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
    
    

    private int maxSize;
    private Queue<Integer> minHeap;
    
    public StackQueue (int k) {
    	maxSize = k;
    	minHeap = new PriorityQueue<>();
    }
    
    public void add(int num) {
    	if (minHeap.size() < maxSize) {
    		minHeap.add(num);
    		return;
    	} else {
    		minHeap.poll();
    		minHeap.add(num);
    	}
    }
    
    // return K largest number . k个数 从大到小
    public List<Integer> topK() {
    	List<Integer> list = new ArrayList<>();
    	Iterator itr = minHeap.iterator();
    	while(itr.hasNext()) {
    		list.add((Integer) itr.next());	//记得cast成Integer
    	}
    	Collections.sort(list, Collections.reverseOrder());		//从大到小
    	return list;
    }
    
    
    
    
    /** 155. Min Stack - easy
     * 主要能用O(1)时间得到所有操作，包括min..
     *
     * 法1
     * 最简单的就是另外定义一个Element类，能存val和min..
     *
     * 法2
     * 或者用一个stack，存val以及再push min
     * 在push时，如果x<=min, 就先push(min)..之后再push(x)
     * pop时.. if(stack.pop() == min) min=stack.pop();
     *
     * 法3
     * 或者2个stack，一个存val，一个存min  pair<val, 1st idx(s.size())>. 更新min时才需要push进去 + 第一次的idx
     */
    public void MinStack() {
    	minStack = new Stack<>();
    }
    
    Stack<Element> minStack;
    
    public void push(int x) {
        int min = minStack.isEmpty() ? x : Math.min(minStack.peek().min, x);
        minStack.push(new Element(x, min));

    }
    
    public void pop() {
    	minStack.pop();
    }
    
    public int top() {
        return minStack.peek().val;
    }
    
    public int getMin() {
        return minStack.peek().min;
    }
    
    class Element {
        int val;
        int min;
        
        public Element(int val, int min) {
            this.val = val;
            this.min = min;
        }
    }

    /** 155. Min Stack - easy
     * 主要能用O(1)时间得到所有操作，包括min..
     *
     * 用2个stack, 其中的minstack只放min.. pair<val, 1st idx(s.size())>
     * 这种是，如果stack里有很多duplicate数，那minStack只用存一次，存min值，和第一次出现的idx, 其实就是按照stack.size()来
     *
     * 还有另外一种，用Node存，里面有现在的Node top;  int min, Node previous.. 这样不用stack..
     */

    Stack<Integer> stack1;
    Stack<MinElement> minS;

    public void push11(int x) {
        stack.push(x);
        if (minS.isEmpty() || minS.peek().val > x) {
            minS.push(new MinElement(x, stack.size()));     // 记录这个新min第一次出现的index
        }
    }

    public void pop11() {
        if (!minS.isEmpty() && minS.peek().idx == stack.size()) {
            minS.pop();
        }
        stack.pop();
    }

    public int top11() {
        return stack.peek();
    }

    public int getMin11() {
        return minS.peek().val;
    }

    class MinElement {
        int val;
        int idx;        // first idx when min pushed to stack, value is stack's size

        public MinElement(int val, int idx) {
            this.val = val;
            this.idx = idx;
        }
    }


    /**
     * 716. Max Stack
     * 能直到max的stack
     *
     * 用2个stack，一个正常num，一个放max
     * 其他都是O(1)，但popMax是O(n)
     *
     * 后面有更好的方法
     */
    public void MaxStack() {
        stack = new Stack();
        maxStack = new Stack();
    }

    Stack<Integer> stack;
    Stack<Integer> maxStack;


    public void push3(int x) {
        int max = maxStack.isEmpty() ? x : maxStack.peek();
        maxStack.push(max > x ? max : x);
        stack.push(x);
    }

    public int pop3() {
        maxStack.pop();
        return stack.pop();
    }

    public int top3() {
        return stack.peek();
    }

    public int peekMax3() {
        return maxStack.peek();
    }

    // 这里都是上面定义好的.. 2个stack一起行动
    public int popMax3() {
        int max = peekMax3();
        Stack<Integer> buffer = new Stack();
        while (top3() != max) {
            buffer.push(pop3());     // 这个pop是stack和maxStack一起pop
        }
        pop3();
        while (!buffer.isEmpty()) {
            push3(buffer.pop());
        }
        return max;
    }


    /**
     * 716. Max Stack
     *
     * 用Double LinkedList (正常node 头尾) + TreeMap(val, List<Node>)
     * 用treemap可以快速找到 max.. 这样根据max找到对应的node, 然后删掉node, unlink node.
     *
     * 除了peek()是O(1), 其他都是O(logN)
     */
    public void MaxStackBST() {
        head = new ListNode(-1);
        tail = new ListNode(-1);
        head.next = tail;
        tail.prev = head;

        bst = new TreeMap<>();
    }

    ListNode head;
    ListNode tail;
    TreeMap<Integer, List<ListNode>> bst;

    public void push4(int x) {
        ListNode node = new ListNode(x);
        node.next = tail;
        node.prev = tail.prev;
        tail.prev = node;
        node.prev.next = node;

        if (!bst.containsKey(x)) {
            bst.put(x, new ArrayList<>());
        }
        bst.get(x).add(node);
    }

    public int pop4() {
        ListNode last = tail.prev;
        removeNode(last);

        List<ListNode> list = bst.get(last.val);
        list.remove(list.size() - 1);               // 若是LinkedList直接 bst.get(last.val).removeLast()
        if (list.isEmpty()) {                       // 按顺序放进来，最后一个肯定是last
            bst.remove(last.val);
        }

        return last.val;
    }

    public int top4() {
        return tail.prev.val;
    }

    public int peekMax4() {
        return bst.lastKey();
    }

    public int popMax4() {
        int max = peekMax4();

        List<ListNode> list = bst.get(max);
        ListNode node = list.remove(list.size() - 1);          // 若是LinkedList直接 bst.get(max).removeLast()

        removeNode(node);

        if (list.isEmpty()) {
            bst.remove(max);
        }

        return max;
    }

    private void removeNode(ListNode node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }


    class ListNode {
        int val;
        ListNode prev;
        ListNode next;

        public ListNode(int v) {
            val = v;
        }
    }
    
    
    /** 232. Implement Queue using Stacks
     * 需要2个stack来放reverse的..
     * 
     * 方法1：push-O(n) 每次push都把stack里的放到tmp里，然后再放回来
     * 
     * 方法2：push不变，但是stack.isEmpty时front = x. 然后正常push O(1)
     *   关键是pop.. 如果备用tmp为空，那就把stack都放tmp上..
     *   	那么tmp都是reverse的.. 之后一直用正常tmp.pop就行。。直到tmp空了再放
     *   pop是amortize O(1)
     */
    public void MyQueue() {
        stack = new Stack<>();
        tmp = new Stack<>();
    }

    Stack<Integer> tmp;
    int front;
    
    
    /** Push element x to the back of queue. *
    public void push(int x) {
        while (!stack.isEmpty()) {
            tmp.push(stack.pop());
        }
        stack.push(x);
        while (!tmp.isEmpty()) {
            stack.push(tmp.pop());
        }
    }
    */
    public void push1(int x) {
        if (stack.isEmpty())    front = x;
        
        stack.push(x);
    }
    
    
    /** 之前push一直放，直到遇到pop
     * 如果tmp为空，那就都把stack放到tmp里，这样就能tmp.pop
     * 直到tmp空了，再放进去
     */
    public int pop1() {
  //      return stack.pop();
        if (tmp.isEmpty()) {
            while (!stack.isEmpty()) {
                tmp.push(stack.pop());
            }
        }
        return tmp.pop();
    }
    
    public int peek1() {
 //       return stack.peek();
        if (!tmp.isEmpty()) {
            return tmp.peek();
        }
        return front;
    }
    
    public boolean empty1() {
        return stack.isEmpty() && tmp.isEmpty();
    }
    
    
    
    /** 225. Implement Stack using Queues
     * 用一个q就行..
     * 关键是push时，要留住新来的x, 然后x之前的全Poll出来放到后面 O(n)
     * 
     * 其他方法都一样
     */
    public void MyStack() {
        Queue<Integer> q = new LinkedList<>();
    }
    Queue<Integer> q;
    
    /** Push element x onto stack. */
    public void push2(int x) {
        q.add(x);
        int size = q.size();
        while (size-- > 1) {
            q.add(q.poll());        //前面的弹出来放到后面..
        }
    }
    
    public int pop2() {
        return q.poll();
    }
    
    public int top2() {
        return q.peek();
    }
    
    
    
	
	/**
     * 295. Find Median from Data Stream
	 * 给串unsorted的数，一个个stream过来，求median。若是偶数就 /2
	 * @param num
	 * 维护一大一小的priorityQueue
	 * 默认larger的heap会多点数。如果和small个数一样，那就/2
	 * 1. assume large会存着median，所以每次add数先加到large里，相当于跟里面的median比
	 * 2. 找large里的min放小的里 （因为要保持 大-小<=1, 所以如果本身large多了一个数，就要放到小里平衡
	 * 3. 要保持size的平衡，所以要check是否再放回large
	 * 
	 * !!!!为何先放large后，再small.add(large.poll()) ?
	 * 因为num不一定是中位数，只能放在heap里面，然后heap再弹出来.. 
	 * 而且这样不需要check是否空
	 * 
	 *如果是偶数想返回较小的做median的话，就让small大一点，最后不论奇偶都返回small.peek()即可
	 */
	public void addNum(int num) {
		large.add(num);             //put in large
		small.add(large.poll());    //add small with min in large
		if (large.size() < small.size()) {
			large.add(small.poll());    //kepp large more nums
		}
	}
	
	PriorityQueue<Integer> large = new PriorityQueue<>();
    PriorityQueue<Integer> small = new PriorityQueue<>(Collections.reverseOrder());
            
	// Returns the median of current data stream
	public double findMedian() {
		return large.size() > small.size() ? large.peek()
		                            : (large.peek() + small.peek()) / 2.0;     //same size
	}
    
	
	
	/**
     * 480. Sliding Window Median
	 * Window position                Median
		---------------               -----
		[1  3  -1] -3  5  3  6  7       1
		 1 [3  -1  -3] 5  3  6  7       -1
		 1  3 [-1  -3  5] 3  6  7       -1
		 1  3  -1 [-3  5  3] 6  7       3
		 1  3  -1  -3 [5  3  6] 7       5
		 1  3  -1  -3  5 [3  6  7]      6
	 * 返回median sliding window as [1,-1,-1,3,5,6]
	 * @param nums
	 * @param k
	 * @return
	 * 比较简单的做法，跟上面一样，维护2个heap.. 加heap也是一样，反正large >= small 最多多一个而已
	 * 
	 * 记得 for i <= len, 因为每次结束后是上一次的最后结果，还没加进去.最后i==len把最后的len-1结果放进去
	 * 
	 * 这题可以用HashHeap来代替普通的heap，主要为了把删除remove的复杂度从O(n)降到O(logn).
	 * 因为有hashheap，就可以用hashmap一下找到要删除的点O(1)
	 */
	public double[] medianSlidingWindow(int[] nums, int k) {
        double[] med = new double[nums.length - k + 1];
        
        PriorityQueue<Integer> large = new PriorityQueue<>();
        PriorityQueue<Integer> small = new PriorityQueue<>(Collections.reverseOrder());
               
        // 刚开始直接加到heap里。直到==k时才算med的结果 & remove最开始的数nums[i-k]
        for (int i = 0; i <= nums.length; i++) {				// 记得i == len, 把最后结果放进去
            if (large.size() + small.size() == k) {
                // 加结果...add median										    分开来/2防止相加越界
                med[i-k] = large.size() > small.size() ? (double) large.peek() : large.peek() / 2.0 + small.peek() / 2.0;

                if (!large.remove(nums[i-k])) {
                    small.remove(nums[i-k]);
                }
            }
            
            if (i == nums.length)   break;		//防止最后越界
            
            // 加进来一个数
            large.add(nums[i]);
            small.add(large.poll());
            if (large.size() < small.size()) {
                large.add(small.poll());
            }
            
        }
        return med;
    }
	
	
	/** 239. Sliding Window Maximum
	 * 跟上面很像，只不过找MAX而已
	 * 给k=3的话，那window大小就是3.每次往后挪一位，看每次window的max是多少，最后都放在int[]里
	 * Given nums = [1,3,-1,-3,5,3,6,7], and k = 3。eturn the max sliding window as [3,3,5,5,6,7].
	 * @param nums
	 * @param k
	 * @return
	 * 很简单的方法用大小为k的maxHeap。但是remove操作要O(n)，即使用HashHeap也要O(logn). 所以总的慢
	 */
	public int[] maxSlidingWindowPQ(int[] nums, int k) {
		if (nums.length == 0 || k <= 0)     return new int[0];
		
        int[] res = new int[nums.length - k + 1];
        PriorityQueue<Integer> heap = new PriorityQueue<>(k, Collections.reverseOrder());
        
        for (int i = 0; i <= len; i++) {
            if (heap.size() == k) {
                res[i - k] = heap.peek();
                heap.remove(nums[i - k]);
            }
            if (i == len)   continue;
            
            heap.add(nums[i]);
        }
        
        /*  另一种写法  还是上面的好.. 跟上一题一致
        for (int j = 0; j < k; j++) {
            heap.add(nums[j]);
        }
        
        for (int i = 0; i < nums.length - k + 1; i++) {
            if (i != 0) {
                heap.add(nums[i+k-1]);		//加last
            }
            res[i] = heap.peek();
            heap.remove(nums[i]);		//去掉first
        }
        */
        
        return res;
	}
	
	
	/** 239. Sliding Window Maximum - better O(n)
	 * @param nums
	 * @param k
	 * @return
	 * 用单调递减decreasing双向队列Deque来存可能的max的index..
	 * 这里用deque的话，不需要是大小为k.. 因为不是heap了.. 而且要维护递减序列，里面的数是pollFirst最大的
	 * 这个deque的大小不确定，会随时变.. 只有当超出sliding window范围，才pollFirst. 否则都是peekFirst
	 * 
	 * 用index才能判断是否超出window大小
	 * 
	 * 1. 若peekFirst超过window，要从前面去掉first
	 * 2. 若队尾peeklast < cur，那要从后面去掉，因为window已经固定住，现在要求max,那么之前的都比cur小，所以去掉
	 * 3. 加入cur(当前i的max)
	 * 4. 结果Arr里加peek first，因为前面更大，但要注意index不能越界
     *
     * PS: 862. Shortest Subarray with Sum at Least K (有可能负数) 跟这题有点像，也是用单调递增deque
	 */
	public int[] maxSlidingWindowDeque(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k == 0) return new int[0];
        
        int[] res = new int[nums.length - k + 1];
        
        int idx = 0;
        Deque<Integer> deque = new ArrayDeque<>();		//存可能为max的index值
        
        for (int i = 0; i < nums.length; i++) {
            // remove the out of range ele from front。first超出K的范围
            if (!deque.isEmpty() && deque.peekFirst() <= i - k) {
                deque.poll();
            }
            //remove from last while peekLast < cur
            while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) {
                deque.pollLast();			//在这window内，把之前小的去掉，放现在比较大的..但总的还是递减
            }
            			
            deque.offer(i);			//add cur. 记得是index，这样才能看K的范围
            
            if (i >= k - 1) {       //记得判断，否则i和idx都为0时，还没算出max
                res[idx++] = nums[deque.peekFirst()];		//也可以是res[i - k + 1]
            }
        }
        return res;
    }


    /**
     * 239. Sliding Window Maximum
     * 另外奇怪的做法
     * 用leftMax[] 和 rightMax[]数组 左右扫2次
     * 最后result[i] = Math.max(rightMax[i], leftMax[i + k - 1]);
     *
     * 没太明白
     * https://leetcode.com/problems/sliding-window-maximum/discuss/65881/O(n)-solution-in-Java-with-two-simple-pass-in-the-array
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums.length == 0 || k <= 0)
            return new int[0];

        int len = nums.length;
        int[] leftMax = new int[len];
        int[] rightMax = new int[len];

        leftMax[0] = nums[0];
        rightMax[len - 1] = nums[len - 1];

        for (int i = 1; i < len; i++) {
            leftMax[i] = i % k == 0 ? nums[i] : Math.max(leftMax[i - 1], nums[i]);
            int j = len - i - 1;
            rightMax[j] = j % k == 0 ? nums[j] : Math.max(rightMax[j + 1], nums[j]);
        }

        int[] result = new int[len - k + 1];
        for (int i = 0; i + k <= len; i++) {
            result[i] = Math.max(rightMax[i], leftMax[i + k - 1]);
        }

        return result;
    }
	
	
	
	
    /** 346. Moving Average from Data Stream  - Easy
     * 维护一个size的window，每次移动，重新算average
     * @param size
     * 最容易想到的就是维护大小为size的queue，顺便存个sum当全局变量
     * 
     * 另外，如果快一点的话，可以用 int[], 然后 % window.len
     */
    public void MovingAverage(int size) {
        queue = new LinkedList();
        this.size = size;
    }
    
    Queue<Integer> queue;
    int sum = 0;
    int size = 0;
    
    public double next(int val) {
        if (queue.size() == size) {
        	sum -= queue.poll();
        }
            
        queue.offer(val);
        sum += val;
        double total = (double) sum / queue.size();
        
        return total;
    }
    
    
    /** 346. Moving Average from Data Stream
     * 另外，如果快一点的话，可以用 int[], 然后 % window.len
     * @param size
     */
    public void MovingAverage2(int size) {
        window = new int[size];
        pos = 0;
    }
     
    int[] window;
    int len;
    int pos;
    
    public double next2(int val) {
        if (len < window.length)      // q.size < size
            len++;
        sum -= window[pos];     //pos每次回++，所以到下一位
        sum += val;
        window[pos] = val;
        pos = (pos + 1) % window.length;	//相当于pos++
        
        return (double) sum / len;
    }
    
    
    
    /**
     * 394. Decode String
     * s = "3[a]2[bc]", return "aaabcbc".
		s = "3[a2[c]]", return "accaccacc".
		s = "2[abc]3[cd]ef", return "abcabccdcdcdef".
     * @param s
     * @return
     * 用2个stack，分别存重复数count 和 之前和current string
     * 注意，在 ] 时，除了push num以外，还要push cur string. 这样下次pre = strStack.pop()时，可以pre.append(cur)
     */
    public String decodeString(String s) {
        Stack<Integer> countStack = new Stack<>();
        Stack<StringBuilder> strStack = new Stack<>();
        int num = 0;
        StringBuilder sb = new StringBuilder();
        
        for (char c : s.toCharArray()) {
            if (Character.isDigit(c)) {
                num = num * 10 + c - '0';
            } else if (c == '[') {
                countStack.push(num);
                num = 0;
                strStack.push(sb);      //push cur str
                sb = new StringBuilder();
            } else if (c == ']') {
                StringBuilder pre = strStack.pop();
                int k = countStack.pop();
                while (k-- > 0) {
                    pre.append(sb);
                }
                sb = pre;
            } else {                //letter
                sb.append(c);      
            }
        }
        return sb.toString();
    }


    /**
     * 895. Maximum Frequency Stack
     * 正常push. pop()时返回most frequent的.. 如果有同样freq就pop最近加进去的
     *
     * 直接用2个hashmap搞定
     */
    class FreqStack {

        Map<Integer, Integer> freqMap;         // <num, freq>
        Map<Integer, Stack<Integer>> map;      // <freq, Stack<num>>
        int maxFreq;

        public FreqStack() {
            freqMap = new HashMap<>();
            map = new HashMap<>();
            maxFreq = 0;
        }

        public void push(int x) {
            int freq = freqMap.getOrDefault(x, 0) + 1;
            freqMap.put(x, freq);
            maxFreq = Math.max(maxFreq, freq);

            map.putIfAbsent(freq, new Stack<>());
            map.get(freq).push(x);
        }

        public int pop() {
            int top = map.get(maxFreq).pop();
            freqMap.put(top, maxFreq - 1);

            if (map.get(maxFreq).isEmpty()) {
                maxFreq--;
            }

            return top;
        }
    }



    public static void main(String[] args) {
    	StackQueue sol = new StackQueue();
    	double[] arr = sol.medianSlidingWindow(new int[]{1,3,-1,-3,5,3,6,7}, 3);
    	for (double d : arr) {
    		System.out.println(d);
    	}
    	String s = "\t\t\t\tfile2.ext";
    	System.out.println(s.length() + "  ."+s.lastIndexOf("\t"));
    	s = s.substring(s.lastIndexOf("\t")+1);
    	System.out.println(s + ".. lenght " + s.length());
	}
}
