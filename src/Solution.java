import java.io.File;
import java.time.Instant;
import java.util.*;

import org.junit.Test;
import org.omg.CORBA.INTERNAL;

/**
 * @author Yan
 *
 */
public class Solution {
	/** Pascal's Triangle
	 * @return
	 * given numRows = 5,  返回整个三角形
	 *    [1],
    	 [1,1],
   		[1,2,1],
  	   [1,3,3,1],
      [1,4,6,4,1]
	 */
	public List<List<Integer>> generate(int numRows) {
		List<List<Integer>> result = new ArrayList<>();
        if (numRows == 0) {
            return result;
        }
        
        //不用另外算 numRows=1的情况，可以直接包含在for里面，因为首先j==0就做了，不会到else get(i-1)
        
        for (int i = 0; i < numRows; i++) {                  //row
        	List<Integer> cur = new ArrayList<>();             //need to new every time
            for (int j = 0; j <= i; j++) {                  // elements in one row
                if (j == 0 || j == i) {                 //两边 设 1
                    cur.add(1);                         //包含了num=2，记住！！！
                } else {
                	List<Integer> pre = result.get(i-1);		//不用担心k=0时越界，因为在前面j==0处理了
                    cur.add(pre.get(j-1) + pre.get(j));
                }
            }
            result.add(cur);                            
        }        
        return result;
    }
	
	
	/** Pascal's Triangle II 
	 * Given an index k, return the kth row of the Pascal's triangle.
	 * k=3, Return [1,3,3,1].
	 * @param rowIndex
	 * @return
	 * 普通方法。用上一题迭代输出结果。但不用result存。只用pre, cur. 每次清空，复制
	 */
	public ArrayList<Integer> getRowII(int rowIndex) {	
        ArrayList<Integer> pre = new ArrayList<Integer>();      //start from pre
        ArrayList<Integer> cur = new ArrayList<Integer>();
        pre.add(1);                         // need to put in pre first.can't put in result AL
        if (rowIndex == 0) {                // index=0 is the 1st row
            return pre;                 
        } 
          		// 给index会减1，所以<= index
        for (int i = 1; i <= rowIndex; i++) {
            cur.clear();                                //无需new AL. 只用清空
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    cur.add(1);
                } else {
                    cur.add(pre.get(j-1) + pre.get(j));
                }
            }
            
            pre.clear();                        // 不能直接pre = cur。只能先清空，再复制
            pre.addAll(cur);
        }
        return cur;
    }
	
	/** O(k)空间的限制
	 * 每当生成下一行的时候，首先数组相应位set 1，然后从右向左计算每一个系数。
	 * @param rowIndex
	 * @return
	 */
	public List<Integer> getRowII2(int rowIndex) {
		List<Integer> list = new LinkedList<>();
        
        for (int i = 0; i <= rowIndex; i++) {
            list.add(0, 1);
            for (int j = 1; j < i; j++) {
                list.set(j, list.get(j) + list.get(j+1));
            }
        }			//也可以从后往前开始..
        
        return list;
	}
	        
		
	
	/**
	 * 36. Valid Sudoku - 用string 很方便
	 * @param board
	 * @return
	 */
	public boolean isValidSudoku11(char[][] board) {
        Set<String> set = new HashSet<>();
        
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.')
                    continue;
                
                char num = board[i][j];
                int k = i / 3 * 3 + j / 3;      // box
                if (!set.add(num + " in row " + i) ||
                    !set.add(num + " in column " + j) ||
                    !set.add(num + " in block " + i / 3 + "-" + j / 3)) {
                    
                    return false;
                }
            }
        }
        return true;
    }
	public boolean isValidSudoku(char[][] board) {
        //first dimension 0/horizontal 1/vertical 2/square
        //second dimension 0-8 represents the ith row/column/square
        //third dimension represents the occurrence of number 1-9
        boolean[][][] occur = new boolean[3][9][9];
        for (int i = 0; i < 9; i++){
            for (int j = 0; j < 9; j++){
                if (board[i][j] == '.') continue;
                int num = board[i][j] - '1';

                if (occur[0][i][num]) return false;
                else occur[0][i][num] = true;
                
                if (occur[1][j][num]) return false;
                else occur[1][j][num] = true;
                
                int s = (i / 3) * 3 + j / 3;	//或者可以用（j/3)*3，为了区分几个格子
                if (occur[2][s][num]) return false;
                else occur[2][s][num] = true;
            }
        }
        return true;
    }
	
	public boolean isValidSudoku1(char[][] board) {
        boolean[][] row = new boolean[9][9];
        boolean[][] col = new boolean[9][9];
        boolean[][] box = new boolean[9][9];
        
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    int num = board[i][j] - '1';
                    int k = i / 3 * 3 + j / 3;      //第几个大方格
                    if (row[i][num] || col[j][num] || box[k][num])
                        return false;
                    row[i][num] = col[j][num] = box[k][num] = true;
                }
            }
        }
        return true;
    }
	
	
	public boolean isValidSudoku2(char[][] board) {
        // check row 
        for (int row = 0; row < 9; row++) {
            boolean[] flag = new boolean[9];
            for (int col = 0; col < 9; col++) {
                if (!checkNew(flag, board[row][col])) {
                    return false;
                }
            }
        }
        // check column
        for (int col = 0; col < 9; col++) {
            boolean flag[] = new boolean[9];
            for (int row = 0; row < 9; row++) {
                if (!checkNew(flag, board[row][col])) {
                    return false;
                }
            }
        }
        // check all 9 box  检查9宫小格
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                 boolean flag[] = new boolean[9];
                 for (int i = 0; i < 3; i++) {
                     for(int j = 0; j < 3; j++) {
                         if (!checkNew(flag, board[row*3+i][col*3+j])) {
                            return false;
                         }
                     }
                 }
            }
        }
        return true;
    }
    
    private boolean checkNew(boolean[] flag, char c) {
        if (c == '.') {
            return true;
        }
        int index = c - '1';            //parse to int and c-1
        // 因为boolean[9],但是c 从1开始，所以index=c-1
        if (flag[index]) {          
            return false;               // already existed
        } else {
            flag[index] = true;         // never before. first occurance
            return true;
        }
    }
    
    
    
    /** 37. Sudoku Solver 给解法
     * @param board
     * solveRec(board, y+1 == 9 ? x+1 : x, y+1 == 9 ? 0 : y+1);
     * 通常是 y++往后扫。如果y==8那就要换下一行所以是x+1,0
     */
    public void solveSudoku(char[][] board){
        solveRec(board, 0, 0);
    }

    private boolean solveRec(char[][] board, int x, int y) {
    	if (x == 9 || y == 9)   return true;
        
        if (board[x][y] != '.')			// 继续往后走
            return solveRec(board, y == 8 ? x + 1 : x, y == 8 ? 0 : y + 1);
        
        for (char c = '1'; c <= '9'; c++) {
            if (isValid(board, x, y, c)) {
                board[x][y] = c;
                
                if (solveRec(board, y == 8 ? x + 1 : x, y == 8 ? 0 : y + 1))
                    return true;
                
                board[x][y] = '.';
            }
        }
        return false;					// 记得返回FALSE. 对于当前空格，如果所有的数字都不满足，则无解！
    }
    
    private boolean isValid(char[][] board, int row, int col, char k) {
        for (int i = 0; i < 9; i++) {               
            if (board[row][i] == k) {         //check same row
                return false;
            } 
            if (board[i][col] == k) {       //check same column
                return false;
            }
        }
        
        for (int i = 0; i < 3; i++) {           
            for (int j = 0; j < 3; j++) {
                // row/3 means which 9-block（小9格）. i is which pos in that 9-block
                // 3 * blockNum + i -> total index in whole table
                if (board[3 * (row/3) + i][3 * (col/3) + j] == k) {
                    return false;
                }
            }
        }
        return true;
    }
    
    
    
    
    /** 6. ZigZag Conversion
     * http://fisherlei.blogspot.com/2013/01/leetcode-zigzag-conversion.html
     * @param s
     * @param nRows
     * @return
     * 在while 循环里面，先vertical down地输入数，
     * 然后 后面的每列只能有一个数，从第nRows-2开始，一直到1 row. 每次只加一个i (从左下往右上 斜着上去)
     */
    public String convert(String s, int nRows) {
        char[] ch = s.toCharArray();
        int len = ch.length;
        StringBuilder[] sb = new StringBuilder[nRows];
        // create sb for every row
        for (int i = 0; i < nRows; i++) {
            sb[i] = new StringBuilder();
        }
        
        int i = 0;
        while (i < len) {
            for (int r = 0; r < nRows && i < len; r++) {    //vertically down
                sb[r].append(ch[i++]);
            }
            for (int r = nRows - 2; r >= 1 && i < len; r--) {    //the one in a col
                sb[r].append(ch[i++]);
            }
        }
        
        for (int r = 1; r < nRows; r++) {
            sb[0].append(sb[r]);
        }
        return sb[0].toString();
    }
    
    
    
    /** 125. Valid Palindrome - easy
     * 要忽略标点和空格，只要字母和数字
     * @param s
     * @return
     */
    public boolean validPalindrome(String s) {
        if (s.length() == 0)    return true;
        
        char[] ch = s.toLowerCase().trim().toCharArray();
        int i = 0, j = ch.length - 1;
        
        while (i < j) {
            if (!Character.isLetterOrDigit(ch[i])) {
            	i++;
            } else if (!Character.isLetterOrDigit(ch[j])) {
            	j--;
            } else {
            	if (ch[i] != ch[j])		return false;
            	i++;
            	j--;
            }
        }
        return true;
    }


    /**
     * 680. Valid Palindrome II - easy
     * 删掉at most 一个character看是否为palindrome
     * @param s
     * @return
     */
    public boolean validPalindromeII(String s) {
        int i = 0, j = s.length() - 1;

        while (i < j && s.charAt(i) == s.charAt(j)) {
            i++;
            j--;
        }
        if (i >= j)
            return true;

        return isPalindrome(s, i + 1, j) || isPalindrome(s, i, j - 1);
    }

    private boolean isPalindrome(String s, int i, int j) {
        while (i < j) {
            if (s.charAt(i++) != s.charAt(j--)) {
                return false;
            }
        }
        return true;
    }
    
    
    /** 266. Palindrome Permutation  - Easy
     * 看string能否组成 Palindrome
     * @param s
     * @return
     * 只允许一个odd
     * 方法1：扫描freq算。
     * 方法2：放hashset里，一add一remove
     */
    public boolean canPermutePalindrome(String s) {
        int[] freq = new int[128];

        // -===========one pass ====================
        int odd = 0;
        for (int i = 0; i < s.length(); i++) {
            freq[s.charAt(i)]++;
            if (freq[s.charAt(i)] % 2 == 0) {
                odd--;
            } else {
                odd++;
            }
        }
//        return odd <= 1;

        // // -=========two pass 扫2遍===================
        for (int i = 0; i < s.length(); i++) {
            freq[s.charAt(i)]++;
        }
        
        boolean isOdd = false;
        for (int n : freq) {
            if (n > 0 && n % 2 != 0) {
                if (!isOdd) {
                    isOdd = true;
                } else {
                    return false;       //more than one odd
                }
            }
        }
        // return true;
        
        // -===========其他方法=====================
        Set<Character> set = new HashSet<>();
        for (char c : s.toCharArray()) {
            if (set.contains(c)) {
                set.remove(c);
            } else {
                set.add(c);
            }
        }
        return set.size() <= 1;      //all even , or just 1 odd
    }
    
    
    /**
     * 409. Longest Palindrome
     * 只有大小写字母，需要区分.. 看能组成palindrome的最长length
     * "abccccdd" 返回7， 其中一种是 dccaccd
     * @param s
     * @return
     * 很简单.. 就跟上面一样..  奇数也能，比如5..但只能有一个奇数可以全部
     */
    public int longestPalin(String s) {
        int[] freq = new int[128];
        for (char c : s.toCharArray()) {
            freq[c]++;
        }

        boolean odd = false;
        int even = 0;

        for (int n : freq) {
            even += n / 2;
            if (n % 2 != 0) {
                odd = true;
            }
        }
        return even * 2 + (odd ? 1 : 0);
    }
    
    // 或者hashset 来remove.. 
    public int longestPalinSet(String s) {
        if(s==null || s.length()==0) return 0;
        HashSet<Character> hs = new HashSet<Character>();
        int count = 0;
        for(int i=0; i<s.length(); i++){
            if(hs.contains(s.charAt(i))){
                hs.remove(s.charAt(i));
                count++;					//存在了说明 偶数，所以count++, 并且remove
            }else{
                hs.add(s.charAt(i));
            }
        }
        if (!hs.isEmpty()) return count * 2 + 1;
        return count * 2;
    }
    
   
    
    /**
     * 5. Longest Palindromic Substring
     * @param s
     * @return
     * 从中间mid往外扩散.. 
     * 要考虑奇偶2种情况，所以要分别call两次expand来查
     * 
     * O(n^2)。因为里面expand也要O(n)
     */ 
    public String longestPalindrome(String s) {
        int i = 0, j = 0;
        for (int k = 0; k < s.length(); k++) {
            int oddLen = expandPalindrome(s, k, k);		//palindrome的最长length
            int evenLen = expandPalindrome(s, k, k + 1);
            
            int len = Math.max(oddLen, evenLen);
            // 更新max len对应的i, j
            if (len >= j - i + 1) {			//若大于之前的j-i+1长度，就要更新
                i = k - (len - 1) / 2;		// 偶数时，中心偏前面，所以要(len-1)/2
                j = k + len / 2;			
            }
            
        }
        return s.substring(i, j + 1);
    }
    
    public int expandPalindrome(String s, int i, int j) {
        while (i >= 0 && j < s.length() && s.charAt(i) == s.charAt(j)) {
            i--;
            j++;
        }
        return j - i - 1;
    }


    /**
     * 5. Longest Palindromic Substring
     * DP 解法
     * boolean[][] dp.. dp[i][j] 表示 i~j is palindrome TRUE
     * 所以  s[i] == s[j] 时，看 j <= i + 1（self或邻居） 或者  dp[i + 1][j - 1]是否也true
     */
    public String longestPalindromeDP(String s) {
        if (s == null || s.length() < 1)
            return "";
        int n = s.length();
        String max = "";

        boolean[][] dp = new boolean[n][n];

        // for (int j = 0; j < n; j++) {            // 或者这样 j 先走
        //     for (int i = 0; i <= j; i++) {
        for (int i = n - 1; i >= 0; i--) {
            // dp[i][i] = true;                 // 如果 j = i + 1 开始，需要设true. 同时初始时max += s.charAt(0)
            for (int j = i; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    // self 或 相邻
                    if (j - i <= 1 || dp[i + 1][j - 1]) {
                        dp[i][j] = true;
                        if (max.length() < j - i + 1) {
                            max = s.substring(i, j + 1);
                        }
                    }
                }
            }
        }


        // ==================或者用一维的===================================
        boolean[] dp1 = new boolean[n];

        for (int i = n - 1; i >= 0; i--) {
            for (int j = n - 1; j >= i; j--) {      // 因为一维，需要从后往前，这样不会覆盖掉上一轮的

                // 每次都要更新dp[j].. 不能if里，否则如果上一次true的话，这次false不覆盖，那会错的
                dp1[j] = s.charAt(i) == s.charAt(j) && (j - i <= 1 || dp1[j - 1]);
                                                            // dp[j-1] is the same as dp[i+1][j-1]
                if (dp1[j] && max.length() < j - i + 1) {
                    max = s.substring(i, j + 1);
                }
            }
        }

        return max;
    }



    /**
     * 647. Palindromic Substrings
     * Count how many palindromic substrings in this string. 算个数
     *
     * Input: "aaa"
     * Output: 6
     * Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
     *
     * 跟上一题非常像
     */
    public int countSubstrings(String s) {
        int count = 0;

        for (int mid = 0; mid < s.length(); mid++) {
            count += expandPalindromeResult(s, mid, mid);
            count += expandPalindromeResult(s, mid, mid + 1);
        }
        return count;
    }

    private int expandPalindromeResult(String s, int i, int j) {
        int count = 0;
        while (i >= 0 && j < s.length() && s.charAt(i) == s.charAt(j)) {
            i--;
            j++;
            count++;
        }
        return count;
    }

    // DP
    public int countSubstrings2(String s) {
        int count = 0;
        int n = s.length();

        boolean[][] dp = new boolean[n][n];

        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                dp[i][j] = s.charAt(i) == s.charAt(j) && (j - i <= 1 || dp[i + 1][j - 1]);
                if (dp[i][j]) {
                    count++;
                }
            }
        }
        return count;
    }
    
    
    /**
     * 516. Longest Palindromic Subsequence
     * 这个是subsequence.. 中间能断开的 给"bbbab"，return 4，因为bbbb
     * @param s
     * @return
     * 大多数subsequence用DP来做。然后这个跟每个区间是否为palindrome相关，所以是区间型DP
     *
     * dp[i][j] means max palindrome length from i ~ j
     */
    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];     // max palindrome length from i ~ j

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

    // 一维的 DP
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


    /** 用 dfs + nextNum
     */
    public int longestPalindromeSubseqDFS(String s) {
        int n = s.length();
        return helper(s, 0, n - 1, new int[n][n]);
    }
    
    public int helper(String s, int i, int j, int[][] cache) {
        if (cache[i][j] > 0)      return cache[i][j];
        if (i > j)             return 0;
        if (i == j)            return 1;
        
        if (s.charAt(i) == s.charAt(j)) {
            cache[i][j] = 2 + helper(s, i + 1, j - 1, cache);
        } else {
            cache[i][j] = Math.max(helper(s, i + 1, j, cache), helper(s, i, j - 1, cache));
        }
        return cache[i][j];
    }


    /**
     * 730. Count Different Palindromic Subsequences
     * 算不同palindromic subsequnce的个数. 需要 modulo 10^9 + 7
     *
     * S = 'bccb'
     * Output: 6
     * Explanation:
     * The 6 different non-empty palindromic subsequences are 'b', 'c', 'bb', 'cc', 'bcb', 'bccb'.
     * Note that 'bcb' is counted only once, even though it occurs twice.
     *
     *
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
     *   还需要看 中间里面的 字母是否相同，减去重复计算的
     *   b1. left > right   里面没有重复的，加上最外层 'a'和'aa'
     *       dp[i][j] += 2
     *   b2. left == right  有一个重复。比如"aaa"的情况，里面中间那个'a'之前算过了，这次只能加外层'aa'
     *       dp[i][j] += 1
     *   b3. left < right   有多个重复 (跟s[i]一样)字母，需要减去之前算过的
     *       dp[i][j] -= dp[left + 1][right - 1]  减 ...
     *
     * https://leetcode.com/problems/count-different-palindromic-subsequences/discuss/112757/Java-solution-using-simple-DP.-O(n2)-run-time-and-O(n2)-space
     */
    public int countPalindromicSubsequences(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];     // i~j 的不同palindrome个数

        // 初始化
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }

        for (int len = 1; len < n; len++) {
            for (int i = 0; i < n - len; i++) {
                int j = i + len;

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
     * 214. Shortest Palindrome
     * 给个String，在最前面加字母，使得最后形成一个最短的palindrome
     * Given "aacecaaa", return "aaacecaaa"。  Given "abcd", return "dcbabcd".
     * @param s
     * @return
     * 这题可以转化成 -> 求 找mid能扩散到left=0的情况的 最长palindrome。然后把后面不是回文的reverse加在前面就行
     * 
     * 这种 n^2 的复杂度，做法借鉴了上一题Longest Palindromic Substring 
     * 因为要找最长palindrome，这样加的数可以最短。
     * 但是要注意 !!! 需要是包括0也是palindrome的. 如果中间才是palindrome, 那么没用，需要整个string都reverse就太长了
     * 1. 首先只用关注 中点i在前半段 i <= len/2. 
     * 		超过一半的话说明center在后半段，那就基本需要整个string reverse了，不对
     * 2. 算maxL的最长palindrome时，记得左边界要一直延伸到0，其他一律返回-1.
     * 		只有0是palindrome的一部分时才能加最少，否则如果bcaacd, 虽然caac是回文，但没法延伸到0，还是需要整个string reverse
     */
    public String shortestPalindrome(String s) {
        if (s == null || s.length() <= 1)
            return s;

        // find the longest palin beginning at the left
        int len = s.length();
        int maxL = 0;

        for (int i = len / 2; i >= 0; i--) {         // 只找前half
            int len1 = expandPalind(s, i, i);
            int len2 = expandPalind(s, i, i + 1);
            maxL = Math.max(maxL, Math.max(len1, len2));    //maxL start from 0 . len1,2返回 -1 的话maxLen还是0

            if (maxL > 0)       // 早点break
                break;
        }

        String suffix = s.substring(maxL + 1);      //suffix is not palindrome

        return new StringBuilder(suffix).reverse().toString() + s;
    }
    
    public int expandPalind(String s, int i, int j) {
        while (i >= 0 && j < s.length() && s.charAt(i) == s.charAt(j)) {
            i--;
            j++;
        }
        if (i < 0)          //only for start from 0 记得左边要延伸到0
            return j - 1;
            
        return - 1;
    }


    /**
     * 214. Shortest Palindrome
     * @param s
     * @return
     * 这个快点
     * https://leetcode.com/problems/shortest-palindrome/discuss/60098/My-7-lines-recursive-Java-solution
     */
    public String shortestPalindrome1(String s) {
        if (s == null || s.length() <= 1)
            return s;

        int len = s.length();
        int i = 0;

        for (int j = len - 1; j >= 0; j--) {
            if (s.charAt(i) == s.charAt(j)) {
                i++;
            }
        }

        if (i == len)
            return s;

        String suffix = s.substring(i);
        String reversed = new StringBuilder(suffix).reverse().toString();

        // 中间需要 shortestPalindrome(s.substring(0, i)) 因为不确定这个本身就是最大回文前缀，我们只能确定后面剩余的部分肯定不属于palindrome
        return reversed + shortestPalindrome1(s.substring(0, i)) + suffix;
    }
    
    
    /** 214. Shortest Palindrome
     *
     * 同样的思路 -> 求 从0开始的最长palindrome。
     * 但是这回不跟上一个方法那么O(n^2)求最长palindrome. 而是用KMP里其中一步，求longest相同前后缀
     * KMP里那个lps[]存的是从0开始，最长的前缀，所以跟这题很符合
     * 
     * 1. tmp = s + "#" + reverse 组成tmp。 这里先s 再reverse，因为是s从0开始的suffix，不能颠倒
     * 		中间加 # 为了区分 s和reverse(s). 否则如果 last跟begin一样的话，没有#在中间会confuse
     *   把s 分为A1, A2. 其中A1是longest相同的prefix, A2对应的另外要加的
     *   所以是 tmp = A1, A2 # A2', A1'。那么希望 头尾的 A1和A1'能尽量最长，这时就用到KMP的longest相同前后缀了
     *
     * 2. 根据tmp来建 lps[]
     * 
     * 3. 最后lps[]里last那个数就是整个s以0开始的最长palindrome，因为是找A1和A1'最长的相同prefix,suffix
     * 
     * 4. 把longestPalin后面那部分reverse，再加上s  就是结果
     */
    public String shortestPalindromeKMP(String s) {
        if (s == null || s.length() <= 1) return s;
        
        String tmp = s + "#" + new StringBuilder(s).reverse().toString();
        int[] lps = longestPrefixSuffix(tmp.toCharArray());
        
        int longestPalin = lps[lps.length - 1];     //the longest palindrome start from 0
        return new StringBuilder(s.substring(longestPalin)).reverse().toString() + s;
    }
    
    
    
    /**
     * 336. Palindrome Pairs
     * Given words = ["abcd", "dcba", "lls", "s", "sssll"]. Return [[0, 1], [1, 0], [3, 2], [2, 4]]
		The palindromes are ["dcbaabcd", "abcddcba", "slls", "llssssll"]
     * @param words
     * @return
     * 用HashMap来存，这样看看map里是否有reverse过的词。因为是两两比较，用map节省时间
     * 1. 首先for一遍，把所有word放map里。这样第二次for时不需要put，因为都在第一次for的map里
     * 
     * 2. 再for一次开始判断。
     * 	 把这个word切分substring成left跟right部分，看看是否其中一个存在map&&另一个isPalindrome
     * 
     * 	其中有4种情况要放result里
     *    a. 当有""空时，只要word是palindrome就可以。"a"也算是 - 这是cut从 =0, <= len，包含了""的情况
     *    		由于包含了""的情况，有可能前面""，或者后面""，所以可以在第二次后面时让 s2 != "", 防止重复加
     *    b. 当整个word是另一个的reverse。这个也包含在cut(k)里了
     *    c. left是palindrome，right的reverse存在map里
     *    d. right是palindrome，left的reverse存在map里
     * 
     * 有几点要注意
     * 1. cut从 =0, <= len，包含了""
     * 2. 第二次后面时让 s2 != ""
     * 3. 第二次for时不要put东西，因为第一次for已经放完了。
     * 4. 判断左右是否palindrome时，是都要check，而非if-else
     * 5. map.get(rvs) != i. 防止duplicate
     * 
     * 复杂度是 O(n * k^2). 其中n是words数组的len, k是平均每个词的长度
     * 因为2个for循环就是 n * k. 但是内循环里还要判断isPalindrome, 这也要O(k) 
     */
    public List<List<Integer>> palindromePairs(String[] words) {
    	List<List<Integer>> result = new ArrayList<>();
        Map<String, Integer> map = new HashMap<>();
        
        // put whole word
        for (int i = 0; i < words.length; i++) { 
            map.put(words[i], i);
        }
        
        // includes all cases : "", "a", whole word, half reverse
        for (int i = 0; i < words.length; i++) {
            for (int k = 0; k <= words[i].length(); k++) {      // =0, <= len , includes ""
                String left = words[i].substring(0, k);
                String right = words[i].substring(k);

                if (isPalindrome(left)) {
                    String reverseR = new StringBuilder(right).reverse().toString();
                    if (map.containsKey(reverseR) && map.get(reverseR) != i) {      // != i avoid dup
                        result.add(Arrays.asList(map.get(reverseR), i));        // ba, ss(ab)
                    }
                }
                                    //avoid duplicate."ab", "ba" 因为上面的left考虑过分成""的情况
                if (isPalindrome(right) && right.length() != 0) {   //avoid duplicate."a", "" 
                    String reverseL = new StringBuilder(left).reverse().toString();
                    if (map.containsKey(reverseL) && map.get(reverseL) != i) {
                        result.add(Arrays.asList(i, map.get(reverseL)));        // ab(ss), ba
                    }					//记得是 (i, map.get(reverse).. 因为right是palindrome,放中间
                }
            }
        }
        return result;
    }
    
    public boolean isPalindrome(String s) {
        int i = 0, j = s.length() - 1;
        while (i < j) {
            if (s.charAt(i++) != s.charAt(j--)) {
                return false;
            }
        }
        return true;
    }
    
    private String reverseStr(String s) {
        return new StringBuilder(s).reverse().toString();
    }


    /**
     * 336. Palindrome Pairs
     * 用Trie实现.. Trie里存的是倒序 suffix
     *
     * a. 用idx表示 整个word结尾.. 那么中间palindrom是看对比word的后半部分  word(left) + word(right) + node
     * b. 用palins表示 看node..倒序后面（也就是正序的左中间）是否palindrome  word + node(left) + node
     *
     * 同时TrieNode里有palins list来记录 中间部分是否为palindrome.
     * 中间部分如果是""，说明我们已经node和word都到最后，说明两者match.. 那么 表示word + node(倒序) 整个是palindrome
     *
     * http://www.allenlipeng47.com/blog/index.php/2016/03/15/palindrome-pairs/
     */
    public List<List<Integer>> palindromePairsTrie(String[] words) {
        List<List<Integer>> result = new ArrayList<>();
        TrieNode root = new TrieNode();

        // 插入时 后缀 suffix
        for (int i = 0; i < words.length; i++) {
            insertSuffix(root, words[i], i);
        }

        for  (int i = 0; i < words.length; i++) {
            search(result, root, words[i], i);
        }

        return result;
    }

    private void insertSuffix(TrieNode node, String word, int idx) {
        // 从后往前插 suffix
        for (int i = word.length() - 1; i >= 0; i--) {
            // 前半部分（Trie里是后面）是palindrome的话也加上, 这样 word + node(palin) + node(倒序 (i,len)) 整体是palindrome
            if (isPalindrome(word, 0, i)) {
                node.palins.add(idx);
            }
            char c = word.charAt(i);
            if (node.children[c - 'a'] == null) {
                node.children[c - 'a'] = new TrieNode();
            }
            node = node.children[c - 'a'];
        }

        node.index = idx;
        node.palins.add(idx);       // 这是结尾了，那么后面其实就是空"", 所以到时比的是 word + node(倒序)
    }

    private void search(List<List<Integer>> result, TrieNode node, String word, int idx) {
        int len = word.length();
        for (int i = 0; i < len; i++) {
            // 这是把word划分   word(left) + word(right) + node
            // 这是一整个词      &&  不是word（位置不同）&&  word后面是palindrome.. w: lasss, node: la (倒序是个词，正序是al)
            if (node.index >= 0 && idx != node.index && isPalindrome(word, i, len - 1)) {
                result.add(Arrays.asList(idx, node.index));
            }

            node = node.children[word.charAt(i) - 'a'];

            if (node == null)
                return;
        }

        // 这是把node划分两部分的情况  word + node(left) + node
        // 最后停在trie的最后或者中间，palins存的是这个node之后是否为palindrome.. word + node(palin) + node(倒序 (i,len))
        // 也就是这个 正序词的前半部分 (中间) 是否palindrom..
        // 比如 [“xyxcba”, “abc”, “xcba”].. 在检查word为"abc"时
        // 正序 xyxcba，Trie里倒序存 abcxyx.. 上面for检查完"abc"后，停在了c.. c后面是"xyx" 和 "x" 都是palindrome，所以加上它们
        for (int i : node.palins) {
            if (i == idx)
                continue;
            result.add(Arrays.asList(idx, i));
        }
    }

    class TrieNode {
        TrieNode[] children;
        int index;
        List<Integer> palins;       // 存node往后是否为palindrome.. 也就是正序词的左边，作为match的中间部分

        public TrieNode() {
            children = new TrieNode[26];
            index = -1;
            palins = new ArrayList<>();
        }
    }


    /**
     * 564. Find the Closest Palindrome
     * 找到里num最近的palindrome. 如果有一样近的，取smaller, 本身是palindrome的话不能是自己
     * 123 -> 121 .  11 -> 9. 10 -> 9 (9比11小).  101 -> 99
     * @param n
     * @return
     * 主要按照left half来.. 而且要考虑3种情况，把 9 的情况也考虑进来
     */
    public String nearestPalindromic(String n) {
        long num = Long.valueOf(n);    //字符串转化为整数
        long half = (long) Math.pow(10L, n.length() / 2);       // 前左半段left half
        long candidate =  num / half * half;

        // num = 123, 左半段half=10,   candidate = 120 (121) ||  119 (111) ||  130 (131)
        // num = 99,  左半段half=10,   candidate = 90 (99)   ||  89 (88)  ||   100  (101)
        // num = 101,  左半段half=10,  candidate = 100 (101) ||  99 (99)  ||   110  (111)
        long[] candidates = new long[] {candidate, candidate - (half > 1 ? half/10 : 1), candidate + half};	//构造三个数字
        long res = 0;

        for (long cand : candidates) {	//对3个数字每次进行构造回文串
            cand = mirroring(cand);
            if (cand == num) {
                continue;
            }
            if (Math.abs(res - num) > Math.abs(cand - num) || (Math.abs(res - num) == Math.abs(cand - num) && cand < res)) { //比较结果
                res = cand;
            }
        }
        return ((Long) res).toString();
    }

    private long mirroring(long n) {
        char[] arr = ((Long) n).toString().toCharArray();
        int len = arr.length;
        int i = len / 2;
        int j = i;
        if (len % 2 == 0) {
            i--;
        }
        while (i >= 0 && j < len) {
            if (arr[i] != arr[j]) {
                arr[j] = arr[i];
            }
            i--;
            j++;
        }
        return Long.valueOf(new String(arr));
    }

    // 也可以  数学方法
    public long mirroring1(long n) { //传入数字构造回文串
        long m = 0, x = n, half = 1, i = 0;
        while (x > 0) {
            m = m * 10 + x % 10;
            x /= 10;
            if (i++ % 2 > 0) half *= 10;
        }
        return n / half * half + m % half;     //n / half * half可以清空后半部分，  + m % half填充后半部分
    }


    
    
    /**
     * 134. Gas Station
     * N gas stations along a circular route, where the amount of gas at station i is gas[i].
     * car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its next station (i+1). 
     * Return the starting gas station's index if you can travel around the circuit once, otherwise return -1
		The solution is guaranteed to be unique.
     * @param gas
     * @param cost
     * @return
     * 1. If car starts at A and can not reach B. Any station between A and B can not reach B.(B is the first station that A can not reach.)
	 * 2. If the total number of gas is bigger than the total number of cost.因为是环形 There must be a solution.
	 *      g1+g2 > c1+c2. 如果g1<c1, 那就看g2. 因为最前面的式子，所以g2 > c2, 所以能从g2开始，并走完g1
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
    	int curr = 0;
        int remain = 0;
        int start = 0;     
        for (int i = 0; i < gas.length; i++) {
            curr += gas[i] - cost[i];			
            remain += gas[i] - cost[i];         // 记录总的从0~n-1, no matter < 0，这样就不用考虑2nd方法的到头重新0开始算的问题
            if (curr < 0) {                     //current round < 0 就重新开始
            	start = i + 1;                    // record the new start。这是不行的i, 所以新开始是i+1
                curr = 0;                       // 既然重新开始，那就要reset to 0
            }
        }
        
        return remain < 0 ? -1 : start ;
    }
    
    /** Gas Station 
     * 方法比较差 O(n^2), 2个循环
     * @param gas
     * @param cost
     * @return
     */
    public int canCompleteCircuit2(int[] gas, int[] cost) {
        int i = 0, j = 0;
        while (i < gas.length) {
            int sumGas = 0;                 //should put inside, not outside
            int sumCost = 0;				// 因为每次要初始
            j = i;
            do {
                sumGas += gas[j];
                sumCost += cost[j];
                if (sumCost > sumGas) {
                    break;
                }
                if (j == gas.length - 1) {
                    j = 0;
                } else {						//记得放else里
                    j++;
                }
            } while (j != i && j < gas.length); 	// while后要 ;  且 && j < gas.length
            
            if (sumCost <= sumGas) {            // 无需写i==j, 因为while loop 的终止条件就是 i==j
                return i;
            }
            i++;                                //after if()
        }
        return -1;
    }
    
    
    
    /**
     * 135. Candy
     * There are N children standing in a line. Each child is assigned a rating value.
     * You are giving candies to these children subjected to the following requirements:
     * Each child must have at least one candy.
     * Children with a higher rating get more candies than their neighbors.
     * What is the minimum candies you must give?
     * Input: [1,0,2]
     * Output: 5
     * Explanation: You can allocate to the first, second and third child with 2, 1, 2 candies respectively.
     *
     * 
     * 1. 最少要每人一个，所以刚开始都设1 (或者也可以第一次for时，前面>后面，那就设1)
     * 2. 从左走一遍，后面大的记得cand[i+1] = cand[i] + 1;
     * 3. 然后再从后往前走一遍看有没少加.. 记得要判断max(cand[i-1], cand[i] + 1)。
     * 
     * 注意加糖时，不能单纯cand[i]++..而要cand[i-1] + 1. 保证比前面大
     */
    public int candy(int[] ratings) {
        if (ratings == null || ratings.length == 0) {
            return 0;
        }
        int[] cand = new int[ratings.length];
        Arrays.fill(cand, 1);
        int sum = 0;
        
        // check r[i] & r[i-1]
        for (int i = 0; i < ratings.length - 1; i++) {
            if (ratings[i] < ratings[i+1]) {
                cand[i+1] = cand[i] + 1;			//注意不是单纯cand[i+1]++
            }	
      //      else cand[i] = 1;   		//如果前面array不初始化1的话，可以在这里改就行 
        }
        
        // check r[i] & r[i+1], so from end to start
        for (int i = ratings.length - 1; i >= 1; i--) {
        	if (ratings[i-1] > ratings[i]) {
                cand[i-1] = Math.max(cand[i-1], cand[i] + 1);   //有可能cand[i-1]本身就已经够大了
            }
            sum += cand[i];
        }
        sum += cand[0];		//或者sum单独开一个for循环从0开始加..这样稍微慢点
        
        return sum;
    }
    
    
    

    /**
     * 31. Next Permutation - 更好
     * rearranges numbers into the lexicographically next greater permutation of numbers.
     * 1,2,3 → 1,3,2     3,2,1 → 1,2,3
     *
     * 需要从 *后* 往前扫，要找到1st decreasing number，那么这个decrease num是需要换的..
     * 然后再从后面找1st larger than 这个数的，替换掉就好.. 最后那段需要reverse (从小到大排好序)
     * 
     * 从右往前需要保持递增。 6，3，4，9，8，7，1
     * 1. 遇到第一个小的数n[i-1]，就要换。此时找到i是 4，second 9
     * 2. 需要找比n[i-1]大一点的数n[nextLarge],互相swap. 因为后面都是递减，所以从尾巴找第一个 > first的7 。变成6，3，7，9，8，4，1
     * 3. 再reverse [i,lastEle]这区间的数 6，3，7，1，4，8，9
     * 
     * 比如1，3，2 --> 1<3, 要比1大一点的2来swap 2, 3, 1 --> reverse剩下的 2,1,3
     * @param nums
     */
    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length <= 1) {
            return;
        }
        
        int last = nums.length - 1;     // the last index in nums[]
        int i = last - 1;

        // 找到i 作为要换的点
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }
        
        if (i == 0) {       // 整个数都decreasing, 那就reverse 5,4,3,2 -> 2,3,4,5
            reverse(nums, 0, last);
            return;
        }
        
        int nextLarge = last;

        while (nextLarge > i && nums[i] >= nums[nextLarge]) {       //正常找到的话, i是4，nextLarge是7
            nextLarge--;
        }

        swap(nums, i - 1, nextLarge);
        reverse(nums, i, last);
    }
    
    public void swap(int[] n, int a, int b) {
        int tmp = n[a];
        n[a] = n[b];
        n[b] = tmp;
    }
    
    public void reverse(int[] n, int i, int j) {
        for (; i < j; i++, j--) {
            swap(n, i, j);
        }
    }
    

	
	/**
     * Next Permutation - 没那么好
	 * @param num
	 * 跟上面一样的思路，只是做法不同，嵌套在里面，觉得没那么清晰。
	 * 
	 * 从后往前扫，如果n[i-1]<n[i],那就考虑替换，但要找大于n[i-1]的最小值。
	 * 所以再加多个循环从最后扫。
     * swap完再排下序
	 */
	public void nextPermutation2(int[] num) {
        if (num == null || num.length <= 1) {
            return;
        }
        
        for (int i = num.length - 1; i > 0; i--) {
            if (num[i - 1] < num[i]) {              // find the 1st descend ele
            	// 后面这段降序的，从后往前找 比i-1 大的 最小的数，很快找到
                for (int j = num.length - 1; j >= i; j--) {  //search smallest from descend parts
                    if (num[j] > num[i - 1]) {      // least bigger than ele needed to swap
                        swap(num, i - 1, j);
                        reverse(num, i, num.length - 1);    // 要从i-1后面一位开始
               //  也可以用Arrays.sort 。[fromIndex,toIndex). 后面的exclusive所以不用len-1
               //  这sort是quicksort，nlog(n)         
              //        Arrays.sort(num, i, num.length);    // 要从i-1后面一位开始
                        return;
                    }
                }
            }
        }
        Arrays.sort(num);	//也可以用reverse。已经是最大全降序，所以变成全升序		
    }
    
    
    
    /**
     * 60. Permutation Sequence
     * 给n算n!, 找第k个组合
     * 找第一位A1时，剩下(n-1)!种排列，所以需要K / (n-1)!
     * http://blog.csdn.net/fightforyourdream/article/details/17483553
     * @param n
     * @param k
     * @return
     */
    public String getPermutation(int n, int k) {
        if (n < 0 || k <= 0) {
            return "";
        }
          
        StringBuilder sb = new StringBuilder();
        List<Integer> num = new ArrayList<Integer>();
        int factor = 1;
        for (int i = 1; i <= n; i++) {
            num.add(i);				//填数字12345, 这样用完的可以去掉
            factor *= i;			//算n!
        }
          
        k--;							// 因为index从0开始，而不是1
        for (int i = 0; i < n; i++) {
            factor = factor / (n - i);		// 第一位(n-1)!, 第二位(n-2)!...以此类推
            int index = k / factor;			// 该位应选择的数的下标
            sb.append(num.get(index));
            num.remove(index);			// 去掉用过的数，比如用过4，后面就只能从1235抽了
            k %= factor;				// 剩下的是余数。
        }
      
        return sb.toString();  
    }
    
    
    
    
    /** 127. Word Ladder 
     * 给2words (start and end), and a dictionary, find the length of shortest transformation sequence from start to end, such that:
		Only one letter can be changed at a time。 Each intermediate word must exist in the dictionary
		start = "hit"， end = "cog"。 dict = ["hot","dot","dog","lot","log"]
		As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
		return its length 5.
     * @param start
     * @param end
     * @param dict
     * @return
     * 其实这是个BFS问题
     * 每次变一个位置的字母.. 找找看dict里是否有。这一层都找完了，就level++, 再找下一层。
     * 这BFS也保证了可以有最短路径.. 因为每层只加1次level，找到就提前返回
     * 
     * 注意几点...
     * a. 可以把string变成char[]，再在第j位换字母，之后整个再转回str.. 这样比substring方便，也快
     * b. 记得用queue来放BFS..
     * c. 记得每次找到dict里的，就要删掉它。否则会重复的transform导致循环。或者用visited,但浪费空间
     * d. char[] ch = word.toCharArray(); 记得这里放在变字母前..如果放在for(i)前，会被改变，需要每次Reset ch[i] = old回来
     */
    public int ladderLength(String start, String end, Set<String> dict) {
        if (dict == null || dict.size() == 0) {
            return 0;
        }
        
        if(!dict.contains(end)) 
            return 0;
        
        Queue<String> queue = new LinkedList<String>();
        queue.offer(start);
        dict.remove(start);

        // 放parent的map这样能打印一条最短path。 key: child; val: parent
        Map<String, String> parentMap = new HashMap<>();
        
        int level = 2;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                String cur = queue.poll();              
                for (int j = 0; j < cur.length(); j++) {
                	char[] ch = cur.toCharArray();
                    for (char c = 'a'; c <= 'z'; c++) {
                    	ch[j] = c;			//在第j个位置换字母
                        String trans = new String(ch);
                        if (trans.equals(end)) {
                            parentMap.put(end, cur);                    // print one path
                            printPath(parentMap, start, end);

                            return level;
                        }
                        if (dict.contains(trans)) {
                            queue.offer(trans);
                            dict.remove(trans);		//记得删掉这个，保证不会重复地transform

                            parentMap.put(trans, cur);
                        }
                    }
                }
            }
            level++;		// 每轮结束后(size), level++
        }
        return 0;
    }

    private void printPath(Map<String, String> parentMap, String start, String end) {
        LinkedList<String> path = new LinkedList<>();

        while (!end.equals(start)) {
            path.addFirst(end);
            end = parentMap.get(end);
        }
        path.addFirst(start);

        System.out.println("shortest path is " + path);
    }
    
    
    /** 127. Word Ladder  跟上面一样... 但是从头尾两头 找
     * 这样更快.. 其他里面的过程都是一样的
     * 
     * 主要是，分begin和end 两个set。for (begin)里的词，看有没在end里出现
     * 1. 每次从begin换字，看字典里有没有，有的话这trans就加到next set里。
     * 2. 一轮(begin set)完成后，begin=next，也就是，从这些新找的trans里换字再找。
     * 
     * !!!!!! 要注意一点，保证begin set更小，这样每次 换字+搜索 会快点
     */
    public int ladderLength1(String beginWord, String endWord, Set<String> wordList) {
    	
    	if(!wordList.contains(endWord)) 
            return 0;
    	
        Set<String> begin = new HashSet<>();   
        Set<String> end = new HashSet<>();   
  //      Set<String> visited = new HashSet<>();   因为删掉wordList匹配的词，所以不需要..否则在加trans到next时要判断是否visit过
        begin.add(beginWord);
        end.add(endWord);
        int level = 2;	
        
        while (!begin.isEmpty() && !end.isEmpty()) {
            // make sure begin has smaller size, 这样有时会跟end调换，从end words出发
            if (begin.size() > end.size()) {    
                Set<String> tmp = begin;
                begin = end;
                end = tmp;
            }
            
            Set<String> next  = new HashSet<>();
            for(String word : begin) {          // check words from begin set
                for (int i = 0; i < word.length(); i++) {
                    char[] ch = word.toCharArray();     //记得这里放在变字母前..如果放在for(i)前，会被改变，需要每次Reset回来
                    for (char c = 'a'; c <= 'z'; c++) {
                        ch[i] = c;
                        String trans = new String(ch);
                        if (end.contains(trans)) {          // see if begin and end can meet
                            return level;   
                        }
                        if (wordList.contains(trans)) {
                            next.add(trans);
                            wordList.remove(trans);
                        }
                    }
                }
            }
            // in this round, some new word trans放到了next set里。下一次要从这些新的trans里找，所以要放在begin里
            begin = next;     
            level++;
        }
        return 0;
    }
    
    /**
     * 127. Word Ladder  法3
     * 
     * 这个比较适用于，如果不单是lower case字母。。 如果还有别的字符，那先preprocess会好点
     * 
     * 先preprocess word list把他们generalize key，放到map里。key "*ot" -> {dot, lot}
     * 
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Map<String, List<String>> map = new HashMap<>();
        
        preProcessWords(wordList, map);
        
        Queue<String> q = new LinkedList<>();
        q.add(beginWord);
        
        Set<String> visited = new HashSet<>();
        
        int level = 2;
        
        while (!q.isEmpty()) {
            int size = q.size();
            while (size-- > 0) {
                String word = q.poll();
                for (int i = 0; i < word.length(); i++) {
                    String key = generalizeString(word, i);
                    if (map.containsKey(key)) {
                        for (String s : map.get(key)) {     // 代替了原先 for(c = 'a' <= 'z')
                            if (s.equals(endWord)) {
                                return level;
                            }
                            if (!visited.contains(s)) {
                                visited.add(s);
                                q.add(s);
                            }
                        }
                    }
                }
            }
            level++;
        }
        return 0;       // can't find
    }
    
    // process words and put in map: key "*ot" -> {dot, lot}
    private void preProcessWords(List<String> list, Map<String, List<String>> map) {
        for (String s : list) {
            for (int i = 0; i < s.length(); i++) {
                String key = generalizeString(s, i);
                if (!map.containsKey(key)) {
                    map.put(key, new ArrayList<>());
                }
                map.get(key).add(s);
            }
        }
    }
    
    private String generalizeString(String s, int i) {
        return s.substring(0, i) + '*' + s.substring(i + 1);
    }
    
    
    /** 126. Word Ladder II
     * 输出所有可能的shortest结果
     *
     *
     * 相当于建graph, 然后输出shortest path..
     * 
     * 1. 首先bfs得到graph, key是word，能用于dfs能知道前面能到达的word list是什么
     *    并且用distance map记录level
     * 2. 再用dfs从后往前输出路径   （其实也可以从前往后加到path，直到遇到end..不过这样慢一点因为也有更长的路）
     * https://wdxtub.com/interview/14520607221562.html
     */
    public List<List<String>> findLadders(String beginWord, String endWord, Set<String> wordList) {
        List<List<String>> result = new ArrayList<>();
        Map<String, List<String>> graph = new HashMap<>();
        Map<String, Integer> distance = new HashMap<>();
        
        bfs(graph, distance, beginWord, endWord, wordList);

        // 打印所有path
        dfsPath(result, new LinkedList<String>(), endWord, beginWord, distance,  graph);

        return result;
    }
    
    
    public void bfs(Map<String, List<String>> graph, Map<String, Integer> distance, String start, String end, Set<String> dict) {
        Queue<String> q = new LinkedList<>();
        q.add(start);
        dict.add(start);			//记得加，才能构造graph
        dict.add(end);
        distance.put(start, 0);
        boolean found = false;
        
        for (String s : dict) {			//字典里的每个词都先初始化一个空list
        	graph.put(s, new ArrayList<String>());
        }
        
        while (!q.isEmpty()) {
            int qsize = q.size();
            for (int i = 0; i < qsize; i++) {
                String word = q.poll();
                List<String> transWords = genTransforms(word, dict);

                for (String neighbor : transWords) {
                    graph.get(neighbor).add(word);
                    if (!distance.containsKey(neighbor)) {
                        distance.put(neighbor, distance.get(word) + 1);
                        if (neighbor.equals(end)) {
                            found = true;
                        } else {
                            q.add(neighbor);
                        }
                    }
                }
            }
            if (found)
                break;
        }
    }
    
    
    // generate a list of words that transform from "word"
    public List<String> genTransforms(String word, Set<String> dict) {
    	List<String> transList = new ArrayList<>();
    	for (int i = 0; i < word.length(); i++) {
            char[] ch = word.toCharArray();
            for (char c = 'a'; c <= 'z'; c++) {
            	if (ch[i] == c)		continue;
                ch[i] = c;
                String trans = new String(ch);
                if (dict.contains(trans)) {
                	transList.add(trans);
                }
            }
        }
    	return transList;
    }
    
    
    // generate path from end to start
    public void dfsPath(List<List<String>> res, LinkedList<String> path, String word, String start, Map<String, Integer> distance, Map<String, List<String>> graph) {
    	if (word.equals(start)) {		//found
    		path.addFirst(word);
    		res.add(new ArrayList<>(path));
    		path.remove(0);             // 记得remove
    		return;
    	}
    	
    	for (String nei : graph.get(word)) {		// 从后往前加，word在nei后面
    		if (distance.containsKey(nei) && distance.get(word) == distance.get(nei) + 1) {
    			path.addFirst(word);		//记得是 Word
    			dfsPath(res, path, nei, start, distance, graph);
    			path.pollFirst();
    		}
    	}
    }
    
    
    
    /** Word Ladder II - 用iterative 比较麻烦
     * 
     * we are essentially building a graph, from start, BF.
     * and at each level we find all reachable words from parent.
     * we stop if the current level contains end,
     * we return any path whose last node is end.
     * 
     * 之前I dict.remove(transformed);
     * 但这里不能直接去掉，因为可能别的path需要它，这样result就少了combination
     * 但也要注意在现有path的duplicate问题
     * 
     * to achieve BFS, use a deuqe;
     * a key improvement is to remove all the words we already reached 所以我们需要remove掉这条path的
     * in PREVIOUS LEVEL; we don't need to try visit them again
     * in subsequent level, that is guaranteed to be non-optimal solution.
     * at each new level, we will removeAll() words reached in previous level from dict.
     
     */
    public List<List<String>> findLaddersIterative(String start, String end, Set<String> dict) {
        List<List<String>> results = new ArrayList<List<String>>();
        dict.add(end);
        // instead of storing words we are at, we store the paths.
        Deque<List<String>> paths = new LinkedList<List<String>>();
        List<String> path0 = new LinkedList<String>();
        path0.add(start);
        paths.add(path0);
        
        // if we found a path ending at 'end', we will set lastLevel,
        // use this data to stop iterating further.
        int level = 1, lastLevel = Integer.MAX_VALUE;
        Set<String> wordsPerLevel = new HashSet<String>();
        while (!paths.isEmpty()) {
            List<String> path = paths.pollFirst();
            if (path.size() > level) {
                dict.removeAll(wordsPerLevel);
                wordsPerLevel.clear();
                level = path.size();
                if (level > lastLevel)
                    break; // stop and return
            }
            //  try to find next word to reach, continuing from the path
            String last = path.get(level - 1);
            char[] chars = last.toCharArray();
            for (int index = 0; index < last.length(); index++) {
                char original = chars[index];
                for (char c = 'a'; c <= 'z'; c++) {
                    chars[index] = c;
                    String next = new String(chars);
                    if (dict.contains(next)) {
                        wordsPerLevel.add(next);
                        List<String> nextPath = new LinkedList<String>(path);
                        nextPath.add(next);
                        if (next.equals(end)) {
                            results.add(nextPath);
                            lastLevel = level; // curr level is the last level
                        } else
                            paths.addLast(nextPath);
                    }
                }
                chars[index] = original;
            }
        }
        
        return results;
    }

    
    /**
     * 44. Wildcard Matching (ok)
     * '?' Matches any single character.
	   '*' Matches any sequence of characters (including the empty sequence).
     * s = "adceb"
     * p = "*a*b"
     * Output: true
     * Explanation: The first '*' matches the empty sequence, while the second '*' matches the substring "dce".
     *
     * 这题比regular expression简单。
     *
     * 重点是当p[j]=* 时，要记录jstar这点，以便后面不匹配时能Reset回来。
     * 同时还要track imatched这个点，刚开始是在jstar那里，后面如果有不匹配成功的，j要退回jstar+1, 而imatch要++
     * 这样i从更新后的imatch开始，往后匹配
     * 比如 从imatch这点，开始往后匹配2。到第3个时发现不匹配，那么j退回jstar+1, i也要往回退。
     * 		但由于之前的imatch已经check过了，所以要从imatch后一位(imatch++)开始再跟j比。(因为遇到*可以有任意数）
     * http://shmilyatui'huiotmail-com.iteye.com/blog/2154716
     */    
    public boolean isMatchW(String s, String p) {
        if (p.length() == 0)    return s.length() == 0;
        
        int sl = s.length(), pl = p.length();
        int i = 0, j = 0;
        int jstar = -1, imatched = -1;		//imatched表示上一个*匹配到s里的位置
        while (i < sl) {
            if (j < pl && (s.charAt(i) == p.charAt(j) || p.charAt(j) == '?')) {     //match
                i++;
                j++;
            } else if (j < pl && p.charAt(j) == '*') {
                imatched = i;			//这个可以不动，有可能*可以为空，imatch刚好==jstar+1
                jstar = j;
                j++;				//往后走，这样才开始match后面的
            } else if (jstar >= 0) {    //if previously has * . 或者当j超过len了，就回到star+1
                j = jstar + 1;
                imatched++;
                i = imatched;
            } else {
                return false;
            }
        }
        // 如果s结束了，p有可能后面有很多*
        while (j < pl && p.charAt(j) == '*') {
            j++;
        }
        return j == pl;
    }
    
    
    /**
     * 44. Wildcard Matching
     * @param s
     * 当 cur j = '*', dp为TRUE if 
     * 	  (1) dp[i-1][j] 使用这个 *。 pre i & *  If '*' means anything, s[0..i-2] must match p[0..j-1].
     * 	  (2) dp[i][j-1] cur i & pre j, 所以* 没用上 If '*' means empty, s[0..i-1] must match p[0..j-2].
     */
    public boolean isMatchWdp(String s, String p) {
        if (p.length() == 0)    return s.length() == 0;
        
        int sl = s.length(), pl = p.length();
        boolean[][] dp = new boolean[sl+1][pl+1];
        dp[0][0] = true;

        for (int j = 1; j <= pl; j++) {
            if (p.charAt(j-1) == '*') {
                dp[0][j] = dp[0][j-1];
            } else {
            	break;		//只要有不符合，就肯定FALSE，提前break
            }
        }
        
        for (int i = 1; i <= sl; i++) {
            for (int j = 1; j <= pl; j++) {
                if (s.charAt(i-1) == p.charAt(j-1) || p.charAt(j-1) == '?') {
                    dp[i][j] = dp[i-1][j-1];
                } else if (p.charAt(j-1) == '*') {
                    dp[i][j] = dp[i-1][j] || dp[i][j-1];
                 //  * used to match pre i || * is not used    
                }
            }
        }
        return dp[sl][pl];
    }
    
    
    // wild card
    public boolean isMatchWildCard(String s, String p) {
           int[][] cache = new int[s.length()][p.length()];    // 0 not visited, 1 false, 2 true
           return dfs(s, p, 0, 0, cache);
       }
       
    public boolean dfs(String s, String p, int i, int j, int[][] cache) {
       if (i == s.length() && j == p.length())
           return true;

        if (i == s.length() && p.charAt(j) == '*')
           return dfs(s, p, i, j + 1, cache);

       if (i == s.length() || j == p.length())
           return false;

       if (cache[i][j] > 0)
           return cache[i][j] == 2;

       if (s.charAt(i) == p.charAt(j) || p.charAt(j) == '?') {
           boolean match = dfs(s, p, i + 1, j + 1, cache);
           cache[i][j] = match ? 2 : 1;
       } else if (p.charAt(j) == '*') {		 // * 用或不用
           boolean match = dfs(s, p, i + 1, j, cache) || dfs(s, p, i, j + 1, cache);
           cache[i][j] = match ? 2 : 1;
       }
       return cache[i][j] == 2;
    }


    /**
     * 10. Regular Expression Matching
     *
     * 比上面难
     *
     * '.' Matches any single character.
     * '*' Matches zero or more of the preceding element. ‘a*’代表能出现0次或多次a 。。*前面要有数..
     * The matching should cover the entire input string (not partial).
     * s = "aa"
     * p = "a*"
     * Output: true
     *
     * 正则匹配，p为正则表达式，s为要匹配字符串，‘.’代表任意字符，‘a*’代表能出现0次或多次a，要返回是否能整体匹配而不是部分匹配。
     1）p 0位要考虑的情况有点多
     2）p 1位 ,  考虑是否相等或者 '.'.
     3）p的第二位不为‘*’，则s位数要大于1，s第一位==p第一位或p第一位为‘.’，之后继续比较后面。因为p(0)不能有'*' 所以就判断p(1)是否*
     4）p第二位为‘*’，则考虑几种情况：
     a）s= abbbbbc ,p= ab*c
     b）s=abcabcab ,p= a.*c.*b等
     都要迭代比较

     后面用DP更快
     */
    public boolean isMatch(String s, String p) {
        if (p.length() == 0) {
            return s.length() == 0;
        }

        if (p.length() == 1) {	//只剩1位，s也要1位
            return s.length() == 1 && (p.charAt(0) == s.charAt(0) || p.charAt(0) == '.');
        }

        // len >= 2 , 第二位不为*
        if (p.charAt(1) != '*') {
            return s.length() > 0 && (p.charAt(0) == s.charAt(0) || p.charAt(0) == '.')
                       && isMatch(s.substring(1), p.substring(1));	//若第一位匹配，就往后挪一位
        }
        // if p.charAt(1) == '*'第二位是*。  若第一位相等，那要while loop比后面的 aaa & a*a 看看能到多长
        while (s.length() > 0 && (p.charAt(0) == '.' || p.charAt(0) == s.charAt(0))) {     // a & a*
            if (isMatch(s, p.substring(2))) {
                return true;
            }
            s = s.substring(1);
        }

        return isMatch(s, p.substring(2));
    }


    /**
     * 10. Regular Expression Matching
     * 因为前后字符都是有联系的,所以想到用DP
     我们可以用一个boolean二维数组来表示两个字符串前几个字符之间的联系，
     * @param s
     * @param p
     * @return
     * dp[i][j]表示s的前i个字符 与 p的前j个字符组成的 两个字符串是否匹配  s[0~i],p[0~j] if match
     * 3. 字符为'*'
    这种比较复杂，因为'*'能匹配任意个数的前面的字符。
    所以我们可以先考虑能够匹配0个和1个前面字符的简单情况，比如

    匹配1个表示 a & a*  .dp[i][j] = dp[i][j - 1];
    匹配0个表示可以忽略自己及前面那个字符，a & b* 可得	dp[i][j] = dp[i][j - 2];

    再考虑匹配2个及2个以上前面字符的情况，这种情况可以这么考虑:
    如果dp[i][j] = true是'*'匹配k(k>=2)个前面字符的结果，那么'*'匹配k-1个前面字符的结果也必须是true，所以条件之一便是dp[i - 1][j] == true，
    另外一个条件便是s的最后一个字符必须匹配p的'*'前面的字符，所以得到 abbbbc & ab*cs
    dp[i][j] = dp[i -1][j] && (p.charAt(j - 2) == s.charAt(i - 1) || p.charAt(j - 2) == '.'
     */
    public boolean isMatchDP(String s, String p) {
        int sl = s.length();
        int pl = p.length();
        boolean[][] dp = new boolean[sl + 1][pl + 1];   //s[0~i],p[0~j] if match
        dp[0][0] = true;
        // by default, dp[i][0] is false
        for (int j = 1; j <= pl; j++) {
            if (p.charAt(j-1) == '*' && dp[0][j-2]) {
                dp[0][j] = true;
            }
        }

        for (int i = 1; i <= sl; i++) {
            for (int j = 1; j <= pl; j++) {
                if (p.charAt(j-1) == '.' || p.charAt(j-1) == s.charAt(i-1)) {
                    dp[i][j] = dp[i-1][j-1];
                } else if (p.charAt(j-1) == '*') {
                    //repeat 0 (a & b*) || 1 time  a & a*
                    if (dp[i][j-2] || dp[i][j-1]) {
                        dp[i][j] = true;
                    } else if (p.charAt(j-2) == '.' || p.charAt(j-2) == s.charAt(i-1)) {
                        dp[i][j] = dp[i-1][j];		//repeat >= 2 times  aaaa  a*. 最后的a匹配上a*, 所以倒数第二个a跟*再比
                    }
                }
            }
        }
        return dp[sl][pl];
    }



    /** 447. Number of Boomerangs
     * 给串数组，找出i, 使得 i~j距离 == i~k距离
     * [[0,0],[1,0],[2,0]]，结果是2. 因为[[1,0],[0,0],[2,0]] and [[1,0],[2,0],[0,0]] 可以有不同顺序
     * @param points
     * @return
     * 这也是用hashmap来存distance, 和距离相等的点的个数freq(count).
     * 1. 两层for循环, i, j < len. 不能 len,因为跟所有点（除了自己）对比
     * 2. 把距离放map里，距离相等的话freq++. 而且count需要 + 2 * freq.因为顺序可以不一样
     * 3. 记得每次i 要clear map
     * 
     * 或者每次只在map里+1. j for完以后，for(map), count += freq * (freq - 1). 是2个数的排列组合
     * 
     * 这里的距离直接 x*x + y*y. 不需要开根号，方便且不容易出错
     */
    public int numberOfBoomerangs(int[][] points) {
        if (points == null || points.length < 3)    return 0;
        
        int count = 0;
        int len = points.length;
        HashMap<Integer, Integer> map = new HashMap<>();    // <distance, freq>
        
        for (int i = 0; i < len; i++) {
            map.clear();        //remember to clear map every i

            for (int j = 0; j < len; j++) {
                if (i == j) continue;
                int dist = distance(points[i], points[j]);
                if (map.containsKey(dist)) {
                    int freq = map.get(dist);
                    count += 2 * freq;     // cause we can change order of j,k
                    map.put(dist, freq + 1);        //记得放回去+1，可能还有第三个match的
                } else {
                    map.put(dist, 1);
                }
            }
        }
        return count;
    }
    
    public int distance(int[] p, int[] q) {
        int x = p[0] - q[0];
        int y = p[1] - q[1];
        return x * x + y * y;       //no need to sqrt
    }
    
    
    /**
     * 356. Line Reflection
     * 给一堆点，看是否存在一个对称轴，使这些点都对称...  PS: 有重复点
     * @param points
     * @return
     * 刚开始想到是先sort一下x轴，找max和min然后除2. 但是sort会有问题，如果x相等，怎么sort Y呢？而且有重复点怎么办
     * 其实for一遍找minX & minY就行了，不用sort
     * 
     * 基于上面的考虑，采用hashset, 这样能去掉重复的。而且即使x相等y不同，也可以根据set里的对比。并且O(n)快
     * 1. 第一遍for，把point放在set里，顺便记录min x, max x.
     * 		一个trick是 把point变成string , x + "," + y 这样，方便对比
     * 		可以用hashCode变成integer，Arrays.hashCode(new int[]{sum - p[0], p[1]});。但很慢
     * 2. 得出sum = min + max
     * 3. 再for一次，看每个点的对称点 sum-x 是否存在set里
     * 
     * Google有个followup, 若能允许k个不对称点，怎么办？
     * 那就只能O(n^2)的找所有可能的对称轴，然后找完对称轴再for一遍所有点看是否超过k O(n). 总共加起来是O(N^3)
     * 找对称轴时，要头尾往中间扫，这样更快找到..否则如果都从头开始j=i+1 那找到可能性很低，h非常慢
     * 
     * **********************************************
     * 其实上面sort X轴也是可以做，nlogn, 不用额外空间
     * a. 有重复点就跳过..if (i > 0 && points[i-1][0] == points[i][0] && points[i-1][1] == points[i][1]) {i++; }
     * b. 如果x相等，y怎么sort？
     * 		那就先for循环找下min, max得出mid.. 然后x <= mid就先排矮的，反之x在mid后面，那就先排高的
     */
    public boolean isReflected(int[][] points) {
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        Set<String> set = new HashSet<>();      //prevent duplicate
        
        // put all points in set && find min, max x
        for (int[] p : points) {
            max = Math.max(max, p[0]);
            min = Math.min(min, p[0]);
            String s = p[0] + "," + p[1];
            set.add(s);
        }

        // 其实不管在mid左还是右，p[0] < mid ? mid + mid - p[0] : mid - (p[0] - mid) 所以只需要算sum即可
        int sum = max + min;

        // 再扫一遍看看 x的对称点(sum-p[0]) 有没在set里
        for (int[] p : points) {
            String s = (sum - p[0]) + "," + p[1];
            if (!set.contains(s)) {
                return false;
            }
        }
        return true;
    }
    
    //不确定对不对
    public boolean isReflectedExceptOne(int[][] points) {
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        Set<String> set = new HashSet<>();      //prevent duplicate
        
        // put all points in set && find min, max x
        for (int[] p : points) {
            max = Math.max(max, p[0]);
            min = Math.min(min, p[0]);
            String s = p[0] + "," + p[1];
            set.add(s);
        }
        
        int sum = max + min;
        List<int[]> bad = new ArrayList<>();
     // 有可能x是max, min. 但高不同，所以max, min还是保留。 这时看看有多少个不一样的，如果只有一个，说明max没变
    	//也有可能需要改max, min
        for (int[] p : points) {
            String s = (sum - p[0]) + "," + p[1];
            if (!set.contains(s)) {
                bad.add(p);
                if (bad.size() > 1) {	//超过2个点不对，那就是max, min要变
                	break;
                }
            }
        }
        
        if (bad.size() > 1) {
        	for (int i = 0; i < 2; i++) {
        		int tmp = bad.get(i)[0];
        		if (tmp == max) {		//need to change max
        	//		max -= 1;    不能-1，不一定有这个数
        			for (int[] p : points) {
        	            max = Math.max(max, p[0]);
        	            min = Math.min(min, p[0]);
        	        }
        		} else if (tmp == min) {
        			min += 1;
        			for (int[] p : points) {
        	            max = Math.max(max, p[0]);
        	            min = Math.min(min, p[0]);
        	        }
        		}
        	}
        	sum = min + max;
        	for (int[] p : points) {
                String s = (sum - p[0]) + "," + p[1];
                if (!set.contains(s)) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    

    
    /**
     * 149. Max Points on a Line - better
     *
     * O(n^2) + hashmap
     * 记得每次新的一轮i时，要map.clear().. 且注意same point的情况
     *
     * 有的Corner case的double会失去精度，所以y/x会算不准斜率
     * 用最大公约数来算
     * 让这两数分别除以它们的最大公约数，这样例如8和4，4和2，2和1，
     *
     * 用string代表slope.. 不需要考虑vertical
     */
    public int maxPoints1(Point[] points) {
        int maxRes = 0;
        Map<String, Integer> map = new HashMap<>();
        
        for (int i = 0; i < points.length; i++) {
            map.clear();			//这里也记得clear
            int maxOnLine = 0;		//这里是0, 指这个斜率的线上有多少个点
            int same = 0;
            for (int j = i + 1; j < points.length; j++) {
                int dx = points[j].x - points[i].x;
                int dy = points[j].y - points[i].y;
                
                if (dx == 0 && dy == 0) {
                    same++;
                } else {
                    int gcd = getGCD(dx, dy);
                    dx /= gcd;
                    dy /= gcd;

                    String slope = dy + "/" + dx;
                    map.put(slope, map.getOrDefault(slope, 0) + 1);

                    maxOnLine = Math.max(maxOnLine, map.get(slope));
                }   
            }
            maxRes = Math.max(maxRes, maxOnLine + same + 1);		//这要+1，算上i自己
        }
        return maxRes;
    }
    
    private int getGCD(int a, int b) {
        return b == 0 ? a : getGCD(b, a % b);
    }



    /**
     * 149. Max Points on a Line
     * 记得主要相同点的情况，还有double转精度，以及除数为0的情况
     * @param points
     * @return
     * O(n^2) + hashmap
     * 重点是有Corner case：
     * 		a. 有相同点的话。 那就用samePoint记录，之后要加上考虑
     * 		b. 有垂直的Y轴。那就用vertical表示。到时 vertical和 map里的max斜率比较
     *
     * 	  有可能斜率存在MAX_VALUE和MIN.所以分别用int vertical变量表示会比较好。
     * 另外记得double转精度
     *
     * 或者用string 表示斜率, 但要先算出GCD...
     *
     * 后面的比较好.. 看那个
     */
    public int maxPoints(Point[] points) {
        int max = 0;
        if (points.length == 1)     return 1;
        Map<Double, Integer> map = new HashMap<>();

        for (int i = 0; i < points.length; i++) {
            map.clear();
            int samePoint = 0;
            int vertical = 1;	//初始也是1
            int count = 1;      //numbers on same line for this i，所以是1
            for (int j = i + 1; j < points.length; j++) {
                int dx = points[j].x - points[i].x;
                int dy = points[j].y - points[i].y;

                if (dx == 0) {
                    if (dy == 0) {
                        samePoint++;
                    } else {
                        vertical++;
                    }
                } else {
                    // 或者用string代表slope
                    /*
                    String sign = "+";
                    if (dx * dy < 0) {
                        sign = "-";
                        dx = Math.abs(dx);
                        dy = Math.abs(dy);
                    }

                    String slope = sign + dy + "/" + dx;

                    */
                    double slope = (double) dy / (double) dx + 0.0;
                    if (map.containsKey(slope)) {
                        map.put(slope, map.get(slope) + 1);
                    } else {
                        map.put(slope, 2);
                    }
                    count = Math.max(count, map.get(slope));	//找出最多一样斜率的count。记得比较



                }
            }
            // for this i round, 看是否正常有斜率的点max.map(slope) > vertical points垂直的
            count = samePoint + Math.max(count, vertical);
            max = Math.max(max, count);
        }
        return max;
    }


    class Point {
	    int x;
	    int y;
	    Point() { x = 0; y = 0; }
	    Point(int a, int b) { x = a; y = b; }
	}
    
    
    
    
    
    /**
     * 418. Sentence Screen Fitting
     * 给rows x cols screen & a sentence represented by a list of words, 看这句话能fit重复多少次
     * 每个单词中间要有空格，最后可以不需要
     *
     * rows = 3, cols = 6, sentence = ["a", "bcd", "e"]  结果 2
     *
     * Explanation:
     * a-bcd-
     * e-a---
     * bcd-e-
     * 
     * naive方法，for每行，然后看这行的remain能否填满
     * 用k代表第k个单词，直到 k==sentence.len才count++.
     * 前面 remain / allLen 或者取%，为了解决sentence很短，空间大的情况
     * 
     * 但也要注意几点
     * a. 先把句子都加空格，这样比较好算
     * b. remain - word时，还要 -1 因为有空格
     */
    public int wordsTyping(String[] sentence, int rows, int cols) {
        if (sentence == null || sentence.length == 0)  return 0;
        
        int len = sentence.length;
        String all = "";

        for (String w : sentence) {
            all += w + " ";
        }

        int allLen = all.length();
        int count = 0;
        int k = 0;

        for (int i = 0; i < rows; i++) {
            int remain = cols;			
            count += remain / allLen;           // 如果cols长于所有词allLen
            remain %= allLen;
            
         // 剩下的remain就一个个填单词
            while (remain >= sentence[k].length()) {		// 记得 >= 
                remain -= (sentence[k].length() + 1);
                k++;
                if (k == len) {			//填满了
                    count++;
                    k = 0;
                }
            }
        }
        return count;
    }
    
    
    /**
     * 418. Sentence Screen Fitting
     * 把下面的行都放上来拉成一行.. COUNT代表总共能放多少个单词的总长度
     * 
     */
    public int wordsTypingBetter(String[] sentence, int rows, int cols) {
        String all = "";
        for (String s : sentence) {
            all += s + " ";
        }
        int allLen = all.length();
        int count = 0;						// 总共能放多少个单词的总长度

        for (int i = 0; i < rows; i++) {
            count += cols;
            if (all.charAt(count % allLen) == ' ') {		//相当于rows摊成一行，看能否fit.
                count++;        
            } else {
                while (count > 0 && all.charAt((count-1) % allLen) != ' ') {
                    count--;
                }
            }
        }
        
        // ========或者========用cache存一下========更快===========
        int[] map = new int[allLen];
        // 吧上面的while那里单独抽出来 缓存
        for (int i = 1; i < allLen; i++) {      //从1开始
            map[i] = all.charAt(i) == ' ' ? 1 : map[i - 1] - 1;
        }
        
        for (int i = 0; i < rows; i++) {
            count += cols;
            count += map[count % allLen];
        }
        
        return count / allLen;
    }
    
    
    /**
     * 418. Sentence Screen Fitting
     * @param sentence
     * @param rows
     * @param cols
     * @return
     * 1. 先扫sentence[], 用nextIdx[]记录每个index能fit的下一个index，顺便算count
     * 2. while里面跟上面的解法差不多（好理解）
     * 3. 再扫一遍rows，看 ans += counts[], 下一个数就是nextIdx[k]
     */
    public int wordsTypingFaster(String[] sentence, int rows, int cols) {
        if (sentence == null || sentence.length == 0)  return 0;
        
        int n = sentence.length;
        int[] nextIdx = new int[n];
        int[] counts = new int[n];
        for (int i = 0; i < sentence.length; i++) {
            int curLen = 0;
            int k = i;
            int count = 0;
            while (curLen + sentence[k].length() <= cols) {
                curLen += sentence[k].length() + 1;
                k++;
                if (k == n) {
                    count++;
                    k = 0;
                }
            }
            nextIdx[i] = k;
            counts[i] = count;
        }
        
        int result = 0;
        int k = 0;
        for (int i = 0; i < rows; i++) {
            result += counts[k];
            k = nextIdx[k];         //go to next index
        }
        return result;
    }
    
    
    
    /**
     * KMP 模式匹配  - text中找到pattern匹配并出现的位置
     * @param text
     * @param pattern
     * @return
     * 最重要是保存 相同的 前缀&后缀 的longest长度。
     * 因为前缀后缀相同的话，就代表这段是一样的，可以直接从前缀后一位比较，避免重复比较
     * 注意prefix是从0开始的
     * 
     * A: 建longestPrefixSuffix的数组
     * 1. j从0开始，走得慢，代表prefix。i从1开始，走得快，代表suffix
     * 2. 当相等时，一起++往后走，并且 lps[i]=j+1 因为prefix == suffix，长度是 j+1，因为j从0开始
     * 3. 不等时，j = lps[j-1],看前一个字符的最长suffix是多少，再跟i比。如果j==0,那就lps[i]=0, i++
     *    
     * B. KMP开始匹配
     * 1. 都从0开始
     * 2. 相等时，都++往后走
     * 3. 不等时，j = lps[j-1]。j就看前一个字符的最长前缀
     * a b a b a b c & a b a b c.  
     * 0 1 2 3 4 5 6   0 1 2 3 4
     * 当到4时，t[4]=a, p[4]=c, 不相等，这时j=lps[j-1] 就是从第一个ab后的字符比，也就是2个a
     * j=lps[j-1]是之前的最长相同suffix
     */
    public int KMP(char[] text, char[] pattern) {
    	int lps[] = longestPrefixSuffix(pattern);		//短的那个
    	int i = 0, j = 0;
    	while(i < text.length && j < pattern.length){
    		if (text[i] == pattern[j]) {
    			i++;
    			j++;
    		} else {
    			if (j != 0) {
    				j = lps[j-1];	//看它前一个最长前后缀的地方. 并且lps[j-1]代表长度，且之前+1过，所以是算匹配后的一个字符
    			} else {
    				i++;
    			}
    		}
    	}
    	if (j == pattern.length) {
    		return i - pattern.length;
    	}
    	return -1;
    }
    
    /** 求 当前缀==后缀时，最长的长度. 
     * @param dest
     * @return
     * 1. j从0开始，走得慢，代表prefix。i从1开始，走得快，代表suffix
     * 2. 当相等时，一起++往后走，并且 lps[i]=j+1 因为prefix == suffix，长度是 j+1，因为j从0开始
     * 3. 不等时，j = lps[j-1],看前一个字符的最长suffix是多少，再跟i比。如果j==0,那就lps[i]=0, i++
     * 
     * 比如这个string "abab", 算完以后数组变成. lps[i]是 以i为suffix结尾，当prefix==suffix的最长 长度
     * 		  int[]  [0012]
     */
    public int[] longestPrefixSuffix(char[] dest) {
    	int[] lps = new int[dest.length];
    	int i = 1, j = 0;
    	while (i < dest.length) {		//i 跑得比j快
    		if (dest[j] == dest[i]) {
    			lps[i] = j + 1;			// j跑得慢，代表从0开始组成的最长prefix
    			j++;
    			i++;
    		} else {
    			if (j != 0) {		//看 lps[j-1]里的j在什么地方，代表之前匹配的最长前缀后缀
    				j = lps[j-1];		//那么j就跳回之前匹配的地方lps[j-1]
    			} else {
    				lps[i] = 0;		//记得设为0
    				i++;
    			}
    		}
    	}
    	return lps;
    }


    /**
     * 408. Valid Word Abbreviation - easy
     * s = "internationalization", abbr = "i12iz4n": 对的。 s = "apple", abbr = "a2e":错的。 abbr="a03e"也错，不能0开头
     * @param word
     * @param abbr
     * @return
     */
    public boolean validWordAbbreviation(String word, String abbr) {
        if (word.length() == 0 || word.length() < abbr.length())  return false;
        int i = 0, j = 0;
        while (i < word.length() && j < abbr.length()) {
            if (word.charAt(i) == abbr.charAt(j)) {
                i++;
                j++;
            } else {
                if (Character.isDigit(abbr.charAt(j))) {
                    if (abbr.charAt(j) == '0')		//不能0开头或者只有0
                        return false;
                    int count = 0;
                    while (j < abbr.length() && Character.isDigit(abbr.charAt(j))) {
                        count = count * 10 + abbr.charAt(j) - '0';
                        j++;
                    }
                    i += count;
                } else {
                    return false;
                }
            }
        }
        return i == word.length() && j == abbr.length();    // 确保数字结尾时 是对的
    }


    /**
     * 288. Unique Word Abbreviation
     * 缩写成收尾字母，中间数字  it (没缩写), dog->d1g , localization->l10n
     * A word's abbreviation is unique if *no other word* from the dictionary has the same abbreviation.
     * dictionary里可以有 多个相同的词，可以跟要查的word一样，这些都是TRUE。但如果有别的相同abbr的话，就FALSE
     * @param dictionary
     * 简单题..  
     * 重点是放map里时，key是abbr，val是那个单词。如果单词一样，就不用管（可能字典有一样的词）
     * 不一样，证明有其他一样缩写的，那就不行，置""即可
     */
    public void ValidWordAbbr(String[] dictionary) {
    	Map<String, String> map = new HashMap<>();      // <abbr, word>

        for (String w : dictionary) {
            String abbr = genAbbr(w);
            if (map.containsKey(abbr)) {
                if (!map.get(abbr).equals(w)) {     //if different, need to update val. otherwise no change
                    map.put(abbr, "");
                }
            } else {
                map.put(abbr, w);
            }
        }
    }

    public boolean isUnique(Map<String, String> map, String word) {
        String abbr = genAbbr(word);
        // 不在字典里也可以
        return !map.containsKey(abbr) || map.get(abbr).equals(word);
    }
    
    private String genAbbr(String s) {
        int len = s.length();
        if (len <= 2)   return s;
        return s.charAt(0) + Integer.toString(len - 2) + s.charAt(len - 1);
    }
    

    
    /**
     * 320. Generalized Abbreviation
     * 组成所有缩写可能性，包括 word, 4, "1ord", "w1rd","3d", "w3"等
     * @param word
     * @return
     * 对于每个character，有两种情况，缩写或不缩写，也就是
     *   a. 变成int -> pos + 1, str, count + 1
     *   b. 保留char -> pos + 1, str + count + word.charAt(pos), 0。。保留char还要看之前的count是否>0,是的话要加count
     * dfs时要保存count长度 + 新的str. 
     * 如果保留char, 记得把count重设为0
     * 记得count==0时就不要加
     */
    public List<String> generateAbbreviations(String word) {
        List<String> list = new ArrayList<>();
        abbrHelper(list, word, 0, "", 0);
        return list;
    }
    
    public void abbrHelper(List<String> list, String word, int pos, String str, int count) {
        if (pos == word.length()) {
            if (count > 0) {		// > 0时才要加
                str += count;
            }
            list.add(str);
            return;
        }
        
        //abbreviate to number
        abbrHelper(list, word, pos + 1, str, count + 1);   
        
        //keep as char, need to reset count to 0
        abbrHelper(list, word, pos + 1, str + (count > 0 ? count : "") + word.charAt(pos), 0);

//        if (count > 0) {
//            abbrHelper(list, word, pos + 1, str + count + word.charAt(pos), 0);    //keep as char, so count need to reset 0
//        } else {
//            abbrHelper(list, word, pos + 1, str + word.charAt(pos), 0);
//        }
    }


    /**
     * 527. Word Abbreviation
     * 这list里的每个数generate 最短的abbreviation.. 要避免冲突.. 如果abbr跟原词一样长那就保留original
     * abbr的话首尾要是letter..中间可letter/num比如 inte4n
     *
     * Input: ["like", "god", "internal", "me", "internet", "interval", "intension", "face", "intrusion"]
     * Output: ["l2e","god","internal","me","i6t","interval","inte4n","f2e","intr4n"]
     *
     * 比较直接的方法 （慢）
     *
     * 规定：abbr的话首尾要是letter..中间可letter/num比如 inte4n
     * 推出：prefix length增加的话，比如前面都是letter，看abbr是否还会duplicate
     *
     * 一个prefix[]数组专门放 每个word需要多长的prefix letter.. 默认1开始。这样刚开始的result[i] 都是prefix为1的abbr
     *
     * for整个dict.. 对于每个abbr, 再for后面的j, 看有没跟i自己的abbr一样，有的话就加到duplicateAbbr (每轮都会新的)。
     * 如果duplicateAbbr有东西，说明有duplicate的abbr, 需要对这几个duplicate重新generate abbr, 把prefixLen加长看能否OK
     *
     * 时间复杂度: O(N^2 * L^2) N是多少单词dict size, L是平均词长.
     * 里面的while() 最多可以循环 L 次，因为最多找单词长度 次。然后里面for(j)又是N..
     *
     * 还有 Trie或者别的快点的解法.. 看LeetCode
     */
    public List<String> wordsAbbreviation(List<String> dict) {
        int n = dict.size();
        String[] result = new String[n];
        int[] prefix = new int[n];          // 存每个词需要多少prefix长度才能valid abbr

        for (int i = 0; i < n; i++) {
            prefix[i] = 1;
            result[i] = genAbbr(dict.get(i), 1);
        }

        for (int i = 0; i < n; i++) {
            while (true) {
                Set<Integer> duplicateAbbr = new HashSet<>();   // 记录duplicate abbr的index

                for (int j = i + 1; j < n; j++) {
                    if (result[i].equals(result[j])) {
                        duplicateAbbr.add(j);
                    }
                }
                if (duplicateAbbr.isEmpty())        // 都不一样，返回结果
                    break;

                duplicateAbbr.add(i);
                for (int k : duplicateAbbr) {
                    result[k] = genAbbr(dict.get(k), ++prefix[k]);      // ++ prefix[k].之前prefix太短了，现在要加长abbr看之后是否还重复
                }
            }
        }
        return Arrays.asList(result);
    }

    private String genAbbr(String word, int prefixLen) {
        int len = word.length();
        if (prefixLen >= len - 2)       // 没必要abbr
            return word;

        return word.substring(0, prefixLen) + (len - prefixLen - 1) + word.charAt(len - 1);
    }
    
    
    /**
     * 411. Minimum Unique Word Abbreviation
     * 给target和一串dictionary, 可能会有缩写abbr跟别的单词重复的.. 找出最短的不一样的abbr
     * "apple", ["blade"] -> "a4" (because "5" or "4e" conflicts with "blade")
     *
     * "apple", ["plain", "amber", "blade"] -> "1p3" (other valid answers include "ap3", "a3e", "2p2", "3le", "3l1").
     *
     * 利用上面的题..NAIVE.. 然后TLE了
     * 1. 要找出target的所有abbr （不是找字典的否则太多了）
     * 2. 对这些abbrs从短到长排列
     * 3. for(abbrs), 里面对dictionary的每一个字看看是否能组成一样的abbr..
     * 
     * 注意先扫一遍，去掉长度不同的单词.. 因为长度不同abbr肯定也不同，不需要比
     */
    public String minAbbreviation(String target, String[] dictionary) {
        if (dictionary == null || dictionary.length == 0)   return String.valueOf(target.length());
        
        List<String> abbrs = new ArrayList<String>(); 
        abbrHelper(abbrs, target, 0, "", 0);		//用上题的Generalized Abbreviation
        
        Collections.sort(abbrs, new Comparator<String>() {		//把abbr从短到长排
            public int compare(String s1, String s2) {
                return s1.length() - s2.length();
            }
        });
        
        // 只加相同长度的单词.. 不同长度肯定不同abbr
        List<String> dict = new ArrayList<>();      //remove str that diff lengh
        for (String w : dictionary) {
            if (target.length() == w.length()) {
                dict.add(w);
            }
        }
        if (dict.size() == 0)   return String.valueOf(target.length());
        
        for (String abbr : abbrs) {
            int i = 0;
            for (; i < dict.size(); i++) {
                if (validWordAbbreviation(dict.get(i), abbr)) 		//用上题Valid Word Abbreviation，如果可以一样的abbr,那就不行
                    break;
            }
            if (i == dict.size())
                return abbr;
        }
        
        return target;
    }
    
    
    
    private int minLen;
    private int result;

    public String minAbbreviation2(String target, String[] dictionary) {
        // only keep words whose length == target in new dict, then compute their bit masks
        Set<Integer> maskSet = new HashSet<>();
        for(String s: dictionary){
            if(target.length() == s.length()){
                maskSet.add(getBitMask(s,target));
            }
        }

        // dfs with pruning
        minLen = target.length()+1;
        result = -1;

        dfs(target,maskSet,0,0,0);

        if(minLen > target.length()){
            return "";
        }

        // convert result to word
        int zeroCnt = 0;
        String res = "";
        for (int i = target.length()-1; i>=0; i--) {
            //遇到0要累加连续零个数,遇到1填原char
            int digit = (result & 1);
            if(digit == 0){
                ++zeroCnt;
            } else {
                if(zeroCnt > 0){
                    res = zeroCnt + res;
                    zeroCnt =0;
                }
                res = target.charAt(i) + res;
            }
            result >>= 1;
        }
        if(zeroCnt > 0) res = zeroCnt + res;
        return res;
    }

    private void dfs(String target,Set<Integer> maskSet,int start,int curLen,int curResult){
        // pruning, no need to continue, already not min length
        if(curLen >= minLen) return;

        if(start == target.length()){
            // check whether curResult mask conflicts with words in dict
            for(int mask:maskSet){
                /**
                 * 单词manipulation的缩写m2ip6n可以转化为100110000001
                 *  m a n i p u l a t i o n
                    m  2  i p      6      n
                    1 0 0 1 1 0 0 0 0 0 0 1
                 * 0代表随意不care,如果这个mask和dict中某个mask的所有1重合代表在意的位置完全相同,
                 * 说明这个mask和dict中那个词冲突
                 * 我们要找的是不冲突的mask
                 */
                if((curResult & mask) == curResult){
                    return; // conflict
                }
            }
            // no conflict happens, can use
            if(minLen > curLen){
                minLen = curLen;
                result = curResult;
            }
            return;
        }

        // case 1: replace chars from start in target with number
        for (int i = start; i < target.length(); i++) {
            //被replace掉的char位置由0代替所以是curResult<<(i+1-start),没replace掉的这里不管,我们只管到i,之后的由backtrack内决定
            //注意:不允许word => w11d这种用数字代替但含义不同
            if(curLen == 0 || (curResult &1) == 1){
                //后者即上一次是保留了字母
                dfs(target,maskSet,i+1,curLen+1,curResult<<(i+1-start));
            }
        }

        // case 2: no replace from start (curResult << 1)+1代表新的这位保留了char,所以是加一
        dfs(target,maskSet,start+1,curLen+1,(curResult << 1)+1);
    }

    // 比如apple 和 amper 字母相同设1,不同设0,所以得到10100
    private int getBitMask(String s1,String s2){
        int mask = 0;
        for (int i = 0; i < s1.length(); i++) {
            mask <<= 1;
            if(s1.charAt(i) == s2.charAt(i)){
                mask += 1;
            }
        }
        return mask;
    }
    
    
    
    
    /**
     * 32. Longest Valid Parentheses
	 * ")()())",  longest valid parentheses substring is "()()", 返回 4.
	 * @param s
	 * @return
     *
	 * 用stack存index!!!
     * '(' 或者 empty时 push进去.
     * 如果empty不push进去，后面算 i-peek 时会出错
     *
	 * 技巧是：一看到 ')' 就先pop。完了之后，如果empty了再push现在的index ， else 算index substring相差值   max=(max, idx - peek)
	 */
	public int longestValidParentheses(String s) {
        Stack<Integer> stack = new Stack<Integer>();
        int max = 0;
        stack.push(-1);		//如果第一对match 那i=1, max=1-(-1) = 2
        
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {        //match ')'
                stack.pop();             // 先pop

                if (stack.isEmpty()) {      // no more '(' in stack, push current index
                    stack.push(i);          // 这样后面可以 idx - s.peek 算range。否则empty.peek()会报错
                } else {
                    max = Math.max(max, i - stack.peek());
                }
            }
        }
        
        return max;
    }
	
	
	/**
     * 32. Longest Valid Parentheses
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
            } else if (right > left) {          // 不匹配
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
            } else if (left > right) {          // 不匹配
                left = right = 0;
            }
        }
        return max;
    }
	
	
	/**
     * 32. Longest Valid Parentheses
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
                    if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                        int pre = i - dp[i-1] - 2 >= 0 ? dp[i - dp[i-1] - 2] : 0;
                        dp[i] = dp[i-1] + 2 + pre;
                        max = Math.max(max, dp[i]); //update max
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
                dp[len] = Math.max(dp[len], dp[i]); //update max
            }
        }
        
        return dp[len];
	}
	
	
	/**
     * 301. Remove Invalid Parentheses
	 * 删掉invalid的，但是删的个数是minimum
	 * ["(a)())()" -> ["(a)()()", "(a())()"]
	 * @param s
	 * @return
	 * 基本思想：找到invalid的，for循环在上次validPos ~ 这次invalid pos(i) 里，remove多余的i这个括号，然后再dfs剩下的..
	 * 
	 * 1. 怎么找invalid？用count记录，left的话++，right的话--。如果count < 0那就是invalid
	 * 
	 * 2. invalid的地方在 i，那么从上一次valid后的地方开始扫，也就是 j从[validPos, i]扫
	 * 		2.1 如果多出的')'. 那么remove时要注意，如果连续的")))",那只能删一个，不能重复
	 * 			所以可以删第一个，后面都跳过..
	 * 		2.2 删完后，dfs(newStr, i, j).. 记得更新i, j位置
	 * 
	 * 3. for完 全部都valid的情况 (count > 0) 之后，再加result
	 * 
	 * 4. 注意，除了从左->右扫，还需要 从右往前扫，跟Longest Valid Parentheses的two pass一样. 可能 ((). 这时是ok的，但是left多，所以会漏掉这情况
     *
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
            else if (s.charAt(i) == par[1])    count--;
            
            // check if has invalid
            if (count < 0) {            // 从上一次的 validPos 开始 到有问题的 i
                for (int j = validPos; j <= i; j++) {
                    if (s.charAt(j) == par[1]) {		//避免重复结果，第一次出现多余的 par[1]
                        if (j == validPos || s.charAt(j - 1) != par[1]) {
                            // 删掉这个invalid j
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


    // BFS 慢 -  试着一层层 delete某位，看是否valid
    public List<String> removeInvalidParenthesesBFS(String s) {
        List<String> result = new ArrayList<>();
        Set<String> used = new HashSet<>();         // all possible strings
        Queue<String> q = new LinkedList<>();

        q.add(s);
        used.add(s);
        boolean found = false;      // after we found, then don't go to next level in the q

        while (!q.isEmpty()) {
            String str = q.poll();

            if (isValid(str)) {
                result.add(str);
                found = true;
            }

            // if found,只用把这个level剩下的都加到result里，不需要check下面的code 来进行next level/round
            if (found)
                continue;

            // try to delete to form new strings/possiblities
            for (int i = 0; i < str.length(); i++) {
                if (str.charAt(i) == '(' || str.charAt(i) == ')') {
                    String newStr = str.substring(0, i) + str.substring(i + 1);

                    if (!used.contains(newStr)) {
                        q.add(newStr);
                        used.add(newStr);
                    }
                }
            }
        }

        return result;
    }


    /** 只有'(' 和 ')'。看是否组成valid的括号对
     * 那就只用count来记录'('就++ .. 是')'就--。 如果 count < 0, 说明太多')'.. 最后记得查count==0. 看有没多余的'('
     */
    private boolean isValid(String s) {
        int count = 0;

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(')        count++;
            else if (c == ')')   count--;

            if (count < 0) {
                return false;
            }
        }

        return count == 0;
    }


    /**
     * Balance Parentheses by removing fewest invalid - Facebook
     *
     * Given a string with parentheses, return a string with balanced parentheses by removing the fewest characters possible.
     * You cannot add anything to the string.
     *
     * Examples:
     *
     * balance("()") -> "()"
     * balance("  )(  ") -> ""
     * balance("  (((((  ") -> ""
     * balance("   (()()(  ") -> "()()"
     * balance("  )(())(  ") -> "(())"
     * @param s
     * @return
     *
     * 这题比上面的简单.. left -> right &  right -> left 左右扫2次 two pass 即可
     */
    public String balancedParentheses(String s) {
        StringBuilder sb = new StringBuilder();
        int left = 0;
        int right = 0;

        for (int i = 0; i < s.length(); i++) {  //skip all invalid right parentheses
            char c = s.charAt(i);
            if (c == '(') {
                left++;
                sb.append(c);
            } else if (c == ')' && left > right) {      // && left > right 这样 invalid就不会right++
                right++;
                sb.append(c);
            } else if (c != ')') {      // 字母..  记得 判断 != ')'. 否则多余的')'也加进来了
                sb.append(c);
            }
        }

        s = sb.toString();
        sb = new StringBuilder();
        left = 0;
        right = 0;
        for (int i = s.length() - 1; i >= 0; i--) {     //skip all invalid left parentheses
            char c = s.charAt(i);
            if (c == ')') {
                right++;
                sb.append(c);
            } else if (c == '(' && right > left) {
                left++;
                sb.append(c);
            } else if (c != '(') {                  // 字母..  记得 判断 != '('. 否则多余的'('也加进来了
                sb.append(c);
            }
        }

        return sb.reverse().toString();     // 最后记得reverse
    }


    // 如果只能 one pass. 就要用stack存invalid括号'(' 或 ')'的index.. 最后sb要delete这个stack里的index
    // 正常看到 '(' 就push进去，看到有')'就pop.. 如果stack没有对应的'('，那就把这个invalid的闭括号也push进去
    public String balancedParentheses1(String s) {

        Stack<Integer> stack = new Stack<>();          // 放invalid的括号index

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                stack.push(i);
            } else if (c == ')') {
                if (stack.isEmpty() || s.charAt(stack.peek()) == ')') {
                    stack.push(i);
                } else {
                    stack.pop();
                }
            }
        }

        StringBuilder sb = new StringBuilder(s);
        while (!stack.isEmpty()) {
            sb.deleteCharAt(stack.pop());           // delete掉invalid 括号 () 的index
        }

        return sb.toString();
    }

    
    
    /**
     * 459. Repeated Substring Pattern
     * 看string是否是重复的pattern。"abab" TRUE.  "aba" FALSE.
     * @param s
     * @return
     * 跟上一题一样可以直接 return  (s + s).indexOf(s, 1) < s.length()
     * 
     * 或者这里，看不同的长度，然后每个长度试一遍看能否都equals substring
     */
    public boolean repeatedSubstringPattern(String s) {

        // easy 方法
//        return  (s + s).indexOf(s, 1) < s.length();


        int n = s.length();
        //最长的substring是len/2,一次次缩小长度来看
        for (int len = n / 2; len >= 1; len--) {
            if (n % len != 0)         //不能整除那就不算
                continue;

            String sub = s.substring(0, len);
            int i = 1;
            for (; i < n / len; i++) {     //看repeat几次
                if (!sub.equals(s.substring(i * len,  i * len + len)))
                    break;
            }
            if (i == n / len)   return true;
        }
        return false;
        
    }

    /**
     * 686. Repeated String Match
     *
     * 看string A需要repeat多少次才能有B 成为它的substring
     *
     * A = "abcd" and B = "cdabcdab".  需要repeat 3次才行
     *
     * 也是用 (s+..+s).indexOf(sub) >= 0 来判断
     */
    public int repeatedStringMatch(String A, String B) {
        StringBuilder sb = new StringBuilder(A);        // 放 A
        int count = 1;

        while (sb.indexOf(B) < 0) {
            if (sb.length() - A.length() > B.length())
                return -1;

            sb.append(A);
            count++;
        }
        return count;
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
     * 2. 分配完dp[i][j]后，要看str是否有重复可以缩写，所以变成 2[a5[b]]
     * 		查t是否repeat时，组成2个t, 查t是否从1开始重新出现，并idx < t.len。
     * 					比如abab, 组成2个t就是abababab,能找到重复的t,idx为2, < 4, 就是找到了
     *   PS: 下面有别的方法找是否repeat
     */
    public String encode(String s) {
        int n = s.length();
        String[][] dp = new String[n][n];       // [i,j] inclusive

        for (int l = 0; l < n; l++) {
            for (int i = 0; i < n - l; i++) {
                int j = i + l;      //end
                String sub = s.substring(i, j + 1);
                dp[i][j] = sub;
                if (l < 4) {
                    continue;
                } else {
                    for (int k = i; k < j; k++) {       //k to cut string into 2 substr
                        if (dp[i][k].length() + dp[k + 1][j].length() < dp[i][j].length()) {
                            dp[i][j] = dp[i][k] + dp[k + 1][j];
                        }
                    }

                    // check for repeat pattern
                    String replace = "";
                    // 看sub里面是否有重复的。如果找到的idx < sub.len，就说明有重复。
                    int idx = (sub + sub).indexOf(sub, 1);
                    if (idx >= sub.length()) {
                        replace = sub;
                    } else {
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
    }



    /**
     * 402. Remove K Digits
     * 删掉k个digits，使得结果最小
     * Input: num = "1432219", k = 3   Output: "1219"
     * Input: num = "10200", k = 1  Output: "200"。 前面的0要去掉
     *
     * 用一个char[] res来存符合的数字，然后 res[idx] 和 n[i]来比较
     * while里面比res[idx-1] > n[i]，是的话就说明要替换，所以idx--。
     * 		因为前面更新值后idx++,新的idx是空的，所以最近的值在idx-1那
     * 	  要注意加上 if len-i > digits-idx 要确保res[]能被填满. 就是说len-i剩下的还有足够的数把res里剩下的数 digits-idx填满。如果没法填满，就直接更新值，不用比了
     * 之后更新res值，res[idx++] = n[i];
     * 注意要判断范围
     * at last, 还要去掉前面是0的串
     * 
     * 其实这也算是一个 stack的array版本，看下面那个。
     */
    public String removeKdigits(String num, int k) {
        int len = num.length();
        char[] n = num.toCharArray();		//因为string只能变成char[]
        int digits = len - k;
        char[] res = new char[digits];
        int idx = 0;
        
        for (int i = 0; i < len; i++) {
        	// len-i > digits-idx 要确保res[]能被填满. 就是说len-i那还有足够的数把res里剩下的数 digits-idx填满。
            while (len - i > digits - idx  && 		// 如果i离得够近的话就可以判断，否则就添加到res里了。比如这个小点的n[i]离得很远，超过k了
                    (idx > 0 && res[idx-1] > n[i])) {     //res[idx-1] is the newly added 因为加完res[idx]后往前了一步
                    idx--;
            }
            if (idx < digits) {			//记得判断，否则idx超出范围还要再赋值会出错
                res[idx++] = n[i];
            }
        }
        idx = 0;	//check leading 0
        while (idx < digits && res[idx] == '0') {		//注意是 == '0'. 不是0
            idx++;
        }
        String s = new String(res).substring(idx);
        
        return s.length() == 0 ? "0" : s;
    }
    
    // 跟上面的区别是，这个stack.len = num.len. 且 stack[top++]=n[i]时不需要判断 top<digits
    public String removeKdigitsStack(String num, int k) {
        int len = num.length();
        char[] n = num.toCharArray();
        int digits = len - k;
        char[] stack = new char[len];
        int top = 0;
        
        for (int i = 0; i < num.length(); i++) {
            while (k > 0 && top > 0 && stack[top-1] > n[i]) {	//这个好理解， k > 0
                top--;
                k--;
            }
            stack[top++] = n[i];
        }
        
        int idx = 0;
        while (idx < digits && stack[idx] == '0') {
            idx++;
        }
        
        String s = new String(stack, idx, digits - idx);	//从idx开始，长度为digits-idx
        
        return s.length() == 0 ? "0" : s;
    }
     
    
       
    /**
     * 321. Create Maximum Number
     * 给2个数组，挑一些数组成k长度的数，使得其最大。并且要保留原先数组里的相对顺序
     * Create the maximum number of length k <= len1 + len2 from digits of the two. 
     * he relative order of the digits from the same array must be preserved. Return an array of the k digits.
     * Input:
     * nums1 = [3, 4, 6, 5]
     * nums2 = [9, 1, 2, 5, 8, 3]
     * k = 5
     * Output:
     * [9, 8, 6, 5, 3]
     *
     * for k, 2个数组取的长度不同，来进行排列组合
     * 1. for i < k, n1取长度为i的数，n2取剩下的长度k-i. 
     * 2. 两个数组分别取最大的数
     * 3. 再merge这2个subarray顺便排序。 
     * 4. 最后for完k次就知道哪种组合方式最大
     * 
     * 当 取i作为subarray长度时，要注意判断 i = Math.max(0, k - len2); i <= Math.min(k, len1)
     * 有可能 l1 + l2 == k, 那就只有一种可能，就是全部都取。
     * !!!!!!这里假定 n2比较长，所以刚开始 i 从 k-len2开始
     * 
     * greater函数里涉及到==的情况，那就看哪个长返回哪个
     */
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int result[] = new int[k];
        int len1 = nums1.length;
        int len2 = nums2.length;
        
         //find each array's max num if i digits for n1, k-i digits for n2
        //i = Math.max(0, k - len2); 不能 len1, 假定len2更长。否则用len1的话会出错
        for (int i = Math.max(0, k - len2); i <= Math.min(k, len1); i++) {     //注意如果 len1+len2 == k,那么每个数组不能取太小 
            int[] n1 = findMaxSubarray(nums1, i);
            int[] n2 = findMaxSubarray(nums2, k - i);
            int[] cand = merge(n1, n2);
            if (greater(cand, 0,result, 0)) {
                result = cand;
            }
        }
        return result;
    }
    
    public int[] merge(int[] n1, int[] n2) {
        int[] merged = new int[n1.length + n2.length];
        int i = 0, j = 0;
        for (int k = 0; k < merged.length; k++) {
            merged[k] = greater(n1, i, n2, j) ? n1[i++] : n2[j++];
        }
        
        return merged;
    }
    
    // 这里主要是如果 == 的情况，就取更长的那个。这样如果 n2短于n1, 就取n1
    public boolean greater(int[] n1, int i, int[] n2, int j) {
        for (; i < n1.length && j < n2.length; i++, j++) {
            if (n1[i] > n2[j])  return true;
            if (n1[i] < n2[j])  return false;   //if ==, then continue loop
        }
        return i < n1.length;      // n1 longer than n2 
    }
    
    // 这个跟emove K digits 一样
    public int[] findMaxSubarray(int[] nums, int len) {
        // if (len == 0)    return null;			// 这里不用返回null.. 如果len=0, 中间会跳过，最后返回new int[0]空数组。这样其他函数也不用check null
        // if (len == nums.length)      return nums;
        
        int[] sub = new int[len];
        int idx = 0;
        for (int i = 0; i < nums.length; i++) {
            while (nums.length - i > len - idx && idx > 0 && nums[i] > sub[idx-1]) {
                idx--;
            }
            if (idx < len) {
                sub[idx++] = nums[i];
            }
        }
        return sub;
    }
    
    
    
    
    /**
     * 422. Valid Word Square
     * 给list of words, 看看第i行组成的string 是否等于 第i列组成的string
     * 	"abcd",
  		"bnrt",
  		"crm",
  		"dt"      	返回TRUE
     * @param words	
     * @return
     */
    public boolean validWordSquare(List<String> words) {
        int n = words.size();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < words.get(i).length(); j++) {
                if (j >= n || i >= words.get(j).length() ||
                    words.get(i).charAt(j) != words.get(j).charAt(i)) {
                        return false;
                    }
            }
        }
        return true;
    }
    
    
    
    
    /**
     * 274. H-Index
     * A scientist has index h if h of his/her N papers have at least h citations each, and the other N − h papers have no more than h citations each.
     * citations = [3, 0, 6, 1, 5] 返回3， 因为最多有3篇文章被引用次数>=3的
     * @param citations
     * @return
     * 类似于counting sort - O(n)
     * 需要用一个数组存一下 "多少篇文章被引用的次数为i"
     *   当一篇文章被引用超过n次(数组长度), 那就默认是n.因为最多也就n篇文章
     * 第二次for循环要 从后往前 扫，因为找max. 所以找 count >= i 的数。count指后面的文章数
     */
    public int hIndex(int[] citations) {
        int n = citations.length;
        if (n == 0)     return 0;
           
        int[] counts = new int[n + 1];		// +1因为还有可能被引用次数 >= n
        
     // Array数组算 有多少篇文章被引用i次
        for (int cnt : citations) {
            if (cnt > n)    counts[n]++;		//当一篇文章被引用超过n次(数组长度), 那就默认是n.
            else            counts[cnt]++;       // 被引用几次，就在c[n]++
        }
        
        int times = 0;
        for (int i = n; i >= 0; i--) {      //从后往前，因为找max
            times += counts[i];
            if (times >= i) {			//这里 i 是次数
                return i;
            }
        }
        return 0;
    }
    
    
    /** 275. H-Index II
     * 如果已经sort好了
     * @param citations
     * @return
     * 这比较naive，O(n)慢
     */
    public int hIndexII(int[] citations) {
        int n = citations.length;
        if (n == 0)     return 0;
        int count = 0;
        for (int i = n - 1; i >= 0; i--) {
            if (count >= citations[i]) {
                return count;
            }
            count++;
        }
        return count;
    }
    
    // 二分法 快点
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
    
    
    
    
    
    /**
     * 318. Maximum Product of Word Lengths
     * 找出2个词的max乘积，这两个词不能有common letters  "abc"和"eaf"不行,因为有"a"
     * @param words
     * @return
     *
     * 普通的O(N^2) brute force
     *
     * 每次第i个单词新开一个int[26]，再for一遍后面的j=i+1看是否有common letters.
     * 注意2点
     *  a. j从i+1算，而不是0，避免重复计算
     *  b. 每次循环j个单词时，只用看 charAt(k)是否存在过，不用真的填进去
     *  
     *  PS: 可以先sort，从长到短的单词排，这样找到就提前break,不用继续找
     */
    public int maxProduct(String[] words) {
        int len = words.length;
        int max = 0;
        for (int i = 0; i < len; i++) {
            int[] arr = new int[26];
            for (char c : words[i].toCharArray()) {
                arr[c - 'a']++;
            }
            // compare other words
            for (int j = i + 1; j < len; j++) {
                int k = 0;
                for (; k < words[j].length(); k++) {
                    if (arr[words[j].charAt(k) - 'a'] != 0)
                        break;          //only check if != 0, no need to set
                }
                if (k == words[j].length()) {
                    max = Math.max(max, words[i].length() * words[j].length());
                }
            }
        }
        return max;
    }
    
    
    /**
     * 318. Maximum Product of Word Lengths
     * @param words
     * @return
     * 用 bit manipulation能加快
     * 首先for 所有数，预处理把每个word变成二进制，比如abcd变成00001111
     * 之后再for i,j两两相比看是否有重复的字母
     * 
     * int有4*8 = 32bits. 小写字母只有26，所以可以用bit表示
     * 跟a离得越远，就左移得越多.. 比如 abcd, 变成bit以后是 dcba.
     *  PS: 这题只care有没common letter，所以出现几次没所谓.. abcd" and "aabbccdd" will both return  ***0 1111
     *  "abcd" and "efgh" 会变成 **** 0000 1111 跟  **** 1111 0000
     *  
     * bytes[i] |= 1 << (c - 'a'); 每个字母用 或| 这样出现的为1，左移1位为了不让a变成0
     * 
     * PS: 可以先sort，从长到短的单词排，这样找到就提前break,不用继续找
     */
    public int maxProductBit(String[] words) {
        int len = words.length;
        int[] bytes = new int[len];
        
        // 预处理，把每个word变成二进制，比如abcd变成00001111
        for (int i = 0; i < len; i++) {
            for(char c: words[i].toCharArray()){
                bytes[i] |= 1 << (c - 'a');
            }
        }
        
        int max = 0;
        for (int i = 0; i < len; i++) {
            for (int j = i + 1; j < len; j++) {
                if ((bytes[i] & bytes[j]) == 0 && words[i].length() * words[j].length() > max) {
                    max = words[i].length() * words[j].length();
                }
            }
        }
        return max;
    }
    
    
    
    
    /**
     * 475. Heaters  - easy
     * 给出houses, heaters位置，return能让每个house都warm的最小半径
     * Input: [1,2,3],[2]。 Output: 1 半径为1就可以
     * @param houses
     * @param heaters
     * @return
     * 先给heaters和house排序 （下面的方法可以house不用排序，在sorted heaters里二分法找）
     * 1. 然后就for循环每个house
     * 2. while 找左右更近的.. 如果heater[i+1]距离更近或==，那就i++.
     *      主要这里用while..有可能重复的数，不能光if
     * 3. result算 所需min半径的最大，才能覆盖
     */
    public int findRadius(int[] houses, int[] heaters) {
        Arrays.sort(houses);
        Arrays.sort(heaters);
        int i = 0, result = 0;
        
        for (int house : houses) {
            // 只在左右的min里算max result
            while (i < heaters.length - 1 && 
                Math.abs(heaters[i+1] - house) <= Math.abs(heaters[i] - house)) {
                i++;
            }			// 这里要求max, 才能覆盖到任意情况
            result = Math.max(result, Math.abs(heaters[i] - house));
        }
        return result;
    }
    
    
    /**
     * 475. Heaters  - easy但并不觉得 = =
     * 给出houses, heaters位置，return能让每个house都warm的最小半径
     * Input: [1,2,3],[2]。 Output: 1 半径为1就可以
     * @param houses
     * @param heaters
     * @return
     * 在找房子对应的heater位置时，如果pos < 0，说明没跟heater重合，这时才算之后的值
     *  ~pos = -pos - 1  取非
     *  
     *  int binarySearch(int[] a, int key)
     *  return index of the search key, if it is contained in the array; 
     *  otherwise, (-(insertion point) - 1). The insertion point is defined as the point at which the key 
     *  would be inserted into the array: the index of the first element greater than the key, 
     *  or a.length if all elements in the array are less than the specified key. 
     *  Note that this guarantees that the return value will be >= 0 if and only if the key is found.
     * 
     * 如果heater跟house在同一个位置，可以不用算，所需radius为0就能warm到这房子了
     * 
     * 1. 先把heaters排序
     * 2. 找到每个house最近的左右heater （用二分法找）
     * 3. 算出 左右heater距离，取min
     * 4. result = max(result, min).最后的result要找到min距离的max
     * 
     */
    public int findRadiusBinary(int[] houses, int[] heaters) {
        Arrays.sort(heaters);			//sort暖气。比较少..
        int result = 0;
        
        for (int house : houses) {		
            // 用二分法，在heaters[]里找house's nearest heaters
            int pos = Arrays.binarySearch(heaters, house);
            
            // if pos >= 0, no need to consider this house 
            if (pos < 0) {      //can't find heaters in house position
                pos = -pos - 1;      //or just pos = ~pos;
                
                // 看house跟左右2个heater哪个更近
                int dist1 = pos > 0 ? house - heaters[pos - 1] : Integer.MAX_VALUE;
                int dist2 = pos < heaters.length ? heaters[pos] - house : Integer.MAX_VALUE;
                
                result = Math.max(result, Math.min(dist1, dist2));
            }
        }
        return result;
    }
    
    
    

    
    /**
     * 302. Smallest Rectangle Enclosing Black Pixels
     * 0 as a white pixel and 1 as a black. 1的点都是连在一起的，找能enclose所有1的最小area
     * [  "0010",
		  "0110",
		  "0100"  ]  并且给出任意1的点，比如 (0,2). 返回最少需要area为6
     * @param image
     * @param x
     * @param y
     * @return
     * 最直接的方法就是dfs找
     * 
     * !!注意 visit过变成0，可以不用再变回来. (要变回来的基本是backtracking, 比如摆列组合)
     */
    public int minArea(char[][] image, int x, int y) {
        if (image == null || image.length == 0)     return 0;
        
        int m = image.length;
        int n = image[0].length;

        int[] bounds = new int[4];
        bounds[0] = x;      //upper   初始值
        bounds[1] = x;      //lower
        bounds[2] = y;      //left
        bounds[3] = y;      //right
        
        dfs(image, bounds, x, y);
        
        return (bounds[1] - bounds[0] + 1) * (bounds[3] - bounds[2] + 1);		//记得+1
    }
    
    int[] idx = {0, -1, 0, 1, 0};
    public void dfs(char[][] image, int[] bounds, int x, int y) {
        if (x < 0 || y < 0 || x >= image.length || y >= image[0].length || image[x][y] == '0')
            return;
        
        if (x < bounds[0])   bounds[0] = x;
        if (x > bounds[1])   bounds[1] = x;
        if (y < bounds[2])   bounds[2] = y;
        if (y > bounds[3])   bounds[3] = y;
        
        image[x][y] = '0';			//记得标为visit
        
        for (int i = 0; i < 4; i++) {
            dfs(image, bounds, x + idx[i], y + idx[i + 1]);
        }
    }
    
    /**
     * 302. Smallest Rectangle Enclosing Black Pixels - 二分法
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
        if (image == null || image.length == 0)
            return 0;
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

            // 查每行 找1，然后缩小col的范围
            int r = top;
            while (r < bottom && image[r][cmid] == '0') {
                r++;
            }
                        // == 保证要么都true要么都false
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
            int rmid = (s + e) / 2;    //binary search row

            // 查每列 找1，然后缩小row的范围
            int c = left;
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
    
    
    
    /**
     * 243. Shortest Word Distance
     * ["practice", "makes", "perfect", "coding", "makes"]
     * Given word1 = “coding”, word2 = “practice”, return 3.
		Given word1 = "makes", word2 = "coding", return 1.  返回最短距离
     * @param words
     * @param word1
     * @param word2
     * @return
     * 用2个index存，就可以比较
     */
    public int shortestDistance(String[] words, String word1, String word2) {
        int min = words.length;
        int p1 = -1, p2 = -1;
        for (int i = 0; i < words.length; i++) {
            if (words[i].equals(word1)) {
                p1 = i;                     // 若w1==w2，if (same)	p2 = p1;
            } else if (words[i].equals(word2)) {
                p2 = i;
            }
            
            if (p1 != -1 && p2 != -1) {
                min = Math.min(min, Math.abs(p1 - p2));
            }
        }
        return min;
    }
    
    
    /** 245. Shortest Word Distance III
     * 跟上面一样.. except 如果word1 == word2怎么办？
     * 那就判断是否一样.. 一样的话，p1先保留上次的值
     */
    public int shortestWordDistanceIII(String[] words, String word1, String word2) {
        int idx1 = -1;
        int idx2 = -1;
        int min = words.length;
        boolean isSame = word1.equals(word2);

        for (int i = 0; i < words.length; i++) {
            if (words[i].equals(word1)) {
                if (isSame) {
                    idx2 = idx1;
                }

                idx1 = i;           // 记得更新 i
            } else if (words[i].equals(word2)) {
                idx2 = i;
            }

            if (idx1 != -1 && idx2 != -1) {
                min = Math.min(min, Math.abs(idx1 - idx2));
            }
        }
        return min;
    }
    
    //用一个index存.. 在算min时 同时check 之前找的index词不能跟当前一样
    public int shortestDistance1(String[] words, String word1, String word2) {
        int min = words.length;
        int idx = -1;
        for (int i = 0; i < words.length; i++) {
            if (words[i].equals(word1) || words[i].equals(word2)) {
                // 非初始-1 &&  如果找到了idx,不能跟当前的相等
                if (idx != -1 && !words[idx].equals(words[i])) {	
                	//如果是III允许w1=w2.这里要判断是否same
                               // 说明 2个word都找到       ||  都找到 & same
             // if (idx != -1 && (!words[idx].equals(words[i]) || same))
                    min = Math.min(min, i - idx);
                }
                idx = i;			// 记得更新idx
            }
        }
        return min;
    }

    // 不管 word1跟word2 是否一样，都可以
    public int shortestWordDistanceIII2(String[] words, String word1, String word2) {
        int idx1 = -1, idx2 = -1;
        int min = words.length;

        for (int i = 0; i < words.length; i++) {
            if (words[i].equals(word1)) {
                idx1 = i;
                if (idx2 != -1) {
                    min = Math.min(min, idx1 - idx2);
                }
            }
            if (words[i].equals(word2)) {
                idx2 = i;
                if (idx1 != -1 && idx1 != i) {
                    min = Math.min(min, idx2 - idx1);
                }
            }
        }
        return min;
    }
    
    
    
    /** 244. Shortest Word Distance II
     * 上一题的followup。如果 shortest()被call了很多次，那要怎么优化？
     * @param words
     * 这就要用HashMap来存每个word出现的坐标index list.
     * 拿到2个词的list以后，就开始比较看i, j坐标最短..
     * 因为放进去时index都是sorted，所以很方便.. 如果某个坐标比较小，那往后挪++
     */
    public void WordDistance(String[] words) {
        map = new HashMap<>();
        for (int i = 0; i < words.length; i++) {
            //把这个word出现的坐标index都放map里
            if (!map.containsKey(words[i])) {
                map.put(words[i], new ArrayList<Integer>());
            }     
            map.get(words[i]).add(i);
        }
    }
    
    Map<String, List<Integer>> map;
    
    public int shortestDistanceII(String word1, String word2) {
        List<Integer> idx1 = map.get(word1);
        List<Integer> idx2 = map.get(word2);
        int i = 0, j = 0;
        int min = Integer.MAX_VALUE;
        
        while (i < idx1.size() && j < idx2.size()) {
            min = Math.min(min, Math.abs(idx1.get(i) - idx2.get(j)));
            
            // 2个index list都是sorted的，所以尽量缩小两者distance
            if (idx1.get(i) < idx2.get(j)) {
                i++;            // 往j靠
            } else {
                j++;
            }
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
     * 414. Third Maximum Number  - easy
     * 找到第三大，没的话就返回最大
     *
     * 最重要是当n == 某个max时就continue
     * 
     * 用long防止int越界
     * 也可以用 Integer , 但注意后面要判断null
     */
    public int thirdMax(int[] nums) {
        // Integer max1 = null;
        // Integer max2 = null;
        // Integer max3 = null;
        long max1 = Long.MIN_VALUE;
        long max2 = Long.MIN_VALUE;
        long max3 = Long.MIN_VALUE;
        
        for (int n : nums) {
            if (n == max1 || n == max2 || n == max3)
      //      if (n.equals(max1) || n.equals(max2) || n.equals(max3))
                continue;
            if (n > max1) {
         //  if (max1 == null || n > max1) {
                max3 = max2;
                max2 = max1;
                max1 = n;
            } else if (n > max2) {
                max3 = max2;
                max2 = n;
            } else if (n > max3) {
                max3 = n;
            }
        }
        return max3 == Long.MIN_VALUE ? (int) max1 : (int)max3;
    }
    
    
    
    /**
     * 289. Game of Life
     * 每个cell的初始是 live (1) or dead (0)
     * 现在要根据邻居更新next状态
     * @param board
     * [2nd bit, 1st bit] = [next state, current state]

		- 00  dead (next) <- dead (current)
		- 01  dead (next) <- live (current)  
		- 10  live (next) <- dead (current)  
		- 11  live (next) <- live (current) 
	 * 看现在的状态，只用 &1就行，因为最右边是current
	 * 看next状态，右移 >>1就行
	 * 
	 * 需要in place, 所以不能用另外的matrix存..
	 * 只有0，1两种状态，那就用bit表示
     */
    public void gameOfLife(int[][] board) {
        if (board == null || board.length == 0) return;
        int m = board.length, n = board[0].length;
    
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int lives = neighborLives(board, m, n, i, j);
                
                // 根据邻居lives来变现在的状态 01表示next0死，1cur生
                if (board[i][j] == 1 && (lives == 2 || lives == 3)) {
                    board[i][j] = 3;    // 01 -> 11
                }
                if (board[i][j] == 0 && lives == 3) {
                    board[i][j] = 2;    // 00 -> 10
                }
            }
        }
        
        // 把current状态去掉变成next来更新
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] >>= 1;          //右移就能把之前的去掉
            }
        }
    }
    
    public int neighborLives (int[][] board, int m, int n, int x, int y) {
        int lives = 0;
        for (int i = Math.max(0, x - 1); i <= Math.min(m - 1, x + 1); i++) {
            for (int j = Math.max(0, y - 1); j <= Math.min(n - 1, y + 1); j++) {
                lives += board[i][j] & 1;   //看他们状态是否live
            }
        }
        lives -= board[x][y] & 1;       //for里面重复算了自己
        return lives;
    }
    
    
    
    /**
     * 311. Sparse Matrix Multiplication
     * 两个矩阵相乘.. 比如A是3*4，B是4*5，结果是3*5的矩阵..
     *  i x k 的矩阵A乘以一个 k x j , 会得到 i x j 的矩阵
     * res[i][j] = A的i行 * B的j列 相加。只是res[i][j]那个数 (不是整行)
     * @param A
     * @param B
     * @return
     */
    public int[][] multiply(int[][] A, int[][] B) {
        int am = A.length;
        int an = A[0].length;
        int bn = B[0].length;
        int[][] res = new int[am][bn];
        
        for (int i = 0; i < am; i++) {
            for (int k = 0; k < an; k++) {
                if (A[i][k] == 0)			// A若是0，就不用算
                    continue;
                for (int j = 0; j < bn; j++) {
                    if (B[k][j] != 0) {
                        res[i][j] += A[i][k] * B[k][j];		//看这2个数，如果为0可以在前面跳过
                    }
                }
            }
        }
        return res;
    }
        
    // 建一个表  如果A很大
    public int[][] multiply1(int[][] A, int[][] B) {
        int am = A.length;
        int an = A[0].length;
        int bn = B[0].length;
        int[][] res = new int[am][bn];
        //存非0的数和位置. size为m.每一行存非0的
        List<int[]>[] nums = new List[am];      //不用[m][2]表示，否则初始时0，无法确定是否0的位置是0
        
        for (int i = 0; i < am; i++) {
            List<int[]> list = new ArrayList<>();
            for (int k = 0; k < an; k++) {
                if (A[i][k] != 0) {
            //        list.add(j);
                    list.add(new int[] {k, A[i][k]});		//存非0的数
                }
            }
            nums[i] = list;
        }
        
        for (int i = 0; i < am; i++) {
          //  List<int[]> list = nums[i];		//一样的写法 
            // for (int j = 0; j < list.size() - 1; j += 2) {
            //     int colA = list.get(j);
            //     int valA = list.get(j + 1);
            for (int[] a : nums[i]) {
                for (int j = 0; j < bn; j++) {
                    res[i][j] += a[1] * B[a[0]][j];
                }
            }
        }
        
        return res;
    }
    
    
    
    /** 311. Sparse Matrix Multiplication
     * 建两个表  O(M * N)
     */
    public int[][] multiply3(int[][] A, int[][] B) {
        int[][] result = new int[A.length][B[0].length];
        List<Node> listA = new ArrayList<>();
        List<Node> listB = new ArrayList<>();
        for (int i=0;i<A.length;i++) {
            for (int j=0; j<A[0].length; j++) {
                if (A[i][j]!=0) listA.add(new Node(i,j));
            }
        }
        for (int i=0;i<B.length;i++) {
            for (int j=0;j<B[0].length;j++) {
                if (B[i][j]!=0) listB.add(new Node(i,j));
            }
        }

        for (Node nodeA : listA) {
            for (Node nodeB: listB) {
                if (nodeA.y==nodeB.x) {		//ay == bx
                    result[nodeA.x][nodeB.y] += A[nodeA.x][nodeA.y] * B[nodeB.x][nodeB.y];
                }
            }
        }

        return result;
    }
    
    class Node {
        int x,y;
        Node(int x, int y) {
            this.x=x;
            this.y=y;
        }
    }
    
    /**
     * 398. Random Pick Index
     * Given an array of integers with possible duplicates, randomly output the index of a given target number.
     * 这个target可能出现在多个index，随机返回某个index
     * nums[]可能存在duplicates, 给个target，给出对应index的random值
     * @param nums
     * @param target
     * @return
     * 每个数放map里，value是list of index
     * 然后根据target，来get对应的index list. 通过random.nextInt(size)来random选index
     */
    public int randomPick(int[] nums, int target) {
    	Map<Integer, List<Integer>> indexMap = new HashMap<>();
    	Random random = new Random();
    	
    	// 预处理 O(n)
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            if (!indexMap.containsKey(num)) {
                indexMap.put(num, new ArrayList<Integer>());
            }
            indexMap.get(num).add(i);
        }
        
        // pick random O(1)
        List<Integer> indices = indexMap.get(target);
        int i = random.nextInt(indices.size());
        return indices.get(i);
    }
    
    
    
    /**
     * 398. Random Pick Index - 数很大的情况，或者data stream那种一个个进来，不确定size为多少
     *
     * 水塘抽样法 每次pick要O(n), 但不用额外空间。
     *
     * 通常..对于第n个数，count=n, 那么抽中rand.nextInt(n)==0的概率为 1/n (1/count)
     * 对于第n-1个数，被抽中的概率是 ( 1 / (n-1) ) * ( (n-1) / n) = 1 / n
     * 							   抽中第n-1		*  其他所有数都抽不中(n-1) / n
     * 
     * 每次找到target,那么count++. 最后在count个数里找random
     * 如果返回0，那就把result设成最后出现这个数的index
     */
    public int pickRandom(int[] nums, int target) {
    	Random random = new Random();
        int count = 0;
        int result = -1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == target) {
                count++;
                if (random.nextInt(count) == 0) {
                    result = i;			//记住这里是i.. 因为count!=i, 结果需要返回index
                }
            }
        }
        return result;
    }


    /**
     * 528. Random Pick with Weight
     * w[i] describes the weight of index i
     * [1, 3], 说明index 0有1次机会抽中，index 1有3次机会抽中
     * @param w
     * 把这些sum加起来放到count[] 里,  二分搜索binary search  也可以 treemap
     *
     * followup: 如果经常改变weights怎么办？
     * binary indexed tree
     * https://leetcode.com/problems/random-pick-with-weight/discuss/182620/Follow-up%3A-what-if-we-can-change-weights-array
     */
    public void randomPickWithWeight(int[] w) {
        sum = 0;
        random = new Random();

        len = w.length;
        count = new int[len];

        for (int i = 0; i < len; i++) {
            sum += w[i];
            count[i] = sum;

            // treemap
            bst.put(sum, i);
        }
    }

    int[] count;
    int len;
    int sum;
    Random random;


    // binary search
    public int pickIndex() {
        int rand = random.nextInt(sum) + 1;     // 找个数count，所以要最后+1 [1, sum]

        int s = 0, e = len - 1;

        while (s < e) {
            int mid = s + (e - s) / 2;
            if (count[mid] == rand) {
                return mid;
            } else if (count[mid] < rand) {
                s = mid + 1;                    // 小的话肯定不会是结果
            } else {
                e = mid;                        // 可能会是结果
            }
        }
        return s;
    }

    TreeMap<Integer, Integer> bst = new TreeMap<>();

    /**
     * 528. Random Pick with Weight  - TreeMap做法
     *
     * 用 treemap，这样 ceiling(4) = 6这个count，对应的value就是index
     * [1,2,3]->{1:0, 3:1, 6:2}
     * 记得在constructor里 for循环时 bst.put(sum, i);
     */
    public int pickIndexTreeMap() {
        int rand = random.nextInt(sum) + 1;
        int key = bst.ceilingKey(rand);
        return bst.get(key);
    }


    /**
     * 497. Random Point in Non-overlapping Rectangles
     * 给一个list of rectangles.. rects[i] = [x1,y1,x2,y2]
     *
     * 需要实现pick() 能randomly返回某个在这些矩形里的点
     *
     * 跟上一题一样，也还是用prefixSum..
     *
     * 1. 用prefixSum 存矩形的面积.. 这样到时random pick时能知道pick哪个矩形
     * 2. 根据pick到的矩形，再pick该矩形里的任一点
     */
    public void randomInRectangle(int[][] rects) {
        rectangles = rects;
        sum = 0;
        bstMap = new TreeMap<>();
        random = new Random();

        prefixSum = new ArrayList<>();

        // 算矩形的面积，加到prefixSum里
        for (int i = 0; i < rects.length; i++) {
            int[] rect = rects[i];
            sum += (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1);
            bstMap.put(sum, i);

            prefixSum.add(sum);
        }
    }

    TreeMap<Integer, Integer> bstMap;       // prefixSum, index (which rectangle)
    int[][] rectangles;

    // treemap比较方便.. ceiling key找大于等于的.. 挑rectangle
    public int[] pickTreeMap() {
        int idx = bstMap.ceilingKey(random.nextInt(sum) + 1);          // nextInt范围是[0, sum-1] 所以要 + 1
        int[] rectangle = rectangles[bstMap.get(idx)];     // 落在哪个矩形

        // 选这个矩形里的某一点
        int left = rectangle[0], right = rectangle[2], bottom = rectangle[1], top = rectangle[3];
        int randX = left + random.nextInt(right - left + 1);
        int randY = bottom + random.nextInt(top - bottom + 1);

        return new int[]{randX, randY};
    }


    List<Integer> prefixSum;

    // 或者 binary search找.. 这样用 list放prefixSum就行
    public int[] pickBS() {
        int rand = random.nextInt(sum);
        int l = 0, r = rectangles.length - 1;

        while (l < r) {
            int mid = (l + r) / 2;
            if (prefixSum.get(mid) == rand) {
                l = mid;
                break;
            } else if (prefixSum.get(mid) < rand) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }

        int[] rectangle = rectangles[l];     // 落在哪个矩形

        // 选这个矩形里的某一点
        int left = rectangle[0], right = rectangle[2], bottom = rectangle[1], top = rectangle[3];
        int randX = left + random.nextInt(right - left + 1);
        int randY = bottom + random.nextInt(top - bottom + 1);

        return new int[]{randX, randY};
    }



    /**
     * 384. Shuffle an Array
     * shuffle数组，并且reset可以返回original
     * @param nums
     */
    public void ShuffleSolution(int[] nums) {
        this.nums = nums;
        random = new Random();
    }

    private int[] nums;

    /** Resets the array to its original configuration and return it. */
    public int[] reset() {
        return nums;
    }

    /** Returns a random shuffling of the array. */
    public int[] shuffle() {
        int[] result = nums.clone();

        for (int i = 1; i < result.length; i++) {
            int idx = random.nextInt(i + 1);
            swap(result, idx, i);
        }
        return result;
    }
    
    

    
    /**
     * 277. Find the Celebrity
     * 给n, label是0 ~ n-1。有一个名人，他不认识其他人，但是others都认得他。
     * 看看有没有这样一个人.. 有也是只有一个。尽可能少的调用knows()
     * 其中knows(a, b) 表示A是否 knows B
     * @param n
     * @return
     *
     * 1. 第一轮for，找candidate
     * 每次如果这个candidate认识i, 那么他就不可能是，需要把i当成新的candidate
     * 否则的话，candidate可以一直不用变..
     * 因为 搜了整个list，每次会换celebrity，所以剩下最后的candidate有可能是
     *
     * 2. 第二轮for，verify这人是否celebrity
     * 之后再for一遍，看这人是否大家都认识他，并且他不认得别人
     */
    public int findCelebrity(int n) {
        int celebrity = 0;
        
        for (int i = 1; i < n; i++) {
            if (knows(celebrity, i))
                celebrity = i;          // 搜了整个list，每次会换，所以剩下最后的candidate有可能是
        }
        
        for (int i = 0; i < n; i++) {
            if (i == celebrity)
                continue;

            if (!knows(i, celebrity) || knows(celebrity, i))
                return -1;
        }
        return celebrity;
    }
    
    private boolean knows(int a, int b) {
    	return true;
    }


    /**
     * 997. Find the Town Judge
     * 跟上面的find celebrity类似.. judge不信其他人，但所有人（除了judge自己）都trust judge
     * trust, an array of pairs trust[i] = [a, b] 代表 a trusts b.
     * @param N
     * @param trust
     * @return
     */
    public int findJudge(int N, int[][] trust) {
        int[] count = new int[N + 1];

        for (int[] t : trust) {
            count[t[0]]--;
            count[t[1]]++;
        }

        for (int i = 1; i <= N; i++) {
            if (count[i] == N - 1)
                return i;
        }
        return -1;
    }
    
    
    
    /** 187. Repeated DNA Sequences
     * 找出长度为10的repeated substring
     * @param s
     * @return
     * 比较简单... 注意result先用hashset, 否则重复
     */
    public List<String> findRepeatedDnaSequences(String s) {
        Set<String> set = new HashSet<>();
        Set<String> result = new HashSet<>();
        for (int i = 0; i <= s.length() - 10; i++) {
            String sub = s.substring(i, i + 10);
            if (set.contains(sub)) {
                result.add(sub);
            } else {
                set.add(sub);
            }
        }
        return new ArrayList<>(result);
    }
    
    
    /** 187. Repeated DNA Sequences
     * string只有A, C, G, and T组成
     * @param s
     * @return
     * 那么4个字母分别用2个bits表示，00, 01, 10, 11. 这样比string要节省空间
     */
    public List<String> findRepeatedDnaSequencesBit(String s) {
        Set<Integer> words = new HashSet<>();
        Set<Integer> doubleWords = new HashSet<>();
        List<String> rv = new ArrayList<>();
        char[] map = new char[26];
        map['A' - 'A'] = 0;
        map['C' - 'A'] = 1;
        map['G' - 'A'] = 2;
        map['T' - 'A'] = 3;
// 也可以 hashmap.put('A', 0);
        
        for(int i = 0; i < s.length() - 9; i++) {
            int v = 0;
            for(int j = i; j < i + 10; j++) {
                v <<= 2;
                v |= map[s.charAt(j) - 'A'];
            }
            if(!words.add(v) && doubleWords.add(v)) {
                rv.add(s.substring(i, i + 10));
            }
        }
        return rv;
    }
    
    
    /** 187. Repeated DNA Sequences
     * 用大小为10的window来滑动
     * @param s
     * @return
     */
    public List<String> findRepeatedDnaSequencesBest(String s) {
        Map<Character, Integer> map = new HashMap<>();
        map.put('A', 0);
        map.put('C', 1);
        map.put('G', 2);
        map.put('T', 3);
        
        Set<Integer> set = new HashSet<>();
        Set<Integer> intRes = new HashSet<>();
        List<String> result = new ArrayList<>();
        int sub = 0;
        
        for(int i = 0; i < s.length(); i++) {
            sub <<= 2;
            sub |= map.get(s.charAt(i));
            sub &= 0xfffff;
            // 0xfffff 是20个1，所以超过20的数就会被去掉，那么就去掉头2个。
            
            if (i < 9)      continue;			//前10个不用算
            
            if (!set.add(sub) && intRes.add(sub)) {
                result.add(s.substring(i - 9, i + 1));
            }
        }
        return result;
    }
    
    
    
    

    /**
     * 444. Sequence Reconstruction
     * Check whether the original sequence org can be uniquely reconstructed from the sequences in seqs. 
     * The org sequence is a permutation of the integers from 1 to n, with 1 ≤ n ≤ 104. 
     * 问这些子序列能否唯一的重建出原序列
     *
     * seqs里任意两个数字的顺序必须跟org里的一致。所以要有2点：
     * 		a. 要有pos[]来表示org里的数字所在的index位置，这样能看顺序是否一致
     * 		b. 要保证org里的所有数都在seqs里出现过，并且顺序OK。所以要有个matched记录
     * 1. 要有pos[]来表示org里的数字所在的index位置，这样seq[i]能通过pos来找到org里对应的位置 pos[pre] < pos[cur]
     * 2. 由于org是 1~n, 所以seq里不能有 <= 0 或>n的情况 ，不能有其他数字
     * 3. 用flg[]来标记cur是否被验证过.. 已经验证过matched不用+
     * 4. 若没被验证过，且seq里i-1和i的顺序跟在org里一样 pos[pre]+1 == pos[cur]，那就OK，match++
     * 		如果pos[pre]和pos[cur]中间隔了挺多，不需要match++, 否则算多了
     * 5. 最后看是否match了n个数，证明所有顺序都验证成功、就可以了
     * 
     * 为了防止org=[1], seqs为空的情况返回TRUE，前面用hasNum来区分是否为空
     *
     *
     * 法2：也可以拓扑排序
     *         int index = 0;
     *         while(!queue.isEmpty()) {
     *             int size = queue.size();
     *             if(size > 1)
     *                 return false;
     *             int curr = queue.poll();
     *             if(index == org.length || curr != org[index++])
     *                 return false;
     *             for(int next: map.get(curr)) {
     *                 indegree.put(next, indegree.get(next) - 1);
     *                 if(indegree.get(next) == 0)
     *                     queue.offer(next);
     *             }
     *         }
     *         return index == org.length && index == map.size();
     */
    public boolean sequenceReconstruction(int[] org, List<List<Integer>> seqs) {
        int n = org.length;
        int[] pos = new int[n + 1];
        for (int i = 0; i < n; i++) {
            pos[org[i]] = i;                    // org里的数字所在的index位置
        }
        
        boolean[] flags = new boolean[n + 1];
        int matched = 1;
        boolean hasNum = false;
        
        for (List<Integer> seq : seqs) {
            for (int i = 0; i < seq.size(); i++) {
                hasNum = true;
                if (seq.get(i) <= 0 || seq.get(i) > n)
                    return false;   //org[]是1~n, 判断越界
                if (i == 0)
                    continue;
                int pre = seq.get(i - 1);
                int cur = seq.get(i);
                if (pos[pre] >= pos[cur])
                    return false;       //seq所对应org里的POS前后顺序不能错
                if (flags[pre] == false && pos[pre] + 1 == pos[cur]) {
                    flags[pre] = true;	  // 因为在seqs里pre在cur前面，所以也需要org里顺序一样，用pos[]查
                    matched++;
                }
            }
        }
        return matched == n && hasNum;
    }
    
    
    
    /** 157. Read N Characters Given Read4
     *  API: int read4(char *buf) reads 4 characters at a time from a file.
     *  call一次read，返回总共read多少个char
     * @param buf
     * @param n
     * @return
     * 有可能 buf里有a, b. 只read 1个
     */
    public int read(char[] buf, int n) {
        for (int i = 0; i < n; i += 4) {
            char[] tmp = new char[4];
            int len = read4(tmp);
            System.arraycopy(tmp, 0, buf, i, len);
            if (len < 4)
                return Math.min(i + len, n);        //有可能需要读的n比较小
        }
        return n;
    }
    
    
    
    
    /**
     * 158. Read N Characters Given Read4 II - Call multiple times
     * @param buf
     * @param n
     * @return
     * 要call多次read.. 那么有可能上一次没读满4个，那这次就要从上次剩下的读起..
     * 所以需要维护一个 !!*全局变量buffIdx和buffCnt, 看上次有多少没读完
     */
    private int buffIdx = 0;
    private int buffCnt = 0;		//看上次有多少没读完
    private char[] buffer = new char[4];
    
    public int readII(char[] buf, int n) {
        int i = 0;
        while (i < n) {
            if (buffIdx == 0) {         //读新的一轮
                buffCnt = read4(buffer);
            }
            //把tmp的buffer copy到buf。 主要是，如果上次没读完的buffIdx,这次继续
            while (i < n && buffIdx < buffCnt) {
                buf[i++] = buffer[buffIdx++];       
            }
            if (buffIdx == buffCnt) 
                buffIdx = 0;            //读完之前剩下的buffer以后，就reset为0
            
            if (buffCnt < 4)		//没满4个，所以是最后了直接return即可
                return i;
        }
        return i;
    }
    
    
    /**  API: int read4(char *buf) reads 4 characters at a time from a file.
     * 并且会把东西copy到buf里
     */
    public int read4(char[] buf) {
    	return buf.length;
    }
    
    
    
    /**
     * 205. Isomorphic Strings
     * isomorphic if the characters in s can be replaced to get t.
     * 并且一个char只能被另一个char或自己代替，不能被多个代替
     * Given "egg", "add", return true.
     * Given "foo", "bar", return false.
     * Given "paper", "title", return true.
     * @param s
     * @param t
     * @return
     * 只能是一对一的映射，那就2个map..
     */
    public boolean isIsomorphic(String s, String t) {
        int[] sm = new int[256];
        int[] tm = new int[256];
        for (int i = 0; i < s.length(); i++) {
            if (sm[s.charAt(i)] != tm[t.charAt(i)])
                return false;
            // []里放index
            sm[s.charAt(i)] = i + 1;        //需要至少为1，因为0是默认没有的
            tm[t.charAt(i)] = i + 1;
        }
        return true;
    }

    /**
     * 205. Isomorphic Strings
     * 用hashmap... 但是，第二次check是 containsValue()... 这样就是O(n) 慢
     * 或者也可以用 additional Set 来存 b.. 但是这样也慢
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic1(String s, String t) {
        if (s.length() != t.length())
            return false;

        Map<Character, Character> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            char c1 = s.charAt(i);
            char c2 = t.charAt(i);
            if (map.containsKey(c1)) {
                if (map.get(c1) != c2) {
                    return false;
                }
            } else {
                if (map.containsValue(c2)) {     // !!!! containsValue(c2), 或者 set.contains(b)
                    return false;
                } else {
                    map.put(c1, c2);            // 或者顺便加 set.add(b)
                }
            }
        }
        return true;
    }


    /**
     * 890. Find and Replace Pattern
     * Input: words = ["abc","deq","mee","aqq","dkd","ccc"], pattern = "abb"
     * Output: ["mee","aqq"]
     *
     * 其实就是上面那道 isIsomorphic 的题.. 只是string变成list而已.. 一样的
     */
    public List<String> findAndReplacePattern(String[] words, String pattern) {
        List<String> result = new ArrayList<>();

        for (String word : words) {
            if (isIsomorphic(word, pattern)) {
                result.add(word);
            }
        }
        return result;
    }
    
    
    
    /**
     * 290. Word Pattern
     * pattern = "abba", str = "dog cat cat dog" should return true.
	   pattern = "abba", str = "dog cat cat fish" should return false.
     * @param pattern
     * @param str
     * @return
     * 跟上面那题很像，也是需要一一映射
     * 可以用一个map+set.. 因为set只是用过的word才会在里面
     */
    public boolean wordPattern(String pattern, String str) {
        String[] strs = str.split(" ");
        if (pattern.length() != strs.length)
            return false;
            
        Map<Character, String> map = new HashMap<>();
        Set<String> set = new HashSet<>();
        
        for (int i = 0; i < pattern.length(); i++) {
            char c = pattern.charAt(i);
            if (map.containsKey(c)) {
                if (!map.get(c).equals(strs[i]))
                    return false;
            } else {
                if (!set.add(strs[i]))      //说明之前这word已经用过pattern
                    return false;   
                map.put(c, strs[i]);
            }
        }
        return true;
    }
    
    
    
    /**
     * 291. Word Pattern II
     * 没有空格 隔开，pattern = "abab", str = "redblueredblue" should return true.
     * 那么只能用backtracking
     */
    public boolean wordPatternMatch(String pattern, String str) {
        Map<Character, String> map = new HashMap<>();
        Set<String> set = new HashSet<>();
        
        return helper(pattern, str, map, set);
    }
    
    // 用i, j的坐标，会快点
    private boolean isMatch(String str, int i, String pat, int j, Map<Character, String> map, Set<String> set) {
        // base case
        if (i == str.length() && j == pat.length()) return true;
        if (i == str.length() || j == pat.length()) return false;

        // get current pattern character
        char c = pat.charAt(j);

        // if the pattern character exists
        if (map.containsKey(c)) {
            String s = map.get(c);

            // then check if we can use it to match str[i...i+s.length()]
            if (!str.startsWith(s, i)) {
                return false;
            }

            // if it can match, great, continue to match the rest
            return isMatch(str, i + s.length(), pat, j + 1, map, set);
        }

        // pattern character does not exist in the map
        for (int k = i; k < str.length(); k++) {
            String p = str.substring(i, k + 1);

            if (set.contains(p)) {
                continue;
            }

            // create or update it
            map.put(c, p);
            set.add(p);

            // continue to match the rest
            if (isMatch(str, k + 1, pat, j + 1, map, set)) {
                return true;
            }

            // backtracking
            map.remove(c);
            set.remove(p);
        }

        // we've tried our best but still no luck
        return false;
    }

    // 或者substring, 方便点
    public boolean helper(String pattern, String str, Map<Character, String> map, Set<String> set) {
    	if (pattern.length() == 0 && str.length() == 0)      return true;
        if (pattern.length() == 0 || str.length() == 0)     return false;

        char p = pattern.charAt(0);
        
        if (map.containsKey(p)) {
            String sub = map.get(p);
            	//需要match
            if (!str.startsWith(sub)) {		// 用坐标的话是if (!str.startsWith(s, i)) 
                return false;		
            } else {
                return helper(pattern.substring(1), str.substring(sub.length()), map, set);
            }		
        }

        // else 不在map的情况

        for (int i = 1; i <= str.length() - pattern.length() + 1; i++) {
            String sub = str.substring(0, i);
            if (set.contains(sub))
                continue;

            map.put(p, sub);
            set.add(sub);

            if (helper(pattern.substring(1), str.substring(i), map, set))
                return true;

            map.remove(p);
            set.remove(sub);
        }


        return false;
    }
    
    
    
    /**
     * Window Sum
     * 返回 窗口size为k的sum数组
     */
    public int[] winSum(int[] nums, int k) {
    	if (nums == null || nums.length < k || k <= 0)
            return new int[0];
    	
    	int len = nums.length;
    	int[] sums = new int[len - k + 1];
    	
    	//第一个window sum
    	for (int i = 0; i < k; i++) {
    		sums[0] += nums[i];
    	}
    	
    	// 开始移动window
    	for (int i = 1; i < len; i++) {
    		sums[i] = sums[i - 1] - nums[i - 1] + nums[i + k - 1];
    	}
    	return sums;
    }



    /**
     * 973. K Closest Points to Origin
     * 给一个origin点 (0, 0)，找k个离他最近的点
     * O(nlogK)
     */
    public int[][] kClosest(int[][] points, int K) {
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>(K, (p1, p2) -> p2[0] * p2[0] + p2[1] * p2[1] - p1[0] * p1[0] - p1[1] * p1[1]);

        for (int[] p : points) {
            maxHeap.offer(p);
            if (maxHeap.size() > K) {
                maxHeap.poll();
            }
        }

        int[][] result = new int[K][2];
        while (K > 0) {
            result[--K] = maxHeap.poll();
        }

        return result;
    }


    /**
     * 973. K Closest Points to Origin
     * 用quick select方法更快
     * @param points
     * @param K
     * @return
     */
    public int[][] kClosest1(int[][] points, int K) {
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
     * K Closest Points
     * 给一个origin点，找k个离他最近的点
     * 如果距离一样，那就sort by x轴，然后再sort by y轴
     */
    private Point global_origin = null;			// PQ里面需要origin变Final，或者外部声明
    
    public Point[] kClosest(Point[] points, Point origin, int k) {
        global_origin = origin;
        PriorityQueue<Point> pq = new PriorityQueue<Point> (k, new Comparator<Point> () {
            public int compare(Point a, Point b) {
                int diff = getDistance(b, global_origin) - getDistance(a, global_origin);
                if (diff == 0)
                    diff = b.x - a.x;
                if (diff == 0)
                    diff = b.y - a.y;
                return diff;
            }
        });

        for (int i = 0; i < points.length; i++) {
            pq.offer(points[i]);
            if (pq.size() > k)
                pq.poll();
        }
        
        k = pq.size();
        Point[] ret = new Point[k];  
        while (!pq.isEmpty())
            ret[--k] = pq.poll();
        return ret;
    }
    
    private int getDistance(Point a, Point b) {
        return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
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

    // Naive 最简单的就是按照 绝对值diff来排序，然后取前K个
    public List<Integer> findClosestElementsNaive(List<Integer> arr, int k, int x) {
        Collections.sort(arr, (a,b) -> a == b ? a - b : Math.abs(a-x) - Math.abs(b-x));
        arr = arr.subList(0, k);
        Collections.sort(arr);          // 需要再sort一次，因为之前按照diff排序的
        return arr;
    }
    
    
    
    /**
     * 68. Text Justification
     * [
		   "This    is    an",
		   "example  of text",
		   "justification.  "
		]
     * 空格要平均分，多的话left加空格
     * 最后一行不要额外多的空格，每个单词之间一个space就行.. 后面的right补齐space
     * @param words
     * @param maxWidth
     * @return
     *
     * 几个关键点
     * a. 初始count = -1, 因为last word没有空格
     * b. 先加 word[i].. 再while() 加空格，再加后面的word，这样最后无需考虑删除空格，因为最后是单词
     * c. space默认1， 最后做完handle特殊情况(one word 或者 last line)就行，不需要if-else
     */
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> result = new ArrayList<>();
        int len = words.length;
        int i = 0;

        while (i < len) {
            int count = -1;             // skip最后word的space，所以刚开始 count = -1
            int j = i;
            for (; j < len && count + words[j].length() + 1 <= maxWidth; j++) {
                count += words[j].length() + 1;     // 1 extra for the space
            }

            int gaps = j - 1 - i; // j 已经是next 多一个了，需要 -1
            int spaces = 1;
            int extraSpaces = 0;

            if (j != i + 1 && j != len) {              // 正常情况 not 1 char, not last line
                spaces = (maxWidth - count) / gaps + 1;     // 上面算每个count时本身有 + 1(default)
                extraSpaces = (maxWidth - count) % gaps;
            }

            StringBuilder sb = new StringBuilder(words[i]);     // !!!! 先加第一个Word

            // 正常情况
            while (++i < j) {
                for (int s = 0; s < spaces; s++) {
                    sb.append(' ');                         // 这样下面先加 空格" " ，再加word
                }
                if (extraSpaces > 0) {
                    sb.append(' ');
                    extraSpaces--;
                }
                sb.append(words[i]);
            }

            // 特殊情况 一个word 或者 last line
            int remaining = maxWidth - sb.length();
            while (remaining > 0) {
                sb.append(' ');
                remaining--;
            }
            result.add(sb.toString());
        }

        return result;
    }


    // 稍微麻烦.. 可以跳过
    public List<String> fullJustify2(String[] words, int maxWidth) {
        List<String> list = new ArrayList<>();
        int len = words.length;
        int i = 0;
        while (i < len) {
            int line = 0;
            int start = i;
            while (i < len && line + words[i].length() <= maxWidth) {
                line += words[i].length() + 1;
                i++;
            }
            int gaps = i - start - 1;               //词之间有几个gap
            int spaceNum = maxWidth - (line - 1) + gaps;     //最后面多了个空格

            StringBuilder sb = new StringBuilder();

         // 如果只能一个单词 || 最后一行
            if (i == len || gaps == 0) {     
                for (int j = start; j < i; j++) {
                	sb.append(words[j] + " ");
                	spaceNum--;
                }
                // 为了防止一个词为"", 多加空格
                sb.deleteCharAt(sb.length() - 1);
                spaceNum++;
                while (spaceNum-- > 0) {
                	sb.append(" ");
                }
                list.add(sb.toString());
            } else {            				// 其他正常的情况
                int evenSpace = spaceNum / gaps;		// 能够平均分的space
                spaceNum %= gaps;
                for (int j = start; j < i; j++) {
                	sb.append(words[j]);
                    for (int k = 0; k < evenSpace; k++) {
                        sb.append(" ");			//先加平分的evenSpace
                    }
                    if (spaceNum > 0) {         //剩下多余的不平均，先加左边
                    	sb.append(" ");
                        spaceNum--;
                    }
                }
                list.add(sb.toString().trim());
            }
        }
        return list;
    }


    /**
     * 953. Verifying an Alien Dictionary
     * 判断这些word list 是否valid order
     * words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"     TRUE.. 第一个字母 h 在 l 前面， OK
     * words = ["word","world","row"], order = "worldabcefghijkmnpqstuvxyz"   FALSE  word那个 d应该在l前面，需要world前 word后
     * words = ["apple","app"], order = "abcdefghijklmnopqrstuvwxyz"          FALSE  app要在 apple之前, 短的在前
     * @param words
     * @param order
     * @return
     */
    public boolean isAlienSorted(String[] words, String order) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < order.length(); i++) {
            map.put(order.charAt(i), i);
        }

        for (int i = 0; i < words.length - 1; i++) {
            String w1 = words[i];
            String w2 = words[i + 1];
            int minLen = Math.min(w1.length(), w2.length());
            int j = 0;
            while (j < minLen) {
                char c1 = w1.charAt(j);
                char c2 = w2.charAt(j);
                if (c1 != c2) {
                    if (map.get(c1) > map.get(c2)) {
                        return false;
                    } else {
                        break;
                    }
                } else {
                    j++;
                }
            }
            if (j == minLen && w1.length() > w2.length()) {     // 短的要在前面
                return false;
            }
        }
        return true;
    }

    // 分开来，比较简洁
    public boolean isAlienSorted1(String[] words, String order) {
        if(words == null || words.length == 0) {
            return true;
        }

        int[] map = new int[26];
        for(int i = 0; i < order.length(); i++) {
            map[order.charAt(i) - 'a'] = i;
        }

        for(int i = 0; i < words.length - 1; i++) {
            if(compare(words[i], words[i + 1], map) > 0) {
                return false;
            }
        }
        return true;
    }

    private int compare(String word1, String word2, int[] map) {
        for(int i = 0, j = 0; i < word1.length() && j < word2.length(); i++, j++) {
            char c1 = word1.charAt(i);
            char c2 = word2.charAt(j);

            if (map[c1 - 'a'] != map[c2 - 'a']) {
                return map[c1 - 'a'] - map[c2 - 'a'];
            }
        }
        return word1.length() - word2.length();
    }


    
    /**
     * 269. Alien Dictionary
     * words are sorted lexicographically
     *   "wrt",
		  "wrf",
		  "er",
		  "ett",
		  "rftt"  返回 "wertf"  . 如果是"z","x","z" 那就是空，因为顺序错了
     * @param words
     * @return
     * 拓扑排序.. 跟course schedule II 很像
     * 建立indegree表，记得顺便count一下总共几个char
     * 同时初始化出现的char的indegree为0. 
     * 
     * ！！注意如果 两个char不相等，就要开始比较。
     * 如果c2之前出现在c1的set里，那就不要重复加否则indegree会多加一次
     */
    public String alienOrder(String[] words) {
        if (words == null || words.length == 0)
            return "";
        
        Map<Character, Set<Character>> map = new HashMap<>();
        int[] indegree = new int[26];
        Arrays.fill(indegree, -1);
        int count = 0;
        
        // 先把出现过的indegree设为0
        for (String w : words) {
            for (char c : w.toCharArray()) {
                if (indegree[c - 'a'] != 0) {
                    indegree[c - 'a'] = 0;
                    count++;
                }
            }
        }
        
        for (int i = 0; i < words.length - 1; i++) {
            char[] w1 = words[i].toCharArray();
            char[] w2 = words[i + 1].toCharArray();
            int minLen = Math.min(w1.length, w2.length);
            for (int j = 0; j < minLen; j++) {
                if (w1[j] != w2[j]) {
                    if (map.containsKey(w2[j]) && map.get(w2[j]).contains(w1[j])) {
                        return "";          //说明有环
                    }
                    if (!map.containsKey(w1[j])) {
                        map.put(w1[j], new HashSet<Character>());
                    }
                    // 注意不要重复加
                    if (!map.get(w1[j]).contains(w2[j])) {
                        map.get(w1[j]).add(w2[j]);    //w1的邻居是w2
                        indegree[w2[j] - 'a']++;
                    }
                    break;
                }
            }
        }
        
        Queue<Character> q = new LinkedList<>();
        for (int i = 0; i < 26; i++) {
            if (indegree[i] == 0) {
                q.add((char) (i + 'a'));
            }
        }
        
        StringBuilder sb = new StringBuilder();
        
        while (!q.isEmpty()) {
            char c = q.poll();
            sb.append(c);
            if (!map.containsKey(c))
                continue;
            for (char cc : map.get(c)) {
                indegree[cc - 'a']--;
                if (indegree[cc - 'a'] == 0) {
                    q.add(cc);
                }
            }
        }
        
        if (sb.length() != count)   return "";		//记得判断是否==出现的次数
        return sb.toString();
        
    }
    
    
    
    /**
     * 395. Longest Substring with At Least K Repeating Characters
     * substring里的所有字母至少出现k次
     * s = "aaabb", k = 3  返回3
     * s = "ababbc", k = 2 返回5
     * @param s
     * @param k
     * @return
     * 找出不valid的，recursive调用substring就行..
     * 可以传进去start & end 规定范围
     */
    public int longestSubstring(String s, int k) {
        if (s == null || s.length() == 0 || s.length() < k)
            return 0;
            
        return divHelper(s, 0, s.length(), k);
    }
    
    public int divHelper(String s, int start, int end, int k) {
        if (end - start < k)    
            return 0;
        
        int[] counts = new int[26];
        // 统计所有char的出现次数
        for (int i = start; i < end; i++) {
            counts[s.charAt(i) - 'a']++;
        }
        
        // 扫多一次，看是否有invalid的就rec substring
        
        for (int i = start; i < end; i++) {
            if (counts[s.charAt(i) - 'a'] > 0 && counts[s.charAt(i) - 'a'] < k) {
                int left = divHelper(s, start, i, k);
                int right = divHelper(s, i + 1, end, k);
                return Math.max(left, right);
            }
        }
        
        return end - start;     //如果都valid直接return
    }
    
    
    /**
     * 395. Longest Substring with At Least K Repeating Characters
     * substring里的所有字母至少出现k次
     * s = "aaabb", k = 3  返回3
     * s = "ababbc", k = 2 返回5
     * @param s
     * @param k
     * @return
     * 这个是找第一个不符合的.. recursive次数比较多
     */
    public int longestSubstring1(String s, int k) {
        if (s == null || s.length() == 0 || s.length() < k)
            return 0;
            
        if (k == 1)     return s.length();
        
        int[] counts = new int[26];
        // count all freq
        for (int i = 0; i < s.length(); i++) {
            counts[s.charAt(i) - 'a']++;
        }

        // 看看哪个frequency < k
        char notValid = ' ';            //只要找到一个就能break
        for (int i = 0; i < 26; i++) {
            if (counts[i] > 0 && counts[i] < k) {
                notValid = (char) (i + 'a');
                break;
            }
        }
        
        if (notValid == ' ')
            return s.length();
        
        // 把这个不合格的点找出来split
        int max = 0;
        String[] subs = s.split(notValid + "");  
        for (String sub : subs) {
            max = Math.max(max, longestSubstring(sub, k));
        }
        return max;
    }
    
    
    /**
     * 554. Brick Wall
     * 砖块交错放.. 问有没一条线可以cross最少的砖块.. 如果在end就不算cross
     * @param wall
     * @return
     *
     * 出现的sum（length）最多，也就是说能cross最少
     */
    public int leastBricks(List<List<Integer>> wall) {
        if (wall == null || wall.size() == 0)
            return 0;
        
        Map<Integer, Integer> map = new HashMap<>();    //length, freq
        int count = 0;
        for (List<Integer> rows : wall) {
            int sum = 0;
            for (int j = 0; j < rows.size() - 1; j++) {     //不用算最后那个
                sum += rows.get(j);
                map.put(sum, map.getOrDefault(sum, 0) + 1);
                count = Math.max(count, map.get(sum));
            }
        }
        
        return wall.size() - count;
    }


    /**
     * Task Schedule I - Facebook - 不能reorder
     * if tasks cannot be reordered, output the total time needed: O(n) space 求最后需要的时间
     * @param tasks
     * @param cooldown
     * @return
     * 用map存这个task下一次能开始的时间 curTime + cooldown + 1
     */
    public int taskScheduleI(int[] tasks, int cooldown) {
        Map<Integer, Integer> map = new HashMap<>();        // <taskId, time> 下一次能执行的时间time

        int time = 0;       // how many time we need to complete all

        for (int task : tasks) {
            if (map.containsKey(task) && map.get(task) > time) {
                time = map.get(task);               //if we need to wait for the cooldown of the same task, just update the slots
            }

            map.put(task, time + cooldown + 1);         // 每次都要update + 1 是下一次这task能做完的时间
            time++;                                 // cur task用了1 time/slot
        }

        return time;
    }



    /**
     * Task Schedule - Facebook  - 不改变顺序
     * if we need to output a string of the task scheduling(without reordering), eg.1,2,1,1,3,4, k=2, -> 12_1__134
     * @param tasks
     * @param cooldown
     * @return
     * 跟上一题一样，只是打印出结果
     */
    public String taskScheduleII(int[] tasks, int cooldown) {
        if (tasks == null || tasks.length == 0) {
            return "";
        }
        Map<Integer, Integer> map = new HashMap<>();//store indices, where cooldown stops, of tasks in window
        int time = 0;
        StringBuilder sb = new StringBuilder();

        for (int task : tasks) {
            if (map.containsKey(task) && map.get(task) > time) {
                //add this code if our output is a string, eg.AA, 2 -> A__A
                int waitingTime = map.get(task) - time;
                for (int i = 0; i < waitingTime; i++) {
                    sb.append("_");
                }
                time = map.get(task);//if we need to wait for the cooldown of the same task, just update the slots
            }
            map.put(task, time + cooldown + 1);//update the time slot to the time when curr task can be done again
            sb.append(task);//remember to append the task !!!
            time++;//update the used 1 slot of curr task
        }
        return sb.toString();
    }



    /**
     * 621. Task Scheduler - 改变顺序
     * cooling interval n that means between two same tasks, 一样的task之间至少间隔n 的cooldown
     * You need to return the least number of intervals the CPU will take to finish all the given tasks. 可以改变顺序
     * tasks = ["A","A","A","B","B","B"], n = 2
     * Output: 8
     * Explanation: A -> B -> idle -> A -> B -> idle -> A -> B.
     *
     * 需要找出frequency最高 的task，排好它以后别的也能满足了..
     *
     * 两种情况：
     *   a. maxCount少，会有多余的idles, 那么会比本身lenghth要长 longer
     *   b. maxCount多，没有idles, 不需要受cooldown影响，这样result其实是 tasks.length
     *
     * 如果有多个 maxCount (一样出现max次)，那也算上.. 如果超过了需要的cooldown n,那没所谓这样就不受cooldown影响，最后是length
     *
     * 由于a, 我们要算多少idles.. 所以用 需要的emptySlots - nonMaxTasks (其他低频的task) 看剩下多少空位
     *
     * a 情况: 多余idels + len
     *  slotLength = cooldown - (maxCount - 1); 指每个slots的length.。 比如 AAA BBB CCC DE
     *  刚开始             A_ _ _ A _ _ _ A
     *  有多一个B，放进去填  AB _ _ AB _ _ A   这样 slotsLength就变成 2了 cooldown - (maxCount - 1) 这里减一 是因为刚开始已经用了A
     *
     * b 情况: 直接len
     *  如果 AAA BBB CCC DDD EEE FFF G
     *  刚开始  A_ _ _ A _ _ _ A
     *  剩下的maxCount有 BCDEF, 超过了 cool down n, 那就cooldown - (6 - 1) < 0.. 这时cool down就不用管了，反正它们之前随便排都能超过cooldown.
     */
    public int leastInterval(char[] tasks, int cooldown) {
        int[] count = new int[26];
        int max = 0;
        int maxCount = 0;

        for (char c : tasks) {
            count[c - 'A']++;
            if (max == count[c - 'A']) {
                maxCount++;
            } else if (max < count[c - 'A']) {          // 找出现最多次数的
                max = count[c - 'A'];
                maxCount = 1;
            }
        }

        int slots = max - 1;
        // 如果有多个max task, 超过了n. 那下面就是负的，没所谓，不受cooldown n的约束，到时返回tasks.length就行
        int slotLength = cooldown - (maxCount - 1);        // - 1 is one of the max task
        int emptySlots = slots * slotLength;
        int nonMaxTasks = tasks.length - max * maxCount;        // 其实少于max的tasks
        int idles = Math.max(0, emptySlots - nonMaxTasks);      // 填满all task后需要用到的idles

        return tasks.length + idles;        // len + idles..

        // ======================或者可以下面这样一句话======================

        // 可以分成 (max - 1)块 interval，每个interval length是 n + 1 (包括其中一个max task本身 比如下面例子的A)
        // 最后需要加上maxCount   比如 AAA BBB CD =>  (A _ _) (A _ _) A  => (A B _ ) (A B _ ) AB

//        return Math.max(tasks.length, (max - 1) * (coolDown + 1) + maxCount);

    }


    /**
     * 621. Task Scheduler - 改变顺序
     *
     * 这个用max PriorityQueue来存frequency..每次优先poll max
     * 每轮cycle (cooldown) 都执行任务，看哪些任务还有freq要加到remainTasks里..
     * 直到pq和remainTasks为空，说明都结束了
     */
    public int leastInterval1(char[] tasks, int cooldown) {
        Map<Character, Integer> counts = new HashMap<Character, Integer>();
        for (char t : tasks) {
            counts.put(t, counts.getOrDefault(t, 0) + 1);
        }

        // pq 存frequency，先poll最高frequency的
        PriorityQueue<Integer> pq = new PriorityQueue<Integer>((a, b) -> b - a);
        pq.addAll(counts.values());

        int time = 0;

        while (!pq.isEmpty()) {
            List<Integer> remainTasks = new ArrayList<>();      // 剩下还需要work的tasks

            for (int i = 0; i <= cooldown; i++) {       // 每次一轮cooldown看做完task后哪些还剩下
                if (!pq.isEmpty()) {
                    if (pq.peek() > 1) {            // 还有多次要做，所以加到remainTasks里下次再做
                        remainTasks.add(pq.poll() - 1);
                    } else {
                        pq.poll();                  // 执行完了这个task
                    }
                }
                time++;             // 每次都 time++

                if (pq.isEmpty() && remainTasks.isEmpty()) {        // 都完成了
                    break;
                }
            }

            // 过完一次cycle (cooldown)后，看还有多少要运行..
            pq.addAll(remainTasks);
        }

        return time;
    }


    /**
     * 358. Rearrange String k Distance Apart
     * Rearrange the string such that the same characters are at least distance k from each other.
     *
     * 其实跟上面的 task scheduler 改变顺序 一样.. 要输出结果
     * Input: s = "aabbcc", k = 3
     * Output: "abcabc"
     *
     * 1. 用maxHeap存entry, 按照frequency排序..
     * 2. 需要有个queue来放freeze的element，这样当queue >= k 时，我们就不会freeze，可以从queue里poll()出来.
     * 3. 从queue.poll() 出来的数，如果还有frequency要加到maxHeap里，这样才能继续加string result
     *
     * 注意frequency变成0时，也需要放 freezeQ 里，这样才算上了处理过的letter，到时freezQ的size++后才能达到k
     */
    public String rearrangeString(String s, int k) {
        Map<Character, Integer> counts = new HashMap<Character, Integer>();
        for (char c : s.toCharArray()) {
            counts.put(c, counts.getOrDefault(c, 0) + 1);
        }

        // pq 存frequency，先poll最高frequency的, freq一样的话按照letter顺序
        PriorityQueue<Map.Entry<Character, Integer>> pq = new PriorityQueue<>((a, b) -> a.getValue() != b.getValue() ? b.getValue() - a.getValue() : a.getKey() - b.getKey());
        pq.addAll(counts.entrySet());

        // waitlist queue to freeze previous appeared char in period of k
        Queue<Map.Entry<Character, Integer>> freezeQ = new LinkedList<>();
        StringBuilder sb = new StringBuilder();

        while(!pq.isEmpty()) {
            Map.Entry<Character, Integer> cur = pq.poll();
            sb.append(cur.getKey());            // 加到string里

            cur.setValue(cur.getValue() - 1);
            // 加waitlist里.即使变成0也加进去，这样说明freeze算上了其他letter，到时freezQ的size++后才能达到k 然后poll，并能继续加到pq里
            freezeQ.offer(cur);

            if (freezeQ.size() >= k) {          // 767. Reorganize String 那题跟这个一样，只是k=2。因为相邻不同same，所以k=2
                Map.Entry<Character, Integer> nextOne = freezeQ.poll();
                if (nextOne.getValue() > 0) {
                    pq.offer(nextOne);
                }
            }
        }

        return sb.length() == s.length() ? sb.toString() : "";
    }


    /**
     * 358. Rearrange String k Distance Apart
     *
     * 除了有个count[]算frequency以外，还需要一个数组存对应letter下一次的valid pos 从哪开始
     *
     * 1. count[]存frequency
     * 2. for string, 对每个index查接下来next valid position..
     *      a. 需要找count max的, greedy
     *      b. 同时保证这个pos是valid的。如果不valid的话不能用
     */
    public String rearrangeString1(String s, int k) {
        int len = s.length();
        int[] count = new int[26];
        int[] validPos = new int[26];

        for (char c : s.toCharArray()) {
            count[c - 'a']++;
        }

        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < len; i++) {
            int nextPos = findNextValidPos(count, validPos, i);
            if (nextPos == -1)
                return "";

            sb.append((char) ('a' + nextPos));
            count[nextPos]--;
            validPos[nextPos] = i + k;      // 下一次这个letter至少间隔k, 所以只能至少在i+k的地方valid用
        }

        return sb.toString();
    }

    private int findNextValidPos(int[] count, int[] validPos, int index) {
        int max = 0;
        int pos = -1;

        for (int i = 0; i < count.length; i++) {
            // 优先考虑最大的.. 这样后面低频的才能随便放.  idx需要至少validPos 才行，否则不能用这个i
            if (count[i] > 0 && count[i] > max && index >= validPos[i]) {
                max = count[i];
                pos = i;
            }
        }
        return pos;
    }


    /**
     * 767. Reorganize String
     * Reorganize string使得same char 不能相邻
     * Input: S = "aab"
     * Output: "aba"
     * @param s
     * @return
     *
     * 这个其实跟上面的358. Rearrange String k Distance Apart 一样，只是 k = 2, 因为需要第2个才能再填这个letter
     *
     * 但是这个其实简单挺多.. 因为可以直接用maxHeap根据frequency来poll 两次，这样保证不会same letter相邻..
     *
     * 用CountChar比较clean，其实上面那题也可以这样 更符合OOD
     *
     * 下面还有个更快的.. 无需PriorityQueue
     */
    public String reorganizeString(String s) {
        int len = s.length();

        int[] count = new int[26];
        for (char c: s.toCharArray()) {
            count[c -'a']++;
        }

        PriorityQueue<CountChar> pq = new PriorityQueue<>((a, b) ->
                                                              a.count == b.count ? a.letter - b.letter : b.count - a.count);

        for (int i = 0; i < 26; i++) {
            if (count[i] > 0) {
                if (count[i] > (len + 1) / 2)       // freq超过一半就肯定有相邻的情况
                    return "";

                pq.add(new CountChar((char) ('a' + i), count[i]));
            }
        }

        StringBuilder sb = new StringBuilder();

        while (pq.size() >= 2) {
            CountChar cc1 = pq.poll();
            CountChar cc2 = pq.poll();

            sb.append(cc1.letter);
            sb.append(cc2.letter);

            if (--cc1.count > 0)    pq.add(cc1);
            if (--cc2.count > 0)    pq.add(cc2);
        }

        if (pq.size() > 0) {
            sb.append(pq.poll().letter);
        }

        return sb.toString();
    }

    class CountChar {
        int count;
        char letter;

        public CountChar(char letter, int count) {
            this.letter = letter;
            this.count = count;
        }
    }


    // 快！  这个找到max frequency, 然后先填even的idx, 每次 idx += 2.. 之后再for循环别的letter填剩下的位置
    public String reorganizeString1(String s) {
        int len = s.length();

        int[] count = new int[26];
        for (char c: s.toCharArray()) {
            count[c -'a']++;
        }

        int max = 0, maxLetter = 0;

        for (int i = 0; i < 26; i++) {
            if (max < count[i]) {
                max = count[i];
                maxLetter = i;
            }
        }

        if (max > (len + 1) / 2)
            return "";

        char[] arr = s.toCharArray();
        int idx = 0;

        // 先放max, 隔着idx来放 比如放偶数位
        while (count[maxLetter] > 0) {
            arr[idx] = (char) ('a' + maxLetter);
            idx += 2;
            count[maxLetter]--;
        }

        // 放别的letters
        for (int i = 0; i < 26; i++) {
            while (count[i] > 0) {
                if (idx >= len) {
                    idx = 1;        // 偶数放完，开始放 奇数
                }
                arr[idx] = (char) ('a' + i);
                idx += 2;
                count[i]--;
            }
        }
        return new String(arr);
    }



    /**
     * 636. Exclusive Time of Functions
     * 给一个list of logs, 按照timestamp sorted. "{function_id}:{"start" | "end"}:{timestamp}".
     * For example, "0:start:3" 表示 job id 0 started at the beginning of timestamp 3.
     * 这个类似于函数的recursion或者call了别的方法.. call别的job时当前这个就暂停，别人完成后再继续自己的..
     *
     * 所以这样的话，就用stack 来放job id
     *
     * 用prevStartTime 来存放上一次start time，这样可以利用当前cur time - preStartTime得到 间隔时长
     *
     * 注意，start时push进去。
     *      end时pop..但这时 duration = curTime - preTime + 1, 要 算上end time这个时间.
     *                并且preTime = cur + 1 , 因为cur是end time.. 但preTime我们要求是 start time
     */
    public int[] exclusiveTime(int n, List<String> logs) {
        int[] result = new int[n];
        Stack<Integer> stack = new Stack<>();       // store job id
        int prevTime = 0;                           // previous START time

        for (String log : logs) {
            String[] strs = log.split(":");
            int job = Integer.valueOf(strs[0]);
            int curTime = Integer.valueOf(strs[2]);

            if (strs[1].equals("start")) {
                if (!stack.isEmpty()) {
                    result[stack.peek()] += curTime - prevTime;
                }
                stack.push(job);
                prevTime = curTime;
            } else {                    // end
                result[stack.pop()] += curTime - prevTime + 1;      // 算上end time这个时间 所以 + 1
                prevTime = curTime + 1;             // end time + 1 = next start time
            }
        }
        return result;
    }


    /**
     * 723. Candy Crush
     * 3个以上一样的就消掉..
     * @param board
     * @return
     * 1. 扫2次，从左到右horizontally + 从上到下vertically
     * 一样的超过3个就设为负数  -val  这样就知道需要消掉他
     *
     * 2. 每列开始，把 +val 正常的放最下面的row，剩下的 -val全都设 0
     */
    public int[][] candyCrush(int[][] board) {
        int m = board.length, n = board[0].length;
        boolean found = true;

        // 一直消.. 直到 found为false为止
        while (found) {
            found = false;

            // find candy to crush & make to -val
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    int val = Math.abs(board[i][j]);
                    if (val == 0)
                        continue;

                    // scan for this row, horizontally
                    if (j < n - 2 && Math.abs(board[i][j + 1]) == val && Math.abs(board[i][j + 2]) == val) {
                        found = true;
                        int k = j;
                        while (k < n && Math.abs(board[i][k]) == val) {
                            board[i][k++] = -val;
                        }
                    }

                    // scan for this col, vertically
                    if (i < m - 2 && Math.abs(board[i + 1][j]) == val && Math.abs(board[i + 2][j]) == val) {
                        found = true;
                        int k = i;
                        while (k < m && Math.abs(board[k][j]) == val) {
                            board[k++][j] = -val;
                        }
                    }
                }
            }

            if (!found)     return board;

            // drop the candy vertically, move +val to bottom, other -val to 0
            for (int j = 0; j < n; j++) {
                int bottom = m - 1;         // +val

                // move +val to bottom
                for (int i = m - 1; i >= 0; i--) {
                    if (board[i][j] > 0) {
                        board[bottom--][j] = board[i][j];
                    }
                }
                // set rest of -val to 0
                for (int i = bottom; i >= 0; i--) {
                    board[i][j] = 0;
                }
            }
        }
        return board;
    }


    /**
     * 836. Rectangle Overlap
     * 看2个rectangle是否overlap  [x1, y1, x2, y2]. 其中(x1,y1)是左下Corner，(x2,y2)是右上Corner
     * @param rec1
     * @param rec2
     * @return
     */
    public boolean isRectangleOverlap(int[] rec1, int[] rec2) {
        // 找不想交的情况
        return !(rec1[2] <= rec2[0] ||   // left
                     rec1[3] <= rec2[1] ||   // bottom
                     rec1[0] >= rec2[2] ||   // right
                     rec1[1] >= rec2[3]);    // top
    }


    /**
     * 410. Split Array Largest Sum - Google  - DP
     * 给一个数组，split成m块，看如何切能找出最小的max sum. array顺序不能变
     * nums = [7,2,5,10,8]  m = 2  结果是 18
     *
     * 跟另一题一样 1011. Capacity To Ship Packages Within D Days
     * Return the least weight capacity of the ship that will result in all the packages on the conveyor belt
     * being shipped within D days.
     *
     * dp[k][j]表示将数组中前j个数字分成k组所能得到的最小的各个子数组中最大值
     *
     * 由于需要知道前面 k - 1各组能 切到第几个数，那么需要for一下j 找到中间的 i
     *
     * 比如 找 i时， 0 ~ i ~ j
     * 那么子数组的max值就是 Math.max(dp[k - 1][i], sums[j] - sums[i])
     *   看 前面分成k-1块时前i个数的dp结果 min (maxSum) 大，还是 这次的结果 sum(i~j)的 大
     *
     *
     * 后面有个更巧妙更快的二分查找
     */
    public int splitArray(int[] nums, int m) {
        int n = nums.length;
        int[] sums = new int[n + 1];
        int[][] dp = new int[m + 1][n + 1];

        // prefix sum
        for (int i = 1; i <= n; i++) {
            sums[i] = sums[i - 1] + nums[i - 1];
        }

        // 初始化
        for (int[] arr : dp) {
            Arrays.fill(arr, Integer.MAX_VALUE);
        }

        dp[0][0] = 0;

        for (int k = 1; k <= m; k++) {          // 前 k 块
            for (int j = 1; j <= n; j++) {
                for (int i = 0; i < j; i++) {   // 0 ~ i~ j，因为不确定怎么cut, 所以for循环找切割点i
                    // 看 前 i 个数分成 k-1 块   VS  这次sum[i, j]  谁大
                    int val = Math.max(dp[k - 1][i], sums[j] - sums[i]);
                    dp[k][j] = Math.min(dp[k][j], val);
                }
            }
        }
        return dp[m][n];
    }


    /**
     * 410. Split Array Largest Sum - Google  - Binary Search
     * 跟另一题一样 1011. Capacity To Ship Packages Within D Days
     *
     * 因为数组不能换顺序，我们知道 sum的min 和 max的值
     * 那么可以二分查找candidate sum, 来看是否能split符合条件，然后继续BS找最小的sum
     *
     * min sum: 最开始left是切割成每个num一个数组，那么最大的min sum 就是数组里的max number
     * max sum: 如果m=0不切，那么min sum 只能是整个数组和sum..
     *
     * 那么我们二分查找这些left, right找可能的sum，看能否split成这个sum，可以的话继续往前缩小sum找，否则只能往后加大sum找
     */
    public int splitArrayBS(int[] nums, int m) {
        long left = 0;          // 最大的数
        long right = 0;         // 整个数组的sum

        for (int num : nums) {
            left = Math.max(left, num);
            right += num;
        }

        while (left < right) {
            long mid = (left + right) / 2;      // possible sum
            if (canSplitToSum(nums, m, mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return (int)left;
    }

    private boolean canSplitToSum(int[] nums, int m, long sum) {
        int pieces = 1;
        long curSum = 0;

        for (int num : nums) {
            curSum += num;

            // curSum <= sum就能继续在这块cut里，如果 >sum就多加一个cut piece,重新算
            if (curSum > sum) {
                pieces++;
                curSum = num;

                if (pieces > m)
                    return false;
            }
        }
        // return true即可，pieces可以小于m, 这还是算在candidate里面，< m说明sum太大了
        return pieces <= m;
    }


    /**
     * Cutting Chocolates - Google
     * 跟上题一样，只是有点不同.. 这里需要分成K块，使得min sum最大
     *
     * 那么就转化成 -> can I split array into K sub-arrays that min subarray sum >= targetSum
     *
     * DP的话就是
     * int val = Math.min(dp[k - 1][i], sums[j] - sums[i]);
     * dp[k][j] = Math.max(dp[k][j], val);
     */
    public int splitChocolate(int[] nums, int K) {
        int left = Integer.MAX_VALUE;          // 最小的数
        int right = 0;                  // 整个数组的sum

        for (int num : nums) {
            left = Math.min(left, num);
            right += num;
        }

        // 二分查找 最max的sum 可能性
        while (left < right) {
            int mid = (left + right) / 2 + 1;       // 右移一位.. 找最右边的(大sum)可能性
            if (canSplitMinSumMoreThanTarget(nums, K, mid)) {
                left = mid;
            } else {
                right = mid - 1;
            }
        }

        return left;
    }

    // can I split array into K pieces that min subarray sum >= targetSum
    private boolean canSplitMinSumMoreThanTarget(int[] nums, int K, int targetSum) {
        int pieces = 0;
        int curSum = 0;

        for (int num : nums) {
            curSum += num;

            if (curSum >= targetSum) {      // 保证是 min sum
                curSum = 0;
                pieces++;
            }
        }
        return pieces >= K;     // 如果分的minSum 都大于 targetSum, 那么分少点块 更加大于targetSum符合题意
    }


    /**
     * 1057. Campus Bikes - Google
     * 给定人和车的(i,j)位置，匹配最近的人车..
     * 最近的人车匹配完一个以后，第二个再匹配..
     *
     * 1. 扫 人 & 车，把距离都放到PQ里..
     * 2. 扫一遍pq, update到result里，并记录used过的bike
     *
     * O(m * n * log(mn))
     */
    public int[] assignBikes(int[][] workers, int[][] bikes) {
        int workerNum = workers.length;

        PriorityQueue<WorkerBikeDistance> pq = new PriorityQueue<>();

        for (int i = 0; i < workerNum; i++) {
            int[] worker = workers[i];
            for (int j = 0; j < bikes.length; j++) {
                int distance = Math.abs(worker[0] - bikes[j][0]) + Math.abs(worker[1] - bikes[j][1]);
                pq.offer(new WorkerBikeDistance(i, j, distance));
            }
        }

        int[] result = new int[workerNum];
        Arrays.fill(result, -1);

        Set<Integer> usedBikes = new HashSet<>();
        int matched = 0;

        while (!pq.isEmpty()) {
            WorkerBikeDistance cur = pq.poll();
            if (result[cur.worker] == -1 && !usedBikes.contains(cur.bike)) {
                result[cur.worker] = cur.bike;
                usedBikes.add(cur.bike);
                matched++;

                if (matched == workerNum)       // early break
                    break;
            }
        }

        return result;
    }


    /**
     * 1057. Campus Bikes - Google
     *
     * 这个快一点, 类似bucket sort
     *
     * 假设 人/车 <= 1000, 那么最大的距离也就2001
     * 那么我们放 dist[] 数组，里面对应 List<int[]> 来放 同样distance的 人车配对(i,j)
     *
     * 那么就是 O(m * n)
     */
    public int[] assignBikes2(int[][] workers, int[][] bikes) {
        int workerNum = workers.length;

        // worker, bike <= 1000, 所以最远的distance也<= 2000
        List<int[]>[] dist = new List[2001];        // 某个distance有多少对 i,j (人，车)

        for (int i = 0; i < workerNum; i++) {
            for (int j = 0; j < bikes.length; j++) {
                int distance = Math.abs(workers[i][0] - bikes[j][0]) + Math.abs(workers[i][1] - bikes[j][1]);
                if (dist[distance] == null) {
                    dist[distance] = new ArrayList<>();
                }
                dist[distance].add(new int[]{i , j});
            }
        }

        int[] result = new int[workerNum];
        Arrays.fill(result, -1);

        Set<Integer> usedBikes = new HashSet<>();
        int matched = 0;

        for (int i = 0; i <= 2000 && matched < workerNum; i++) {
            if (dist[i] == null)
                continue;

            for (int[] pair : dist[i]) {
                if (result[pair[0]] == -1 && !usedBikes.contains(pair[1])) {
                    result[pair[0]] = pair[1];
                    usedBikes.add(pair[1]);
                    matched++;
                }
            }
        }
        return result;
    }


    class WorkerBikeDistance implements Comparable<WorkerBikeDistance> {
        int worker;
        int bike;
        int distance;

        public WorkerBikeDistance(int worker, int bike, int distance) {
            this.worker = worker;
            this.bike = bike;
            this.distance = distance;
        }

        @Override
        public int compareTo(WorkerBikeDistance other) {
            if (this.distance == other.distance) {
                if (this.worker == other.worker) {
                    return this.bike - other.bike;
                } else {
                    return this.worker - other.worker;
                }
            } else {
                return this.distance - other.distance;
            }
        }
    }


    /**
     * 849. Maximize Distance to Closest Person - easy  - google
     * In a row of seats, 1 represents a person sitting in that seat, and 0 represents that the seat is empty.
     * 找max distance to closest person
     *
     * 考虑3种情况
     * a. 刚开始00001 , 直接按照i 就行  多少个0
     * b. 中间情况，那要 /2
     * c. 最后很多0
     */
    public int maxDistToClosest(int[] seats) {
        int n = seats.length;

        int pre = -1;
        int max = 0;

        for (int i = 0; i < n; i++) {
            if (seats[i] == 1) {
                if (pre == -1) {
                    max = i;
                } else {
                    max = Math.max(max, (i - pre) / 2);
                }
                pre = i;
            }
        }

        if (seats[n - 1] == 0) {
            max = Math.max(max, n - 1 - pre);       // 如果最后也是0
        }

        return max;
    }



    /**
     * 855. Exam Room
     * 考试座位，尽可能离得远
     *
     * 用PriorityQueue存这些interval，根据distance大小来排序
     *
     * 其实用TreeSet也可以跟PQ一样.. 还能remove..  不知道为什么用PQ remove不了..
     *
     * seat 复杂度是 O(logn)
     *
     * leave 复杂度
     *   a. 如果是PQ 正常方法是O(n), 扫整个PQ找left, right邻居，然后删掉，加入新的merged
     *   b. 也可以用多一个TreeSet存所有座位seats. 这样能快速logn 找到邻居.. 再删。。 这样的话不能用PQ, 要用treeset..
     */
    class ExamRoom {

        TreeSet<Interval> pq;
//        PriorityQueue<Interval> pq;
        TreeSet<Integer> seats;         // 方便快速找到seat的前后邻居，这样能从pq里remove
        int N;

        public ExamRoom(int N) {
            pq = new TreeSet<>((a, b) -> a.distance == b.distance ? a.start - b.start : b.distance - a.distance);
//            pq = new PriorityQueue<>((a, b) -> a.distance == b.distance ? a.start - b.start : b.distance - a.distance);
            seats = new TreeSet<>();
            this.N = N;

            pq.add(new Interval(-1, N));        // 初始值
        }

        public int seat() {
            int seat = 0;
            Interval interval = pq.pollFirst();

            if (interval.start == -1) {
                seat = 0;
            } else if (interval.end == N) {
                seat = N - 1;
            } else {
                seat = (interval.start + interval.end) / 2;
            }

            pq.add(new Interval(interval.start, seat));
            pq.add(new Interval(seat, interval.end));
            seats.add(seat);

            return seat;
        }

        public void leave(int p) {
            Integer left = seats.lower(p);
            Integer right = seats.higher(p);

            if (left == null)   left = -1;
            if (right == null)  right = N;

            seats.remove(p);
            pq.remove(new Interval(left, p));
            pq.remove(new Interval(p, right));

            pq.add(new Interval(left, right));
        }


        // O(n) 把整个pq放list里，iterate找left, right
        public void leave1(int p) {
            Interval left = null, right = null;
            List<Interval> intervals = new ArrayList<>(pq);

            for (Interval interval : intervals) {
                if (interval.start == p)    right = interval;
                if (interval.end == p)      left = interval;
                if (left != null && right != null)
                    break;
            }

            pq.remove(left);
            pq.remove(right);

            pq.add(new Interval(left.start, right.end));
        }

        class Interval {
            int start;
            int end;
            int distance;

            public Interval(int start, int end) {
                this.start = start;
                this.end = end;

                // calculate distance
                if (start == -1) {
                    this.distance = end;
                } else if (end == N) {
                    this.distance = N - 1 - start;
                } else {
                    this.distance = (end - start) / 2;
                }
            }

        }
    }


    /**
     * 853. Car Fleet
     * 如果target是终点. 如果有的车可以赶上来，那么假设赶上来后速度就变成一样，这就形成了car fleet. 求问有多少car fleet
     * Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]
     * Output: 3
     * 10 & 8 一个车队，因为8可以赶上.. 0 自己， 5 & 3一个车队，在6的时候meet
     *
     * 主要就是看每个车，距离target所需要的时间多少.. sort完后就比较，如果能追上就是一个fleet，否则就自己一个fleet++
     *
     * 1. 最开始按照position来排序，大的排前面，最接近target
     * 2. 循环一遍，看后面的车到target所需的time是否 <= curtime. 是的话就是一队，否则就要fleet++. 记得更新curTime
     *
     * 也可以用TreeMap<-pos[i], time> 这样不用sort，treemap帮你sort好.. 其实也是一样的
     */
    public int carFleet(int target, int[] position, int[] speed) {
        int n = position.length;
        Car[] cars = new Car[n];

        for (int i = 0; i < n; i++) {
            cars[i] = new Car(position[i], (double) (target - position[i]) / speed[i]);
        }

        // 按照position来, pos越大越接近target的排到前面
        Arrays.sort(cars, (a, b) -> b.pos - a.pos);

        int carFleet = 0;
        double curTime = 0;

        // 如果后面的车 到target所需的time > curTime, 那永远追不上之前的车，那就另起carFleet
        for (int i = 0; i < n; i++) {
            if (cars[i].time > curTime) {
                curTime = cars[i].time;
                carFleet++;
            }
        }
        return carFleet;
    }

    class Car {
        int pos;
        double time;        // time to target

        public Car(int pos, double time) {
            this.pos = pos;
            this.time = time;
        }
    }


    /**
     * 857. Minimum Cost to Hire K Workers
     * 需要返回 min wage来hire k workers..
     * 条件：
     * 1. Every worker in the paid group should be paid in the ratio of their quality compared to other workers in the paid group.
     * 2. Every worker in the paid group must be paid at least their minimum wage expectation.
     *
     * 根据条件1，quality1/quality2 = wage1/wage2
     * 推出 -> wage1/quality1 = wage2 / quality2 = ratio, 那么让ratio尽可能小
     * 我们可以用Arrays.sort来根据ratio排序
     *
     * minWage = (q1 + q2 +..) * ratio.. 那么让quality尽可能小
     * 为了保证最终结果最小..那么维护大小为K 的maxHeap, 这样能poll出最大，剩下最小的
     *
     * 只要保证了ratio, 感觉第2个条件也会自动满足
     *
     */
    public double mincostToHireWorkers(int[] quality, int[] wage, int K) {
        int len = quality.length;
        Worker[] workers = new Worker[len];

        for (int i = 0; i < len; i++) {
            workers[i] = new Worker(quality[i], wage[i]);
        }

        // 按照ratio从小到大排
        Arrays.sort(workers, (a, b) -> Double.compare(a.ratio, b.ratio));

        // 为了保证最终结果最小..那么维护大小为K 的maxHeap, 这样能poll出最大，剩下最小的
        // minWage = (q1 + q2 +..) * ratio.. 那么让quality尽可能小
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);

        int qualitySum = 0;
        double minWage = Integer.MAX_VALUE;

        for (Worker worker : workers) {
            maxHeap.offer(worker.quality);
            qualitySum += worker.quality;

            if (maxHeap.size() > K) {
                qualitySum -= maxHeap.poll();
            }
            if (maxHeap.size() == K) {
                minWage = Math.min(minWage, qualitySum * worker.ratio);
            }
        }

        return minWage;
    }

    class Worker {
        int quality;
        int wage;
        double ratio;

        public Worker(int quality, int wage) {
            this.quality = quality;
            this.wage = wage;
            this.ratio = (wage + 0.0) / quality;
        }
    }


    /**
     * 659. Split Array into Consecutive Subsequences  - Google
     * given sorted array, 可能有duplicate，看能否split成多个 subsequence, 每个至少3个数的 连续递增组
     * Input: [1,2,3,3,4,5]   可以
     * You can split them into two consecutive subsequences :
     * 1, 2, 3
     * 3, 4, 5
     *
     * 有2个map，一个放出现频率freq, 另一个放 新开的顺子需要的频率need.
     * 1. 第一遍算频率
     * 2. 第二遍就看连续的.. 如果有need, 那看后面num+1 & +2 有没freq, 有的话就OK，顺便减掉..
     *
     * https://www.cnblogs.com/grandyang/p/7525821.html
     */
    public boolean isPossible(int[] nums) {
        Map<Integer, Integer> freq = new HashMap<>();
        Map<Integer, Integer> need = new HashMap<>();

        // 先count频率
        for (int num : nums) {
            freq.put(num, freq.getOrDefault(num, 0) + 1);
        }

        // 第二遍，主要看need能否新开一个顺子
        // c++版本更clear
        /*
        for (int num : nums) {
            if (freq[num] == 0) {
                continue;
            } else if (need[num] > 0) {     // 需要num.. 那这次匹配上了，会--freq, 但这次也要 need--，同时后面也想 need[num+1]++
                --need[num];                // 接之前的顺子
                ++need[num + 1];
            } else if (freq[num + 1] > 0 && freq[num + 2] > 0) {        // 新开顺子，要保证后面2个都OK
                --freq[num + 1];
                --freq[num + 2];
                ++need[num + 3];
            } else {
                return false;
            }

            --freq[num];
        }
        */
        for (int num : nums) {
            if (freq.get(num) == 0) {
                continue;
            } else if (need.getOrDefault(num, 0) > 0) {
                need.put(num, need.getOrDefault(num, 0) - 1);   // 接上面的顺子
                need.put(num + 1, need.getOrDefault(num + 1, 0) + 1);
            } else if (freq.getOrDefault(num + 1, 0) > 0 && freq.getOrDefault(num + 2, 0) > 0) {
                freq.put(num + 1, freq.get(num + 1) - 1);       // 新开顺子 freq[num+1 & 2]--
                freq.put(num + 2, freq.get(num + 2) - 1);
                need.put(num + 3, need.getOrDefault(num + 3, 0) + 1);   // need num+3
            } else {
                return false;
            }

            freq.put(num, freq.get(num) - 1);       // 记住减当前num freq--
        }

        return true;
    }


    public static void main(String[] args) {
    	Solution sol = new Solution();


//        start = "hit"， end = "cog"。 dict = ["hot","dot","dog","lot","log"]
//        As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",

        HashSet<String> dict = new HashSet<>();
        dict.add("hit");
        dict.add("hot");
        dict.add("dot");
        dict.add("dog");
        dict.add("lot");
        dict.add("let");
        dict.add("log");
        dict.add("cog");
        System.out.println(sol.ladderLength("hit", "cog", dict));


    	System.out.println("taskScheduleI : " + sol.taskScheduleII(new int[]{1,1,1,2,2,2}, 2));
    	System.out.println("taskScheduleII : " + sol.taskScheduleII(new int[]{1,2,1,1,3,4}, 2));

    	System.out.println(sol.alienOrder(new String[]{"wrt","wrtkj"}));
    	System.out.println(sol.fullJustify(new String[]{"This", "is", "an", "example", "of", "text", "justification."}, 16));
    //	System.out.println(sol.findDisappearedNumbers(new int[]{4,3,2,7,8,2,3,1}));
    	int[] nums = {0, 1, 0, 3, 12};
    	int i = 0;
        for (int j = 0; j < nums.length; j++) {
    //    	System.out.println("num[j] is "+nums[j]+" , j is "+j);
        	if (i == 0 && nums[i] != 0) continue;
            if (nums[j] != 0) {
                nums[i++] = nums[j];
   //             System.out.println("  in num[i] is "+nums[i]+" , i is "+i);
                nums[j] = 0;
                
            }
        }
        
        File f = new File("file1");
        File[] list = f.listFiles();
        
  //      System.out.println(sol.getPermutation(6, 9));
        
        int[][] points = new int[][]{{1,1},{-1,-1},{-1,-1},{-3,1}};
        
        
        String[] words = {"word","good","best","good"};
        String[] words1 = {"aa","aa","aa"};
        
        String str = "abcxabcdabcdabcy";
        String subString = "abcdabcy";
        System.out.println(sol.maxProductBit(words));
        
        System.out.println(sol.minAbbreviation2("apple", words));


        System.out.println(sol.balancedParentheses("))(())())"));
        System.out.println(sol.balancedParentheses1("))(())())"));
    }

    public static class UnitTest {

        @Test
        public void test1() {

            System.out.println(System.currentTimeMillis());
            System.out.println(Instant.now());

            ExamRoom sol = new Solution().new ExamRoom(10);
            sol.seat();
            sol.seat();
            sol.seat();
            sol.leave(0);
            sol.leave(4);
            sol.seat();
            sol.seat();
            sol.seat();
        }
    }
    
}
