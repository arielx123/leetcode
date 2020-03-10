import java.util.*;


public class Game {

	 /** 293. Flip Game  - easy
     * 连续2个+变成2个-, 返回所有result
     * @param s
     * @return
     */
    public List<String> generatePossibleNextMoves(String s) {
        List<String> list = new ArrayList<>();
        for (int i = 0; i < s.length() - 1; i++) {
            if (s.charAt(i) == '+' && s.charAt(i + 1)== '+') {
                list.add(s.substring(0, i) + "--" + s.substring(i + 2));
            }
        }
        return list;
    }
    
    
    /** 294. Flip Game II
     * 每个人轮流flip++成--，如果最后那人没法flip，那就另一个人win。return第一个玩家能否赢
     * @param s
     * @return
     * dfs + nextNum
     * 这种需要不断dfs试剩下的可能性，所以要recursion。在flip完后，看 对方canWin(t)能否赢
     * 
     * 既然有dfs, 那就要cache优化，因为这过程会有很多重复的string，所以用个set来存
     * 记得 call完canWin以后，再把这个t加到set里。
     * 
     * 光是dfs的话，大概是(n-2) * (n-4) *... ~ O(n!). 或者 2^n
     */
    public boolean canWinFlip(String s) {
        if (s == null || s.length() < 2)    return false;
        
        Set<String> cache = new HashSet<>();		//加cache优化
        return canWinWithCache(s, cache);
    }
    
    public boolean canWinWithCache(String s, Set<String> cache) {
        if (cache.contains(s))
            return true;        //都是可以flip-- 才放进set里
    
        for (int i = 0; i < s.length() - 1; i++) {
            if (s.charAt(i) == '+' && s.charAt(i + 1)== '+') {
                String t = s.substring(0, i) + "--" + s.substring(i + 2);
                
                if (!canWinWithCache(t, cache)) {       //根据这次结果，下一个人如果false的话就win了
                    return true;
                }
                
                cache.add(t);
            }
        }
        return false;
    }
    
    // 用map更快, 因为还会存不同的 输赢结果
    public boolean canWinWithCache(String s, Map<String, Boolean> cache) {
        if (cache.containsKey(s))
            return cache.get(s);        //都是可以flip-- 才放进cache里
    
        for (int i = 0; i < s.length() - 1; i++) {
            if (s.charAt(i) == '+' && s.charAt(i + 1)== '+') {
                String t = s.substring(0, i) + "--" + s.substring(i + 2);
                
                if (!canWinWithCache(t, cache)) {       //根据这次结果，下一个人如果false的话就win了
                    cache.put(s, true);
                    return true;
                }
            }
        }
        cache.put(s, false);
        return false;
    }
    
    
    
    /** 464. Can I Win
     * 从1 到 maxChoosableInteger，看谁能最先到desiredTotal，谁就赢
     * maxChoosableInteger <= 20，同时挑过的数不能再用
     * @param maxChoosableInteger
     * @param desiredTotal
     * @return
     * 又是一样的dfs.. 这次的dfs看看对手 dfs(total - i)是否输..
     * 
     * 由于不能用used的数，那要存下状态.. 因为maxChoosableInteger <= 20，所以可以用boolean[]或int[]
     * 用hashmap存 方案和结果.. 那方案就是boolean[]状态
     * 但是数组里面的值会变，只能用int或string.. 所以有boolToInt(bool[])转成101010
     * PS: map的key也可以直接用Arrays.toString(boolean[])。不过这里用int[]代表0, 1会更短
     * 
     * 注意.!! 如果 对方输了，map.put(key,true)时key不用更新，因为总的状态是这次dfs的，不需要used[i]就能赢
     * 同理，最后return false前，也是把这次状态key放到map里
     */
    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        int sum = maxChoosableInteger * (1 + maxChoosableInteger) / 2;
        if (sum < desiredTotal)     return false;
        if (desiredTotal <= 0)      return true;
        
        Map<Integer, Boolean> map = new HashMap<>();
        boolean[] used = new boolean[maxChoosableInteger + 1];
        return dfsWin(map, used, desiredTotal);
    }
    
    public boolean dfsWin(Map<Integer, Boolean> map, boolean[] used, int total) {
        if (total <= 0)
            return false;       //证明上一次对手已经挑完
        
        int key = boolToInt(used);
        if (map.containsKey(key))
            return map.get(key);
        
        for (int i = 1; i < used.length; i++) {    //for一遍maxChoosableInteger
            if (!used[i])              //不能用used过的
                continue;

            used[i] = true;

            if (!dfsWin(map, used, total - i)) {        //看对手是否输
                map.put(key, true);
                used[i] = false;        //回溯
                return true;
            }

            used[i] = false;        //回溯
        }
        map.put(key, false);		//记得放
        return false;
    }
    
    public int boolToInt(boolean[] used) {
        int num = 0;
        for (boolean b : used) {
            num <<= 1;              //先向左移一位
            if (b)
                num |= 1;
        }
        return num;
    }
    
    
    
    /** Climbing Stairs
	 * 一次爬1或2步
	 * @param n
	 * @return
	 * a[i]时，可以从[i-1]走一步到；也可以从[i-2]走2步到，看他们原先有多少种走法到他们位置
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
        int[] state = new int[n + 1]; 		//现在剩i个硬币，当前取硬币的人能否赢
        return memSearch(n, state);
    }
    
    public boolean memSearch(int i, int[] state) {
        if (state[i] != 0)   return state[i] == 1;		//dp[] 为0代表空，-1为false，1为true
        
        if (i == 0) {		//现在没coin了，所以当前player输了
        	state[i] = -1;
        } else if (i == 1) {    //1 coin left for 1st player,win
        	state[i] = 1;
        } else if (i == 2) {
        	state[i] = 1;
        } else {
            if (!memSearch(i - 1, state) || !memSearch(i - 2, state)) {
            	state[i] = 1;			//之前对手不论拿1个或2个都错的话，那现在i就TRUE。
            } else {
            	state[i] = -1;
            }
        }
        
        return state[i] == 1;
    }
    
    
    /**Coins in a Line II - lintcode
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
    
    
    
    /** 486. Predict the Winner 
     * Coins in a Line III - lintcode 一样的题 
     * 每次只能从头或尾取一个，看谁的sum多
     *
     * dp[i][j]表示在i~j范围内，当前最多能取到的分数，可以去头，或尾
     * 1. 初始化dp为-1，这样就省掉visited[][]
     * 2. 如果i==j只剩一个数,那就返回n[i]
     * 3. 其他情况就dfs..分别算取start和end，哪个max取哪个
     * 		- 用nums[i] - dfs(对手), 尽可能让得分高
     * 4. 最后只要总的dfs() >= 0就行，证明先手比对手高分
     */
    public boolean PredictTheWinner(int[] nums) {
        int n = nums.length;
        int[][] dp = new int[n][n];     // 在[i,j]区间，当前人最多能有多少分
        
        for (int[] row : dp) {
            Arrays.fill(row, -1);            //初始化，-1代表没visit过
        }
        
        return dfsWin(nums, dp, 0, n - 1) >= 0;
    }
    
    public int dfsWin(int[] nums, int[][] dp, int i, int j) {
        if (dp[i][j] != -1)     
            return dp[i][j];
            
        if (i == j) {           //只剩一个数
            dp[i][j] = nums[i];
        } else {
            // 当前n[i] - dfs(对手分数) = 现在得分
            int start = nums[i] - dfsWin(nums, dp, i + 1, j);
            int end = nums[j] - dfsWin(nums, dp, i, j - 1);
            dp[i][j] = Math.max(start, end);
        }
        
        return dp[i][j];
    }
    
    
    
    /** Stone Game - Lintcode
     * 有一堆石子数组A，每个石头value是A[i]，把他们两两合并，每次合并要算上石子的value。
     * 最后合并成一堆石子，会有总的value数。最后找出minimum合并次数的sum
     * 比如石子[3,4,7,6], 如果刚开始是3,4合并成7，然后变成7，7，6. 如果7,6合并那就是7+13=20。每一步合并所用的value要加起来
     * 所以是7 + 13 + 20 = 40
     * 
     * 若刚开始合并中间的4,7，则这步用了11，变成3,11,6 -> 合并11,6成17 -> 3 + 17 = 20
     * 所以是11 + 17 + 20 = 48
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
    
    public int search(int i, int j, int[][] f, boolean[][] visit, int[][] sum) {
        if(visit[i][j])
            return f[i][j];
        
        visit[i][j] = true;
        
        if(i == j) {
            return f[i][j];
        }
        
        f[i][j] = Integer.MAX_VALUE;
        // 切分区间，要循环 r-l次
        for (int k = i; k < j; k++) {	//记得还要加上这次 区间合并的sum
            f[i][j] = Math.min(f[i][j], sum[i][j] + search(i, k, f, visit, sum) + search(k + 1, j , f, visit, sum));
            // f[i][j]跟 （sum[i][j] + dp[i][k] + dp[k+1][j]） 相比
        }
        return f[i][j];
    }


    /**
     * 843. Guess the Word - Google - random找candidate, 快，猜对的概率较低
     *
     * 给6位数，和一些wordlist.。 让你猜secret
     * 每次guess() 会返回值&位置都match的个数number of exact matches (value and position) of your guess to the secret word.
     * 只有10次猜的机会
     * Input: secret = "acckzz", wordlist = ["acckzz","ccbazz","eiowzz","abcczz"]
     *
     * 基本的策略就是：
     * 1. 选candidate
     * 2. guess.. 看有多少match
     * 3. 利用match个数来 减少candidate的wordlist.. 只找match数 一样的作为candidates
     *
     * 比较快的方法是下面这种，random找candidate，但是猜对的概率较低
     */
    public void findSecretWord(String[] wordlist, Master master) {
        Random random = new Random();
        // 循环10次
        for (int i = 0, matches = 0; i < 10; i++) {
            // random 找candidate
            int idx = random.nextInt(wordlist.length);
            String candidate = wordlist[idx];

            matches = master.guess(candidate);

            if (matches == 6)
                break;

            // shrink wordlist, 找一样macth的string作为candidate
            wordlist = shrinkWordList(wordlist, candidate, matches);
        }
    }

    /**
     * 843. Guess the Word - Google - 利用策略找candidate, 猜对概率高，慢
     *
     * 为了尽量删掉最多的词，那么选candidate时就要找出这种..
     * 两个词0 match的概率是 (25/26)^6 = 80% 很高.. 而且guess完match为0就不会是candidates
     * 那我们找跟这个guess词match为0作为candidate，这样能大大减少candidate的数量
     *
     * 找出第一个要猜的word, 这个word有 最少的 0 match，也就是说大部分 word 都和他 match 一点
     * 这样如果 guess 和 这个 word 也有 0 match 的话(大概率) ，那么一下就可以删除很多word，剩下的就是这些最少的0 match的candidates
     *
     *
     * 当一个单词和其他单词match number为0的次数越多，那么这个单词越不好，因为match number为0时我们减少搜索空间的速度最慢。
     *
     * 假如现在有无限多长度为6的单词，对于word X，和他match number为0的单词有25^6这么多个，
     * 然而和X match number为1的单词则减少到了25^5 * 6这么多个，
     * 为2时为 C(6, 2) * 25^4，以此类推，match number越大我们下一轮的搜索空间会越小，
     * 所以这里我们每一轮都挑选出当前搜索空间中和其他单词match number为0的次数最少的单词作为guess word来猜，
     * 这样minimize了每次猜词的worse case。
     */
    // 这个 尽量缩小candidates, 提高猜对的准确率，但是慢
    public void findSecretWordMoreAccurate(String[] wordlist, Master master) {
        // 循环10次
        for (int i = 0, matches = 0; i < 10; i++) {
            String candidate = findCandidate(wordlist);

            matches = master.guess(candidate);

            if (matches == 6)
                break;

            wordlist = shrinkWordList(wordlist, candidate, matches);
        }
    }


    // 因为match为0的string不会作为candidate, 而且这种情况很多.那我们找跟这个guess词match为0作为candidate，这样能大大减少candidate的数量
    // 找出第一个要猜的word, 这个word有 最少的 0 match，也就是说大部分 word 都和他 match 一点
    // 这样如果 guess 和 这个 word 也有 0 match 的话 ，那么一下就可以删除很多word，剩下的就是这些最少的0 match的candidates
    private String findCandidate(String[] wordlist) {
        Map<String, Integer> noMatchMap = new HashMap<>();

        for (String w1 : wordlist) {
            noMatchMap.put(w1, 0);

            for (String w2 : wordlist) {
                if (match(w1, w2) == 0) {
                    noMatchMap.put(w1, noMatchMap.get(w1) + 1);
                }
            }
        }
        // min match zero word
        String candidate = wordlist[0];
        int candidateCount = noMatchMap.get(candidate);

        for (String word : wordlist) {
            if (noMatchMap.get(word) < candidateCount) {
                candidate = word;
                candidateCount = noMatchMap.get(word);
            }
        }
        return candidate;
    }


    // 找到matches个数一样的，可能是candidate，减少猜的范围
    private String[] shrinkWordList(String[] wordlist, String word, int matches) {
        List<String> newList = new ArrayList<>();
        for (String w : wordlist) {
            if (match(word, w) == matches) {
                newList.add(w);
            }
        }
        return newList.toArray(new String[newList.size()]);
    }

    private int match(String a, String b) {
        int matches = 0;
        for (int i = 0; i < a.length(); i++) {
            if (a.charAt(i) == b.charAt(i)) {
                matches++;
            }
        }
        return matches;
    }


    interface Master {
        public int guess(String word);
    }

   
    
}
