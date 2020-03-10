
import java.util.*;

import javafx.util.Pair;


public class DataStructure {


    /**
     * Design HashMap
     */
    public void MyHashMap() {
        nodes = new ListNode[100001];
    }

    ListNode[] nodes;

    /** value will always be non-negative. */
    public void put(int key, int value) {
        int i = getIndex(key);
        if (nodes[i] == null) {
            nodes[i] = new ListNode(-1, -1);        // dummy node
        }

        ListNode prev = findElement(nodes[i], key);
        if (prev.next == null) {
            prev.next = new ListNode(key, value);     // new element
        } else {
            prev.next.val = value;                  // exist, just update value
        }
    }

    /** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
    public int get(int key) {
        int i = getIndex(key);
        if (nodes[i] == null) {
            return -1;
        }

        ListNode prev = findElement(nodes[i], key);
        return prev.next == null ? -1 : prev.next.val;

    }

    /** Removes the mapping of the specified value key if this map contains a mapping for the key */
    public void removeFromMap(int key) {
        int i = getIndex(key);
        if (nodes[i] == null) {
            return;
        }

        ListNode prev = findElement(nodes[i], key);
        if (prev.next != null) {
            prev.next = prev.next.next;
        }
    }

    private int getIndex(int key) {
        return Integer.hashCode(key) % nodes.length;
    }

    private ListNode findElement(ListNode bucket, int key) {
        ListNode cur = bucket;
        ListNode prev = null;

        while (cur != null && cur.key != key) {
            prev = cur;
            cur = cur.next;
        }
        return prev;
    }

    class ListNode {
        int key;
        int val;
        ListNode next;

        public ListNode(int key, int val) {
            this.key = key;
            this.val = val;
        }
    }


	/** 173. Binary Search Tree Iterator
	 * Calling next() will return the next smallest number in the BST.
	 * next() and hasNext() should run in average O(1) time and uses O(h) memory
	 * next()平均要O(1). 那n个点平均就O(1)吧..
	 * 
	 * 只能用O(h)的空间，就是说不能把整个inorder放进去。
	 * 对于每个node，下一个最小的，是右子树的最左，所以需要再往left走..
	 * 
	 * 所以刚开始先把left都放进去（也算是parent node），
     * 然后pop一个node以后，把right子树的left也push进去, 这样才能保证next会是right子树里left most最小的
	 */
	class BSTIterator {
	    private Stack<TreeNode> stack = new Stack<>();

	    public BSTIterator(TreeNode root) {
	        pushLeft(root);
	    }

	    /** @return whether we have a next smallest number */
	    public boolean hasNext() {
	        return !stack.isEmpty();
	    }

	    /** @return the next smallest number */
	    public int next() {
	        TreeNode node = stack.pop();
	        pushLeft(node.right);
	        return node.val;
	    }
	    
	    private void pushLeft(TreeNode n) {
	        while (n != null) {
	            stack.push(n);
	            n = n.left;
	        }
	    }
	}
	
	class TreeNode {
		 int val;
		 TreeNode left;
		 TreeNode right;
		 TreeNode(int x) { val = x; }
	}
	
	
	/**
     * 208. Implement Trie (Prefix Tree)
	 */
	public class Trie {
	    private TrieNode root;

	    public Trie() {
	        root = new TrieNode();
	    }

	    // Inserts a word into the trie.
	    public void insert(String word) {
	        TrieNode node = root;
            for (char c : word.toCharArray()) {
	            if (!node.containsKey(c)) {
	                node.put(c, new TrieNode());
	            }
	            node = node.get(c);     //go to next after match
	        }
	        node.setEnd();
	    }
	    
	    // Returns if the word is in the trie.
	    public boolean search(String word) {
	        TrieNode node = searchPrefix(word);
	        return node != null && node.isEnd();
	    }

	    // Returns if there is any word in the trie
	    // that starts with the given prefix.
	    public boolean startsWith(String prefix) {
	        TrieNode node = searchPrefix(prefix);
	        return node != null;
	    }

        // return TrieNode比较好，这样能知道是否为空，也能判断是不是word
        private TrieNode searchPrefix(String word) {
            TrieNode node = root;
            for (char c : word.toCharArray()) {
                if (node.children[c - 'a'] == null) {
                    return null;
                }
                node = node.children[c - 'a'];
            }
            return node;
        }
	}
	
	class TrieNode {
	    private TrieNode[] children;
	    private final int R = 26;
	    private boolean isEnd;
	    private String word;
	    
	    public TrieNode() {
	        children = new TrieNode[R];
	    }
	    
	    // 下面这些不需要
	    public boolean containsKey(char c) {
	        return children[c - 'a'] != null;
	    }
	    
	    public TrieNode get(char c) {
	        return children[c - 'a'];
	    }
	    
	    public void put(char c, TrieNode node) {
	        children[c - 'a'] = node;
	    }
	    
	    public void setEnd() {
	        isEnd = true;
	    }
	    
	    public boolean isEnd() {
	        return isEnd;
	    }
	}
	
	

    /** 211. Add and Search Word - Data structure design 字典树
     * 跟Trie树一样
     * '.' 可以代替任何数，search(".ad") -> true
     * 因为需要match '.'，那么后面就有多种可能性，那么就需要DFS来尝试不同可能性
     */
    WordNode root = new WordNode();
    
    public void addWord(String word) {
        WordNode node = root;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (node.children[c - 'a'] == null) {
                node.children[c - 'a'] = new WordNode();
            }
            node = node.children[c - 'a'];
        }
        node.isLeaf = true;
    }

    // Returns if the word is in the data structure. A word could
    // contain the dot character '.' to represent any one letter.
    public boolean search(String word) {
        return match(word.toCharArray(), 0, root);
    }
    
    public boolean match(char[] chs, int k, WordNode node) {
        if (k == chs.length) {		// 记得查
            return node.isLeaf;
        }
        
        if (chs[k] == '.') {
        	for (WordNode next : node.children) {
                if (next != null && match(chs, k + 1, next)) {		// recursion
                    return true;
                }
            }
        	return false;		// no next match
        }
        
        // normal case, not '.'
        if (node.children[chs[k] - 'a'] == null) {
            return false;
        }
        
        return  match(chs, k + 1, node.children[chs[k] - 'a']);
    }
    
    class WordNode{
        boolean isLeaf;
        WordNode[] children = new WordNode[26];
        String word;		// whole word
    }
    
    
    /**
     * 676. Implement Magic Dictionary
     * 只返回 replace一次char的情况（neighbor），一样same的都不行
     * 
     *  Input: buildDict(["hello", "leetcode"]), Output: Null
		Input: search("hello"), Output: False
		Input: search("hhllo"), Output: True
		Input: search("hell"), Output: False
		Input: search("leetcoded"), Output: False
     * 
     * 不要存neighbor，否则会太大。。 
     * 
     * search时，每个char变，然后for 26个字母查，所以是 O(26 * k) k是单词长度 
     */
    class MagicDictionary {
        
        TrieNode root;

        /** Initialize your data structure here. */
        public MagicDictionary() {
            root = new TrieNode();
        }
        
        /** Build a dictionary through a list of words */
        public void buildDict(String[] dict) {
            for (String s : dict) {
                addWord(s);
            }
        }
        
        private void addWord(String word) {
            TrieNode node = root;
            for (char c : word.toCharArray()) {
                if (node.children[c - 'a'] == null) {
                    node.children[c - 'a'] = new TrieNode();
                }
                node = node.children[c - 'a'];
            }
            node.isEnd = true;
        }
        
        /** Returns if there is any word in the trie that equals to the given word after modifying exactly one character *
        public boolean search(String word) {
            for (int i = 0; i < word.length(); i++) {
                if (findWithOneReplace(word, root, 0, i))
                    return true;
            }
            return false;
        }
        
        private boolean findWithOneReplace(String word, TrieNode node, int i, int changeIdx) {
            // 记得check
            if (i == word.length())
                return node.isEnd;
            
            char c = word.charAt(i);
            
            if (i == changeIdx) {
                
                for (char j = 'a'; j <= 'z'; j++) {
                    if (j == word.charAt(changeIdx)) 
                        continue;                    
                    
                    if (node.children[j - 'a'] != null && findWithOneReplace(word, node.children[j - 'a'], i + 1, changeIdx)) {
                        return true;   
                    }
                }
                return false;   // else, no '.' matched
            } else if (node.children[c - 'a'] == null) {
                return false;
            }
            
            return findWithOneReplace(word, node.children[c - 'a'], i + 1, changeIdx);
        }
        */
        

        // word每个位置试着换26个字母，然后看这替换过一次的new word能否find到
        public boolean search(String word) {
            char[] ch = word.toCharArray();
            for (int i = 0; i < word.length(); i++) {
                char orig = ch[i];
                for (char c = 'a'; c <= 'z'; c++) {
                    if (c == orig)
                        continue;
                    
                    ch[i] = c;
                    if (find(new String(ch), root)) {
                        return true;
                    }
                }
                ch[i] = orig;   // reset to orig
            }
            return false;
        }
        
        private boolean find(String s, TrieNode root) {
            TrieNode node = root;
            for (char c : s.toCharArray()) {
                if (node.children[c - 'a'] == null) {
                    return false;
                }
                node = node.children[c - 'a'];
            }
            return node.isEnd;
        }
    }
    
    
    /** 212. Word Search II
     * words = ["oath","pea","eat","rain"] and board =["oaan","etae","ihkr","iflv"]矩阵
     * 返回["eat","oath"]
     * @param board
     * @param words
     * @return
     *
     * 可以overlap，重复用某个prefix
     * 
     * 1. 把words加到Trie里build，这样之后dfs board时就compare Trie即可，无需每次compare不同word。
     * 这样如果有相同prefix的话可以一直往下用Trie来搜，而不是call很多次dfs来重复计算
     * 
     * 2. 记得加word到list时，不能重复加，因为dfs会调用4次，所以要去重, 把node.word=null
     * 3. How do we instantly know the current character is invalid? HashMap?
		  How do we instantly know what's the next valid character? LinkedList?
		  But the next character can be chosen from a list of characters. "Mutil-LinkedList"?
		Combing them, Trie is the natural choice
	 * 为何用Trie树? 普通DFS每次都重新搜后面的单词，慢。
	 * 如果prefix一样的话，比如"aa", "aab", "aac"，那Trie树就能继续往下搜，而不用从头"a"开始
	 * 
	 * 4. 重复4次加list的话，用Trie里面一个word变量来存，如果还没isEnd, 那就默认null
     */
    public List<String> findWords(char[][] board, String[] words) {
        List<String> list = new ArrayList<>();
        if (board == null || board.length == 0 || words == null || words.length == 0)
            return list;
        
        WordNode root = buildTrie(words);
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                dfs(board, list, root, i, j);
            }
        }
        return list;
    }

    private void search(char[][] board, boolean[][] visited, int x, int y, TrieNode node, List<String> result) {
        if (!inBound(x, y) || visited[x][y] || node.children[board[x][y] - 'a'] == null)
            return;

        node = node.children[board[x][y] - 'a'];

        visited[x][y] = true;

        if (node.word != null) {             // found word
            result.add(node.word);
            node.word = null;           // 这样下次不会重复算这个
        }

        search(board, visited, x - 1, y, node, result);
        search(board, visited, x + 1, y, node, result);
        search(board, visited, x, y - 1, node, result);
        search(board, visited, x, y + 1, node, result);

        visited[x][y] = false;          // 记得backtrack
    }


    // 这个比 dfs_solution的快点
    public void dfs(char[][] board, List<String> list, WordNode node, int i, int j) {
    	char c = board[i][j];
        //  visisted ||   not in dict, doesn't match
        if (c == '#' || node.children[c - 'a'] == null)
            return;
        
        node = node.children[c -'a'];    //match, 所以node移到下一个
        								//刚开始是root，没有ch，所以要移到下一个才开始有字
        if (node.word != null) {
            list.add(node.word);
            node.word = null;       //prevent duplicate
        }
        
        board[i][j] = '#';

        if (i > 0) dfs(board, list, node, i - 1, j); 
        if (j > 0) dfs(board, list, node, i, j - 1);
        if (i < board.length - 1) dfs(board, list, node, i + 1, j); 
        if (j < board[0].length - 1) dfs(board, list, node, i, j + 1); 
        
        board[i][j] = c;		//记得回溯
    }
    
    public WordNode buildTrie(String[] words) {
    	WordNode root = new WordNode();
        for (String w : words) {
        	WordNode node = root;
            for (char c : w.toCharArray()) {
                if (node.children[c - 'a'] == null) {
                    node.children[c - 'a'] = new WordNode();
                }
                node = node.children[c - 'a'];
            }
            node.word = w;      //at the end(leaf), add whole word instead of just isEnd
        }
        return root;
    }
    
    
    Set<String> res = new HashSet<String>();
    
    public List<String> findWords2(char[][] board, String[] words) {
        Trie trie = new Trie();
        for (String word : words) {
            trie.insert(word);			//用java 自带的trie
        }
        
        int m = board.length;
        int n = board[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dfs(board, visited, "", i, j, trie);
            }
        }
        
        return new ArrayList<>(res);
    }
    
    public void dfs(char[][] board, boolean[][] visited, String str, int x, int y, Trie trie) {
        if (x < 0 || x >= board.length || y < 0 || y >= board[0].length) return;
        if (visited[x][y]) return;
        
        str += board[x][y];
        if (!trie.startsWith(str)) return;
        
        if (trie.search(str)) {
            res.add(str);
        }
        
        visited[x][y] = true;
        dfs(board, visited, str, x - 1, y, trie);
        dfs(board, visited, str, x + 1, y, trie);
        dfs(board, visited, str, x, y - 1, trie);
        dfs(board, visited, str, x, y + 1, trie);
        visited[x][y] = false;
    }


    /**
     * 在board上找最多单词个数，不能重复用某个单词的letter.. 也就是不能有overlap
     */
    private static final int[] DIR = {0, -1, 0, 1, 0};
    private int rows;
    private int cols;
    private TrieNode root1;
    private int max;
    private List<String> maxList;

    // board上能组成的最多max 单词数, 不能reuse已经用过的单词某个letter
    public int findMaxWords(char[][] board, String[] words) {
        if (board == null || board.length == 0 || board[0].length == 0 || words == null)
            return 0;

        max = 0;
        rows = board.length;
        cols = board[0].length;
        root1 = new TrieNode();

        // build trie tree based on words
        for (String word : words) {
            addWord(word);
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                search(board, new boolean[rows][cols], i, j, new ArrayList<>(), root1);
            }
        }

        return max;
    }

    private void search(char[][] board, boolean[][] visited, int x, int y, List<String> wordList, TrieNode node) {
        if (!inBound(x, y) || visited[x][y] || node.children[board[x][y] - 'a'] == null)
            return;

        node = node.children[board[x][y] - 'a'];
        visited[x][y] = true;

        if (node.word != null) {             // found word
            wordList.add(node.word);
            if (max < wordList.size()) {
                max = wordList.size();
                maxList = new ArrayList<>(wordList);
            }

            // search for other place
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    search(board, visited, i, j, wordList, root1);       // 从root和(0,0)开始
                }
            }

            // backtrack
            wordList.remove(wordList.size() - 1);

            return;
        }

        // 正常4个方向搜词
        for (int d = 0; d < 4; d++) {
            int newX = x + DIR[d];
            int newY = y + DIR[d + 1];
            search(board, visited, newX, newY, wordList, node);
        }

        visited[x][y] = false;          // 记得backtrack
    }

    private boolean inBound(int i, int j) {
        return i >= 0 && i < rows && j >= 0 && j < cols;
    }




    /** Build Segment Tree  线段树   求某段区间最大值
     * @param A
     * @return
     */
    public SegmentTreeNode build(int[] A) {
        return buildTree(A, 0, A.length - 1);
    }
    
    public SegmentTreeNode buildTree(int[] A, int start, int end) {
        if (start > end)
            return null;
        
        SegmentTreeNode node = new SegmentTreeNode(start, end, A[end]);
        
        if (start == end) {
            return node;
        }
        
        int mid = (start + end) / 2;
        node.left = buildTree(A, start, mid);
        node.right = buildTree(A, mid + 1, end);
        
        if (node.left != null && node.left.max > node.max)
            node.max = node.left.max;
        if (node.right != null && node.right.max > node.max)
            node.max = node.right.max;
        return node;
    }
    
    public class SegmentTreeNode {
    	public int start, end, max;
    	public SegmentTreeNode left, right;
    	
    	public SegmentTreeNode(int start, int end, int max) {
    		this.start = start;
    		this.end = end;
    		this.max = max;
    		this.left = this.right = null;
    	}
    }
    
    //==========用SegmentSumNode实现 segment tree，求sum的=====跟上面很像==============================
    
    class SegmentSumNode {
        int start, end;
        int sum;
        SegmentSumNode left, right;
        
        public SegmentSumNode (int start, int end) {
            this.start = start;
            this.end = end;
            this.left = null;
            this.right = null;
            this.sum = 0;
        }
    }
    
    SegmentSumNode rootss = null;
        
    public SegmentSumNode buildTree2(int[] nums, int start, int end) {
        if (start > end) {
            return null;
        }
        
        SegmentSumNode node = new SegmentSumNode(start, end);
        if (start == end) {
            node.sum = nums[start];
            return node;
        }
        
        int mid = start + (end - start) / 2;
        node.left = buildTree2(nums, start, mid);
        node.right = buildTree2(nums, mid + 1, end);
        node.sum = node.left.sum + node.right.sum;
        
        return node;
    }
    
    void update(int i, int val) {
        update(rootss, i, val);
    }
    
    // 不断缩小范围，直到start==end, 找到那个点才更新。。 同时记得把parent node也更新
    void update (SegmentSumNode node, int pos, int val) {
        if (node.start == node.end) {
            node.sum = val;					//找到要更新的点
            return;
        }
        
        int mid = node.start + (node.end - node.start) / 2;
        if (pos <= mid) {
            update(node.left, pos, val);			// recursion
        } else {
            update(node.right, pos, val);
        }
        node.sum = node.left.sum + node.right.sum;      //记得要update parent sum too
    }
    
    
    public int sumRange2(int i, int j) {
        return sumRange2(rootss, i, j);
    }
    
    
    /** 找出区间的sum，需要一步步缩小范围。
     * 1. 退出条件：找到那个node..start和node.start以及end也一样，就return node.sum
     * 2. 求mid，看区间落在哪个地方
     * 3. 如果start在mid右边，那就直接找right子树
     * 4. 如果end <= mid，那就直接找left子树
     * 5. 否则，无法判断就需要两边left和right子树都要找 (left, start, mid) + (right, mid + 1, end). 记得mid分隔
     */
    private int sumRange2(SegmentSumNode node, int start, int end) {
        if (node.start == start && node.end == end) {
            return node.sum;        // the node has same range,  just return
        }
        
        int mid = node.start + (node.end - node.start) / 2;     //this node's range mid
        if (start >= mid + 1) {
            return sumRange2(node.right, start, end);
        } else if (end <= mid) {
            return sumRange2(node.left, start, end);
        }
        return sumRange2(node.left, start, mid) + sumRange2(node.right, mid + 1, end);
    }
     
    
    
  //===========用tree[] iterative实现 segment tree，求sum的==================================
    
    int[] tree;
    int n;
    
    public void segmentSum (int[] nums) {
        if (nums.length > 0) {
            n = nums.length;
            tree = new int[n * 2];		
            buildTree(nums);
        }
    }
    
    
    /** 用tree[]建树。size是 2n, 因为后面放原数组，前面放sum
     */
    private void buildTree(int[] nums) {
        // put nums as leaf to the right half
        for (int i = n, j = 0; i < 2 * n; i++, j++) {
            tree[i] = nums[j];  
        }
        
        // sum up leaves and add to front
        for (int i = n - 1; i > 0; i--) {
            tree[i] = tree[i*2] + tree[i*2+1];
        }
    }

    void update2(int i, int val) {
        i += n;     //find i in tree
        tree[i] = val;
        
        // need to update之前的和sum
        while (i > 0) {
        	int left = i;
            int right = i;
            if (i % 2 == 0) {
                right = i + 1;
            } else {
                left = i - 1;
            }
            // parent is updated after child is updated
            tree[i / 2] = tree[left] + tree[right];
            i /= 2;
        }
    }

    public int sumRange(int l, int r) {
        int sum = 0;
        l += n;
        r += n;
        while (l <= r) {
            if (l % 2 == 1) {   //left is in the right of parent, then add own, not parent
                sum += tree[l];
                l++;
            }
            if (r % 2 == 0) {   //right is in the left of parent, then add own, not parent
                sum += tree[r];
                r--;
            }
            l /= 2;
            r /= 2;
        }
        return sum;
    }
    
    
    
    /** 304. Range Sum Query 2D - Immutable
     * 普通的prefix sum
     */
    public int sumRegion(int[][] matrix, int row1, int col1, int row2, int col2) {
        int m = matrix.length;
        if (m == 0)     return 0;
        int n = matrix[0].length;
        int[][] sums = new int[m + 1][n + 1];
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                sums[i][j] = matrix[i - 1][j - 1] + sums[i - 1][j] + sums[i][j - 1] - sums[i - 1][j - 1]; 
            }
        }
        
        // 求某一块submatrix sum
        return sums[row2 + 1][col2 + 1] - sums[row2 + 1][col1] - sums[row1][col2 + 1] + sums[row1][col1];
    }
    
    
    
    /** 308. Range Sum Query 2D - Mutable
     * 有update，和求出submatrix sum 的方法。
     * @param matrix
     * 用colSums代表每列的sum..
     * 那么求submatrix sum时，就可以知道每列的m行sum是多少，再for一遍列数j就知道
     * 
     * colSums[i][j] = the sum of ( matrix[0][j], matrix[1][j], matrix[2][j],......,matrix[i - 1][j] ).
     * 
     * google followup --- 如果没有给定matrix大小，可能突然来一个很大的坐标，怎么办？  要用segment tree来建
     */
    public void rangeMatrix2D(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)  return;
        
        this.matrix = matrix;
        int m = matrix.length;
        int n = matrix[0].length;
        colSums = new int[m + 1][n];		//这里 m+1 因为多出一行dummy row，可以不用另外循环i-1的情况
        for (int i = 1; i <= m; i++) {
            for (int j = 0; j < n; j++) {
                colSums[i][j] = colSums[i - 1][j] + matrix[i - 1][j];
            }									//这个其实算是m[i][j],只因为m多了一行
        }
    }

    int[][] colSums;        // the sum of cols j
    int[][] matrix;
    
    public void update(int row, int col, int val) {
        // deal with later rows first, then update m[r][c]
        for (int i = row + 1; i < colSums.length; i++) {
            colSums[i][col] = colSums[i][col] - matrix[row][col] + val;
        }
        matrix[row][col] = val;
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        int sum = 0;
        // 只用for列数就行.. 因为colSum很容易求出这一列的m行sum是多少
        for (int j = col1; j <= col2; j++) {
            sum += colSums[row2 + 1][j] - colSums[row1][j];
        }
        return sum;
    }


    /**
     * 715. Range Module
     * addRange[left, right)
     * queryRange[left, right)
     * removeRange[left, right)
     *
     * 用TreeMap，这样是sorted有序
     *
     */
    public void RangeModule() {
        bstMap = new TreeMap<>();
    }

    TreeMap<Integer, Integer> bstMap;

    public void addRange(int left, int right) {
        if (left >= right)
            return;

        Integer start = bstMap.floorKey(left);
        Integer end = bstMap.floorKey(right);

        // 也可以 稍微麻烦
//        if (start == null && end == null) {
//            bstMap.put(left, right);
//        } else if (start != null && left <= bstMap.get(start)) {
//            bstMap.put(start, Math.max(bstMap.get(start), Math.max(bstMap.get(end), right)));
//        } else {
//            bstMap.put(left, Math.max(bstMap.get(end), right));
//        }

        // include
        if (start != null && left <= bstMap.get(start)) {
            left = start;
        }
        if (end != null && right < bstMap.get(end)) {
            right = bstMap.get(end);
        }
        bstMap.put(left, right);

        // 删掉中间多余的intervals
        bstMap.subMap(left, false, right, true).clear();
    }

    public void removeRange(int left, int right) {
        if (left >= right)
            return;

        Integer start = bstMap.floorKey(left);
        Integer end = bstMap.floorKey(right);

        //  !!!! 顺序不能换，因为right可能包含在start这一整个interval里
        if (end != null && right < bstMap.get(end)) {
            bstMap.put(right, bstMap.get(end));
        }
        if (start != null && left < bstMap.get(start)) {
            bstMap.put(start, left);
        }

        // 真正删掉这个range
        bstMap.subMap(left, true, right, false).clear();
    }

    // 不能用 bst.containsKey(left).. 因为有可能left在中间，所以只能floorkey
    public boolean queryRange(int left, int right) {
        Integer start = bstMap.floorKey(left);
        return start != null && bstMap.get(start) >= right;
    }

    
    
    /** 352. Data Stream as Disjoint Intervals
     * 不停地有新数进来，要不停地update区间
     * @author Yan
     * 这里我们可以用BST来插入，这样复杂度就是 O(logn). 如果用ArrayList就是O(n),慢点
     * 
     * TreeMap里(start, Interval). 用***start***来做Interval的key
     * 
     * 另外，bst.lowerKey和higherKey是严格的< 或 >, 不会==。 floor跟ceiling可以 <=, >=
     */
    public class SummaryRanges {
        TreeMap<Integer, Interval> bst;
        
        public SummaryRanges() {
            bst = new TreeMap<>();
        }
        
        public void addNum(int val) {
            if (bst.containsKey(val))       //已经有 val开始的interval了
                return;

            Integer lo = bst.lowerKey(val);		// < key, 不能==。 floor会 <= 
            Integer hi = bst.higherKey(val);	  
            
            if (hi != null && lo != null && bst.get(lo).end + 1 == val && hi ==  val + 1) {		//we can merge low & val & hi
            	bst.get(lo).end = bst.get(hi).end;
            	bst.remove(hi);
            } else if (lo != null && bst.get(lo).end + 1 >= val) {
            	bst.get(lo).end = Math.max(val, bst.get(lo).end);		//有可能val在lo的区间内（重复了）
            } else if (hi != null && hi - val == 1) {		// merge with hi
            	bst.put(val, new Interval(val, bst.get(hi).end));
            	bst.remove(hi);
            } else {					// need to create one
            	bst.put(val, new Interval(val, val));
            }
        }
        
        public List<Interval> getIntervals() {
            return new ArrayList<>(bst.values());
        }
    }
    
    public class Interval {
    	int start;
    	int end;
    	Interval() { start = 0; end = 0; }
    	Interval(int s, int e) { start = s; end = e; }
    }
    
    
    
    /** 380. Insert Delete GetRandom O(1) 
     * 随机generate 数，要求都要O(1).
     * 第一反应是用hashset, 但是在genRandom时，需要扫一遍set,这样就O(n)不符合要求
     * So... 用HashMap保存每个值的index, 另外用ArrayList放值
     * 		这样genRandom就直接random得到index, 然后再list.get(idx)就行
     * 
     * 删除val时，在map找到对应的index, 然后list里last数和index互换，然后删掉最后就行
     */
    public void RandomizedSet() {
        map1 = new HashMap<>();
        list = new ArrayList<>();
    }
    
    Map<Integer, Integer> map1;     // <val, index>
    List<Integer> list;
    Random random = new Random();

    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if (map.containsKey(val)) {
            return false;
        }
        map.put(val, list.size());
        list.add(val);
        return true;
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        if (!map.containsKey(val)) {
            return false;
        }
       // 在map找到对应的index, 然后list里last数和index互换，然后删掉最后就行
        int idx = map.get(val);
     //   if (idx < list.size() - 1) {				//如果不是最后就swap，可以省略判断
            int last = list.get(list.size() - 1);
            list.set(idx, last);
            map.put(last, idx);         //不要放错，更新last的index
    //    }
        map.remove(val);
        list.remove(list.size() - 1);
        return true;
    }
    
    
    
    /** 381. Insert Delete GetRandom O(1) - 会有duplicate的数值
     * 既然有duplicate，那就要额外用set来记录重复数的index..
     *
     * !!! 注意，这里index的list不能用list，需要set.. 这样方便找/删除 最后的index
     * （用list的话 原先想着最后总是last index, 但其实如果直接第一次swap后更新的最后idx很可能小于前面的idx，那么第二次后面挑last idx会出错）
     *
     * linkedHashSet可以快点
     * LinkedHashSet for O(1) iteration over large items. 
     * An iterator over a normal HashSet is actually O(h/n), where h is table capacity. 
     * 在删除时，记得是删除set里的index
     */
    public boolean removeII(int val) {
        if (!maps.containsKey(val))
            return false;
        
        int idx = maps.get(val).iterator().next();       //set里的第一个index
        maps.get(val).remove(idx);
        
        // 要跟最后一个swap，删掉
        if (idx < list.size() - 1) {				//要判断这个，否则会错
            int last = list.get(list.size() - 1);
            list.set(idx, last);
            
            //删掉之前map里的last，更新
            maps.get(last).remove(list.size() - 1);     //注意是删n-1
            maps.get(last).add(idx);
        }
        list.remove(list.size() - 1);       //删掉list最后一个
        if (maps.get(val).isEmpty()) {
            maps.remove(val);
        }
        
        return true;
    }
    
    Map<Integer, LinkedHashSet<Integer>> maps = new HashMap<>();

    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insertII(int val) {
        boolean contain = maps.containsKey(val);
        if (!contain) {
            maps.put(val, new LinkedHashSet<Integer>());
        }
        maps.get(val).add(list.size());
        list.add(val);
        return !contain;
    }
    
    
    /** Get a random element from the set. */
    public int getRandom() {
        return list.get(random.nextInt(list.size()));
    }
    
    
    
    /** 339. Nested List Weight Sum
     * [1,[4,[6]]], return 27. (one 1 at depth 1, one 4 at depth 2, and one 6 at depth 3; 1 + 4*2 + 6*3 = 27)
     * @param nestedList
     * @return
     */
    public int depthSum(List<NestedInteger> nestedList) {
        return dfsGetSum(nestedList, 1);
    }
    
    public int dfsGetSum(List<NestedInteger> nestedList, int depth) {
        int sum = 0;
        for (NestedInteger nest : nestedList) {
            if (nest.isInteger()) {
                sum += nest.getInteger() * depth;
            } else {
                sum += dfsGetSum(nest.getList(), depth + 1);
            }
        }
        return sum;
    }
    
    
    // iterative方法
    public int depthSumIterative(List<NestedInteger> nestedList) {
        int level = 1, sum = 0;

        // reverse 版本的话
//        int intSum = 0, totalSum = 0;   无需level

        while (nestedList.size() != 0){
            List<NestedInteger> next = new LinkedList<>();
            for (NestedInteger n : nestedList){
                if(n.isInteger())
                    sum += n.getInteger() * level;      // reverse的话  intSum += n.getInteger();
                else
                    next.addAll(n.getList());
            }
            level++;                                   // reverse的话  totalSum += intSum; 不用level
            nestedList = next;
        }
        return sum;                                     // reverse的话  最后返回  totalSum
    }
    
    

 	/** Nested List Weight Sum - 用queue BFS
 	 * @param nestedList
 	 * @return
 	 */
 	public int depthSum2(List<NestedInteger> nestedList) {
        int depth = 1;
        int sum = 0;
        Queue<NestedInteger> q = new LinkedList<>();

        // 第一次要放在q里 初始化
        for (NestedInteger ni : nestedList) {
            q.add(ni);
        }

     //II 里 reverse需要preSum存之前的sum，加多次，所以不用*depth
// 	        int preSum = 0;		
// 	        int levelSum = 0;		

        while (!q.isEmpty()) {
            int size = q.size();
// 	           levelSum = 0;			reverse里
            for (int i = 0; i < size; i++) {
                NestedInteger ni = q.poll();
                if (ni.isInteger()) {
                    sum += ni.getInteger() * depth;
             //      levelSum += ni.getInteger();   //reverseII 里直接加
                } else {
                    for (NestedInteger nn : ni.getList()) {
                        q.add(nn);
                    }
                }
            }
            depth++;

            // reverse里就加上每层。而且每次加多一遍
// 	            preSum += levelSum;
// 	            sum += preSum;
        }
        return sum;
    }

    
    
    /** 364. Nested List Weight Sum II  - 反着来
     *
     * 反过来，最里面是depth为1， 外面大点..
     * [1,[4,[6]]], return 17.  1*3 + 4*2 + 6*1 = 17
     * @param nestedList
     * @return
     *
     * Iterative 版本.. 每次把intSum加进后面的，这样就不需要乘
     */
    public int depthSumInverseIIIterative(List<NestedInteger> nestedList) {
        int intSum = 0, sum = 0;
        while (!nestedList.isEmpty()) {
            List<NestedInteger> nextLevel = new ArrayList<>();
            for (NestedInteger n : nestedList) {
                if (n.isInteger())
                    intSum += n.getInteger();
                else
                    nextLevel.addAll(n.getList());
            }
            sum += intSum;
            nestedList = nextLevel;
        }
        return sum;
    }

    // recursion版本.. intSum当做参数传进去
    public int depthSumInverseII(List<NestedInteger> nestedList) {
        return helper(nestedList, 0);
    }

    public int helper(List<NestedInteger> list, int intSum) {
        int sum = 0;
        List<NestedInteger> nextLevel = new ArrayList<>();

        for (NestedInteger n : list) {
            if (n.isInteger())
                intSum += n.getInteger();
            else
                nextLevel.addAll(n.getList());
        }
        sum += helper(nextLevel, intSum);           // 传 intSum进去

        return sum;
    }



    public int depthSumInverseII1(List<NestedInteger> nestedList) {
   //     return helper(nestedList, 0);
    	
        List<Integer> levelSums = new ArrayList<>();
        
        // 因为刚开始有n个node，不像tree一开始只有一个root，所以要for一下
        for (NestedInteger n : nestedList) {
            dfs(levelSums, n, 0);
        }
        
        // levelSum放每层的sum，顺序放，所以要从后往前处理
        int level = 1;
        int sum = 0;
        for (int i = levelSums.size() - 1; i >= 0; i--) {
            sum += levelSums.get(i) * level;
            level++;
        }
        return sum;
    }
    
    // 很像level order的DFS版
    public void dfs(List<Integer> levelSum, NestedInteger n, int depth) {
        if (depth == levelSum.size())
            levelSum.add(0);              //相当于之前new AL<>();
            
        if (n.isInteger())
            levelSum.set(depth, levelSum.get(depth) + n.getInteger());
        else {
            for (NestedInteger ni : n.getList()) {
                dfs(levelSum, ni, depth + 1);
            }
        } 
    }


    /** 364. Nested List Weight Sum II
     *
     * @param nestedList
     * @return
     *
     * 不大好，要traverse 2次
     */
    public int depthSumInverse(List<NestedInteger> nestedList) {
        // 先得到最下面是多深
        int depth = getMaxDepth(nestedList);
        return dfsGetSum(nestedList, depth);
        //跟上题很像，但需要dfs时depth - 1, 而非+1
    }
    public int getMaxDepth(List<NestedInteger> list) {
        if (list == null || list.size() == 0)
            return 0;

        int max = 0;
        for (NestedInteger n : list) {
            if (n.isInteger()) {
                max = Math.max(max, 1);
            } else {
                max = Math.max(max, getMaxDepth(n.getList()) + 1);
            }
        }
        return max;
    }

        

    class NestedIterator implements Iterator<Integer> {

        /**
         * 341. Flatten Nested List Iterator - 最简单，但不好 - O(n) space, Flatten the list in constructor
         *
         * O(n) space  初始化时在Constructor里面直接 flatten
         *
         * q = new LinkedList<>();
         * flatten(nestedList);
         *
         * 这样适合 多次next()的情况..
         *
         * 但是这个不太好：
         * 1. implement Iterator接口.. 所以通常要iterate on the fly
         * 1. 如果list很大 刚开始flatten也很占空间不好。而且万一只要几个数.. 那这个flatten就没必要
         */
        Queue<Integer> q;

        private void flatten(List<NestedInteger> list) {
            for (NestedInteger in : list) {
                if (in.isInteger()) {
                    q.offer(in.getInteger());
                } else {
                    flatten1(in.getList());
                }
            }
        }

        // 用iterator来循环  flatten
        private void flatten1(List<NestedInteger> list) {
            Iterator<NestedInteger> itr = list.iterator();

            while(itr.hasNext()) {
                NestedInteger nextInt = itr.next();

                if (nextInt.isInteger()) {
                    q.offer(nextInt.getInteger());
                } else {
                    flatten1(nextInt.getList());
                }
            }
        }

        public Integer next() {
            if (!hasNext()) {
                return null;
            }

            return q.poll();
        }

        public boolean hasNext() {
            return !q.isEmpty();
        }



        /**
         * 341. Flatten Nested List Iterator - 法2 - O(n) space, Flatten the list as you go
         *
         * 这个用了挺多space.. 后面还有更像正常的iterator的
         *
         * 用Stack<NestedInteger>做，而且是从后往前放 每层，确保stack.peek()是Integer
         *
         * 我们的hasNext()函数需要遍历栈，并进行处理
         * 如果栈顶元素是整数，直接返回true
         * 如果不是，那么移除栈顶元素，并开始遍历这个取出的list，还是从后往前压入栈，循环停止条件是栈为空，返回false

         */
        public void NestedIterator2(List<NestedInteger> nestedList) {
            flatten2(nestedList);
        }

        Stack<NestedInteger> stack = new Stack<>();

        public Integer next2() {
            if (!hasNext())         // call hasNext() 这样才能处理元素
                return null;

            return stack.pop().getInteger();            // 因为peek保证是integer, 直接pop就行
        }

        public boolean hasNext2() {
            while (!stack.isEmpty()) {                  // 用 while..直到 peek()是Integer为止
                if (stack.peek().isInteger())
                    return true;
                else
                    flatten2(stack.pop().getList());		//如果是list, 就要pop出来..flatten也是从后往前，放stack里
            }
            return false;
        }


        public void flatten2(List<NestedInteger> list) {
            for (int i = list.size() - 1; i >= 0; i--) {
                stack.push(list.get(i));
            }
        }



        /** 341. Flatten Nested List Iterator - O(h) space, Real iterator
         * [[1,1],2,[1,1]], return 1,1,2,1,1
         *
         * 这是用stack放iterator
         * 正常顺序放入iterator..
         * 当NestInteger是list时，push进list的iterator. 这样下次先处理这个list的
         */
        Stack<Iterator<NestedInteger>> itrstack = new Stack<>();
        NestedInteger nextInt;

        public NestedIterator(List<NestedInteger> nestedList) {
        	itrstack.push(nestedList.iterator());
        }

        
        public Integer next3() {
            Integer result = nextInt == null ? null : nextInt.getInteger();

            nextInt = null;         // 记得设null，连续call几次haxNext()的话会跳过Integer

            return result;
        }

        public boolean hasNext3() {
            if (nextInt != null)        // 如果已经是Integer那就不往后走了，否则会skip掉Integer
                return true;

            while (!itrstack.isEmpty()) {
                if (!itrstack.peek().hasNext()) {
                	itrstack.pop();        //iterator后面没东西，那就把这个iterator pop掉
                } else { 
                    nextInt = itrstack.peek().next();
                    if (nextInt.isInteger())
                        return true;
                    else 
                    	itrstack.push(nextInt.getList().iterator());
                }
            }
            return false;
        }



		@Override
		public void remove() {
			// TODO Auto-generated method stub
			
		}
    }
        
    
    
    
    /** 251. Flatten 2D Vector
     * @param vec2d
     * 可以用iterator, 也可以直接int i,j表示行/列
     */
    public void Vector2D(List<List<Integer>> vec2d) {
   //     x = vec2d.iterator();
        vec = vec2d;
        row = 0;
        col = 0;
    }
    
    List<List<Integer>> vec;
    int row, col;
    
    public Integer next3() {
        return vec.get(row).get(col++);
    }
    
    public boolean hasNext3() {
        while (row < vec.size()) {
            if (col < vec.get(row).size()) {
                return true;
            } else {
                col = 0;
                row++;
            }
        }
        return false;
    }

    
    /** 用iterator
     */
    public Integer next4() {
        hasNext4();
        return y.next();
    }

    Iterator<List<Integer>> x;
    Iterator<Integer> y;
    
    public boolean hasNext4() {
        while ((y == null || !y.hasNext()) && x.hasNext()) {
            y = x.next().iterator();
        }
        return y != null && y.hasNext();
    }
    
    
    class NestedInteger {
    	
    	public NestedInteger() { }
    	
    	public NestedInteger(int n) { }
    	
    	public Integer getInteger() { return (Integer) 0; }
    	
    	public boolean isInteger() { return true;}
    	
    	public void add(NestedInteger ni) {};
    	
    	public List<NestedInteger> getList() {
    		return new ArrayList<NestedInteger>();
    	}
    	
    }




    /** 385. Mini Parser
     * 给的string是"[123,[456,[789]]]"，返回NestedInteger
     * 跟536. Construct Binary Tree from String很像
     * @param s
     * @return
     * 用stack放。stack的peek总是上一层的parent
     * 当'[', 就要新建一个空的ni, 然后加到peek那里，并且push进stack，因为[ 表示有嵌套
     * 当','和']'，表示结束了，那么可以把前面的num生成ni, 加到peek父亲里
     * 		']'的话就pop，说明这一层已经完了
     */
    public NestedInteger deserialize(String s) {
        if (s == null || s.length() == 0)
            return new NestedInteger();

        if (s.charAt(0) != '[')           // only integer
            return new NestedInteger(Integer.parseInt(s));

        Stack<NestedInteger> stack = new Stack<>();
        NestedInteger res = new NestedInteger();
        stack.push(res);

        int start = 1;
        for (int i = 1; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '[') {
                NestedInteger ni = new NestedInteger();
                stack.peek().add(ni);           //加到上一层parent
                stack.push(ni);
                start = i + 1;
            } else if (c == ',' || c == ']') {      //结束，那前面是num
                if (i > start) {
                    int num = Integer.parseInt(s.substring(start, i));
                    stack.peek().add(new NestedInteger(num));       //加到上一层parent
                }
                if (c == ']') {
                    stack.pop();    //这层孩子结束
                }
                start = i + 1;
            }
        }
        return res;
    }


    /** 385. Mini Parser - recursion
     * 给的string是"[123,[456,[789]]]"，返回NestedInteger
     * @param s
     * @return
     * 也是用count来数'['和']'. 如果配对了，就说明这个“子树”node完成, 就可以加到root里
     */
    public NestedInteger deserialize1(String s) {
        NestedInteger result = new NestedInteger();
        if (s == null || s.length() == 0)
            return result;

        if (s.charAt(0) != '[')           // only integer
            return new NestedInteger(Integer.parseInt(s));

        if (s.length() <= 2)        // only "[]", empty inside
            return result;

        // 这的string被包在[]里
        int start = 1, count = 0;
        for (int i = 1; i < s.length(); i++) {
            char c = s.charAt(i);
            if (count == 0 && (c == ',' || i == s.length() - 1)) {      //记得考虑i为最后一个数字的情况
                result.add(deserialize(s.substring(start, i)));
                start = i + 1;
            } else if (c == '[') {          //括号++，--要放在后面，否则"[[]]"只会输出[]，漏掉最外面那个
                count++;
            } else if (c == ']') {
                count--;
            }
        }
        return result;
    }

    
    
    /** 379. Design Phone Directory
     * 数字从0，1，2...开始累加，directory总共不超过maxNumbers个数
     * 每次get()就用 之前release过的，否则就最后的数+1
     * @param maxNumbers
     */
    public void PhoneDirectory(int maxNumbers) {
        max = maxNumbers;
        candiQ = new LinkedList<>();  
        usedSet = new HashSet<>();
    }
    
//    int max;
    Queue<Integer> candiQ;          //之前被release的，会放在这，到时优先用它
    Set<Integer> usedSet;
    
    
    /** Provide a number which is not assigned to anyone.
        @return - Return an available number. Return -1 if none is available. */
    public int get() {
        if (usedSet.size() == max) {
            return -1;
        }
        
        // 其实这里不需要last记录最后set出现的数，因为都是先用q..知道q空了，那么直接用set.size()就是last
        int cand = !candiQ.isEmpty() ? candiQ.poll() : usedSet.size();
        usedSet.add(cand);
        return cand;
    }
    
    /** Check if a number is available or not. */
    public boolean check(int number) {
        return !usedSet.contains(number);
    }
    
    /** Recycle or release a number. */
    public void release(int number) {
        if (usedSet.remove(number))			//如果set不存在就return
            candiQ.offer(number);
        
        // if (number == last) {
        //     last--;			//不需要
        // } else {
        //     candiQ.offer(number);
        // }
    }
    
    
    /** 170. Two Sum III - Data structure design
     * 这里add是O(1). find需要for循环所有，是O(n)
     * 注意如果 key == val-key, 那么需要map里面key出现过2次或以上才行
     */
    public void TwoSum() {
        map = new HashMap<Integer, Integer>();
    }

    Map<Integer, Integer> map;
    
    /** Add the number to an internal data structure.. */
    public void add(int number) {
        if (map.containsKey(number)) {
            map.put(number, 2);
        } else {
            map.put(number, 1);
        }
    }
    
    /** Find if there exists any pair of numbers which sum is equal to the value. */
    public boolean find(int value) {
        for (int key : map.keySet()) {
            int num = value - key;
            if (map.containsKey(num) && (num != key || map.get(key) > 1)) {
                return true;
            }
        }
        return false;
    }
    
    
    
    /** 170. Two Sum III - Data structure design
     *
     * 这里add是O(n).  慢， 很不适合大数据量
     * 因为要预处理，每加一个数，就要for一遍num来循环更新sum
     */
    public void TwoSum2(){
        sums = new HashSet<Integer>();
        num = new HashSet<Integer>();

    }
    
    Set<Integer> sums;
    Set<Integer> num;
    
    // Add the number to an internal data structure.
	public void add2(int number) {
	    for (int n : num) {				//注意是num !!
            sums.add(n + number);
        }
        num.add(number);
	}

    // Find if there exists any pair of numbers which sum is equal to the value.
	public boolean find2(int value) {
	    return sums.contains(value);
	}


    /**
     * 346. Moving Average from Data Stream
     * @param size
     */
    public void MovingAverage(int size) {
        nums = new LinkedList<>();
        this.size = size;
        sum = 0;
    }

    Deque<Integer> nums;
    int size;
    int sum;

    public double next(int val) {
        sum += val;
        nums.addLast(val);

        if (nums.size() > size) {
            sum -= nums.pollFirst();
        }
        return (double) sum / nums.size() ;
    }



    /**
     * 981. Time Based Key-Value Store
     * 存有timestmap..
     * get(string key, int timestamp)需要返回最近一次的set(key, value, timestamp_prev)且 timestamp_prev <= timestamp
     * 如果上一次的timestamp都大于当前，那就返回 ""
     *
     * ["TimeMap","set","set","get","get","get","get","get"],
     * inputs = [[],["love","high",10],["love","low",20],["love",5],["love",10],["love",15],["love",20],["love",25]]
     * Output: [null,null,null,"","high","high","low","low"]
     * 第一个get的timestamp是5，但是都小于 prevTime.. 所以返回""
     *
     * 直接用map存，value是一个list，带有timestamp和value的
     * get的时候 根据timestamp来 binary search 尽快找到 最近的 <= timestamp的值
     *
     *  O(1) for set(), and O(logN) for get()
     */
    public void TimeMap() {
        timeMap = new HashMap<>();
    }

    Map<String, List<Pair<Integer, String>>> timeMap;

    public void set(String key, String value, int timestamp) {
        if (!timeMap.containsKey(key)) {
            timeMap.put(key, new ArrayList<>());
        }
        timeMap.get(key).add(new Pair(timestamp, value));
    }

    public String get(String key, int timestamp) {
        if (!timeMap.containsKey(key))
            return "";

        return binarySearch(timeMap.get(key), timestamp);
    }

    private String binarySearch(List<Pair<Integer, String>> list, int time) {
        int lo = 0, hi = list.size() - 1;

        while (lo < hi) {
            int mid = lo + (hi - lo) / 2 + 1;     // 注意 + 1，因为lo = mid..不加1的话会靠左死循环
            if (list.get(mid).getKey() <= time) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }

        // 记得要判断 lo <= time.. 有可能lo > time 那就无解
        return list.get(lo).getKey() <= time ? list.get(lo).getValue() : "";
    }


    /**
     * 981. Time Based Key-Value Store
     *
     * 对应的value不用list<Pair>， 而是用 TreeMap存.. key是timestamp，这样能迅速找到floorKey
     *
     * 但是 set()需要 O(logn)  因为 TreeMap 要balance
     */
    Map<String, TreeMap<Integer, String>> bstTimeMap;

    // O(logn) TreeMap 要balance
    public void set1(String key, String value, int timestamp) {
        if (!bstTimeMap.containsKey(key)) {
            bstTimeMap.put(key, new TreeMap<>());
        }
        bstTimeMap.get(key).put(timestamp, value);
    }

    // O(logn)
    public String get1(String key, int timestamp) {
        if (!bstTimeMap.containsKey(key))
            return "";

        TreeMap<Integer, String> bst = bstTimeMap.get(key);
        Integer prevTime = bst.floorKey(timestamp);
        return prevTime == null ? "" : bst.get(prevTime);
    }






    /** 281. Zigzag Iterator
     * 可以看成按照列来取数
     *  [1,2,3]
     [4,5,6,7]
     [8,9]       返回 [1,4,8,2,5,9,3,6,7].
     * @author Yan
     * 如果给 k 个List，那就把这k个list的iterator都先放在queue里
     */
    public class ZigzagIterator {
        Queue<Iterator> queue;

        public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
            queue = new LinkedList<Iterator>();
            if (!v1.isEmpty())    queue.add(v1.iterator());
            if (!v2.isEmpty())    queue.add(v2.iterator());
        }

        public int next() {
            Iterator poll = queue.poll();
            int result = (int) poll.next();
            if (poll.hasNext()) {		// 记得如果hasNext,要再放回queue里
                queue.add(poll);
            }
            return result;
        }

        public boolean hasNext() {
            return !queue.isEmpty();
        }
    }


}
