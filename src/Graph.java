
import java.util.*;

public class Graph {
    /*
    Construct Quad Tree
    DFS O(n^2)
    */
    /*
// Definition for a QuadTree node.
class Node {
    public boolean val;
    public boolean isLeaf;
    public Node topLeft;
    public Node topRight;
    public Node bottomLeft;
    public Node bottomRight;

    
    public Node() {
        this.val = false;
        this.isLeaf = false;
        this.topLeft = null;
        this.topRight = null;
        this.bottomLeft = null;
        this.bottomRight = null;
    }
    
    public Node(boolean val, boolean isLeaf) {
        this.val = val;
        this.isLeaf = isLeaf;
        this.topLeft = null;
        this.topRight = null;
        this.bottomLeft = null;
        this.bottomRight = null;
    }
    
    public Node(boolean val, boolean isLeaf, Node topLeft, Node topRight, Node bottomLeft, Node bottomRight) {
        this.val = val;
        this.isLeaf = isLeaf;
        this.topLeft = topLeft;
        this.topRight = topRight;
        this.bottomLeft = bottomLeft;
        this.bottomRight = bottomRight;
    }
};
*/
    public Node construct(int[][] grid) {
        return helper(grid, 0, 0, grid.length);
    }
    public Node helper(int[][]grid, int x, int y, int len){
        if(len == 1) {
            return new Node(grid[x][y] != 0, true, null, null, null, null);
        }
        Node result = new Node();
        Node topLeft = helper(grid, x, y, len / 2);
        Node topRight = helper(grid, x, y + len / 2, len / 2);
        Node bottomLeft = helper(grid, x + len / 2, y, len / 2);
        Node bottomRight = helper(grid, x + len / 2, y + len / 2, len / 2);
        //To do this recusively, we have to split the grid into 4 smaller sub-grids until the sub-grid's length is 1. The sub-grid whose length is 1 is the leaf node.
        //We merge the sub-grids if all four sub-grids are leaf nodes and have same value.
        /*
        Time Complexity: O(N^2), N is the length of the grid.
        Space Complexity: O(N^2)

        Time complexity:
        Recursive equation: T(N) = 4 * T(N/2) + N, which is O(N^2)
        */
        if (topLeft.isLeaf && topRight.isLeaf && bottomLeft.isLeaf && bottomRight.isLeaf
           && topLeft.val == topRight.val && topRight.val == bottomLeft.val && bottomLeft.val == bottomRight.val) {
            result.isLeaf = true;
            result.val = topLeft.val;
        } else {
            result.topLeft = topLeft;
            result.topRight = topRight;
            result.bottomLeft = bottomLeft;
            result.bottomRight = bottomRight;
        }
        return result;
        }
    }


	/** Clone Graph UndirectedGraphNode  
	 * recursive 更简单直观
     */
    public GraphNode cloneGraphDFS(GraphNode node) {
        return cloneHelper(node, new HashMap<GraphNode, GraphNode>());
    }

    public GraphNode cloneHelper(GraphNode node, Map<GraphNode, GraphNode> cloneMap) {
        if (node == null)   return null;
        
        if (cloneMap.containsKey(node))
            return cloneMap.get(node);
        
        GraphNode clone = new GraphNode(node.label);
        cloneMap.put(node, clone);           // 记住先放进map.. 而不是for完再放
        
        for (GraphNode nei : node.neighbors) {
            clone.neighbors.add(cloneHelper(nei, cloneMap));		//这里recursive..假设已经clone好了的
        }
        return clone;
    }
    
    
	/** Clone Graph UndirectedGraphNode  
	 * 用bfs方法 
	 * 用hashmap的key存原先node, value存新nodeClone. 判断是否被clone过
     * queue中放的节点都是 original 未处理 的节点. 处理过了的会放在map里，不需要再add进去重复看
     * toVisitQ基本放了所有点
     * @param node
     * @return
     */
    public GraphNode cloneGraph(GraphNode node) {
        if (node == null) {
            return null;
        }
        Queue<GraphNode> toVisitQ = new LinkedList<GraphNode>();        // 放original
        // 放原始node 和 cloneNode
        HashMap<GraphNode, GraphNode> map = new HashMap<GraphNode, GraphNode>();
        
        GraphNode nodeClone = new GraphNode(node.label);		//复制根节点
        map.put(node, nodeClone);
        toVisitQ.offer(node);           // 都是original nodes
        
        while (!toVisitQ.isEmpty()) {
            GraphNode orig = toVisitQ.poll();
            for (GraphNode nei : orig.neighbors) {		//clone neighbor
            	if (map.containsKey(nei)) {
                    map.get(orig).neighbors.add(map.get(nei));       // 把clone neighbor加进来
                } else {
                    GraphNode neiClone = new GraphNode(nei.label);
                    map.get(orig).neighbors.add(neiClone);
                    map.put(nei, neiClone);
                    toVisitQ.offer(nei);						///未处理过得要放queue里
                }
            }
        }
        
        return nodeClone;
    }
    
    
    class GraphNode {
    	int label;
	    ArrayList<GraphNode> neighbors;

	    GraphNode(int x) {
	        label = x;
	        neighbors = new ArrayList<GraphNode>();
	    }
    }
    
    

	
	/** 130. Surrounded Regions
	 * 边界有O就保留，如果O都被X包围就变成X
     * @param board
     * 挺像 trap water + nums of island.
     *
     * 从四周外墙开始  + DFS清掉邻居
     * 1. 先从四周外墙开始dfs，找到O的就变成# ，表示不能消掉的（这样还能省略掉visit[][])
     * 2. 再循环一遍，这次的O都是在里面，可以变成X. 顺便吧 '#' 变回'O'
     * 
     * 注意dfs里就不要再检查=0, =m-1的情况，刚开始border都会遍历过。否则overflow
     */
    public void solve(char[][] board) {
        if (board.length == 0 || board[0].length == 0)  return;
        
        int m = board.length;
        int n = board[0].length;
        
     // 要从四周围墙开始, 把不能消掉的变成#.. 所以不是从任意点开始,只能从edge开始
        for (int i = 0; i < m; i++) {   
            for (int j = 0; j < n; j++) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {     // 四周墙
                    if (board[i][j] == 'O') {
                        dfs(board, i, j);
                    }
                }
            }
        }
        
     // 正常里面的'O'就是可以消掉变成X, 而#不能消，变回O
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O')     board[i][j] = 'X';
                if (board[i][j] == '#')     board[i][j] = 'O';
            }
        }
    }
    
    public void dfs(char[][] board, int i, int j) {
        int m = board.length, n = board[0].length;
        
        board[i][j] = '#';  //mark as # first, will change back to O
        for (int k = 0; k < 4; k++) {
            int x = i + d[k];
            int y = j + d[k+1];
            // 这里 > 0   < m-1  去掉=0, =m-1的情况，因为传进来时四个角都包括了，DFS就不要再检查，否则overflow
            if (x > 0 && x < m-1 && y > 0 && y < n-1 && board[x][y] == 'O') { 
                dfs(board, x, y);
            }
        }
    }
    
    
    /**
     * 323. Number of Connected Components in an Undirected Graph  - Union Find
     * Given n nodes labeled from 0 to n - 1 and a list of undirected edges 
     * Given n = 5 and edges = [[0, 1], [1, 2], [3, 4]], return 2.
     * @param n
     * @param edges
     * @return
     * 用Union Find方法 3ms
     */
    public int countComponents(int n, int[][] edges) {
    	if (edges == null || edges.length == 0 || edges[0].length == 0)     
    		return n;
    	
        // initialize parent[] (parent map)
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;			// 刚开始的parent都设为自己
        }
        
        //union
        for (int[] e : edges) {
            int node1 = find(parent, e[0]);
            int node2 = find(parent, e[1]);
            if (node1 != node2) {
                parent[node1] = node2;       //union islands
                n--;       				//这个比较smart 直接-n
            }
        }
        
        return n;
    }
    
    // 简洁，指向grandfather，循环次数少
    public int find(int[] roots, int id) {
        while(id != roots[id]) {
            roots[id] = roots[roots[id]];  // 根roots[id]指向grandfather 
            id = roots[id];
        }
        return id;
    }
    
    // 用rec 把所有节点的parent都变成root，find2的rec方法
    public int find3(int[] root, int id) {
        if (id != root[id]) {
            root[id] = find(root, root[id]);
        }
        return root[id];	//记得是root[id]，不是id.
    }
    
    
    /** 323. Number of Connected Components in an Undirected Graph - DFS方法
     * @param n
     * @param edges
     * @return
     * 1. 怎么把edge[]转化成图里的点的关系？用HashMap来存，这样就知道他们的neighbors.-->建图
     * 2. 对于每个点(这里是1~n)，一直dfs邻居，并记录visited
     *     找点的时候，这里由于是1~n, 所以就直接1~n.. 如果不是连续的，那就在map里循环for (int i : map.keySet()) 
     */
    public int countComponentsDFS(int n, int[][] edges) {
    	if (edges == null || edges.length == 0 || edges[0].length == 0)     
    		return n;
    	
    	HashMap<Integer, List<Integer>> map = new HashMap<>();   
        
        // initialize
        for (int i = 0; i < n; i++) {
            map.put(i, new ArrayList<Integer>());   
        }
        
        // put neighbors in map 把edges放map里
        for (int[] e : edges) {
            map.get(e[0]).add(e[1]);
            map.get(e[1]).add(e[0]);
        }
        
        boolean[] visited = new boolean[n];
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                count++;
                dfs(map, visited, i);
            }
        }
        return count;
    }
    
    public void dfs (Map<Integer, List<Integer>> map, boolean[] visited, int key) {
        visited[key] = true;
        for (int nei : map.get(key)) {      //scan neighbors
            if (!visited[nei]) {
                dfs(map, visited, nei);
            }
        }
    }
    
    
    /** 323. Number of Connected Components in an Undirected Graph - BFS 更慢
     * @param n
     * @param edges
     * @return
     * 前面都一样，只是把dfs(map, visited, i);那句去掉，换成BFS.
     * 每层bfs需要清空queue, 把neighbor放进去，标为visited
     */
    public int countComponentsBFS(int n, int[][] edges) {
    	if (edges == null || edges.length == 0 || edges[0].length == 0)     
    		return n;
    	
        HashMap<Integer, List<Integer>> map = new HashMap<>();   
        // initialize
        for (int i = 0; i < n; i++) {
            map.put(i, new ArrayList<Integer>());   
        }
        
        for (int[] e : edges) {
            map.get(e[0]).add(e[1]);
            map.get(e[1]).add(e[0]);
        }
        
        boolean[] visited = new boolean[n];
        int count = 0;
        
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                count++;
                visited[i] = true;		//先TRUE
                
                // bfs
                Queue<Integer> q = new LinkedList<Integer>();
                q.offer(i);
                while (!q.isEmpty()) {
                    int node = q.poll();
                    for (int nei : map.get(node)) {
                        if (!visited[nei]) {
                            q.add(nei);
                            visited[nei] = true;
                        }
                    }
                }
            }
        }
        return count;
    }
    
    
    /** 463. Island Perimeter
     * 0是水，1是地。中间的1都是连起来的。看成一个格，所以求所有1的格加起来的周长
     * @param grid
     * @return
     *
     * for循环grid，islands数++
     *
     * 同时，islands的话，还需要 看 右边 & 下面 那格是否也为1，是的话neighbor++. (记得分2次，而不是||)
     *
     * 最后周长是 islands * 4 - neighbors * 2;
     * 		因为一有n，比如2个方格，那中间internal那条边共用，这样都要-1，加起来就是8-2
     */
    public int islandPerimeter(int[][] grid) {
        int islands = 0;
        int neighbors = 0;
        
        int m = grid.length, n = grid[0].length;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    islands++;
                    if (i < m - 1 && grid[i+1][j] == 1)     neighbors++;
                    if (j < n - 1 && grid[i][j+1] == 1)     neighbors++;
                }   
            }
        }
        return islands * 4 - neighbors * 2;
    }
    
    
    
    /**200. Number of Islands - DFS
     * 1是陆地，0是水。找多少个岛屿
     * @param grid
     * @return
     */
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        
        int count = 0;
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '1') {
                    dfsDisapper(grid, i, j);    //make connected island disappear
//                    bfs(grid, i, j);      // 或者bfs

                    count++;
                }
            }
        }
        return count;
    }
    
    // dfs来消掉相邻为1的岛，变成0. 
    public void dfsDisapper(char[][] grid, int i, int j) {
        if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] == '0') {
            return;
        }
        // mark this island to 0 and disappear
        grid[i][j] = '0';
        
        dfsDisapper(grid, i-1, j);
        dfsDisapper(grid, i+1, j);
        dfsDisapper(grid, i, j-1);
        dfsDisapper(grid, i, j+1);
        
        // 可以用这个代替上面4个dfs
//        for (int k = 0; k < 4; k++) {
//            dfsDisapper(grid, i + d[k], j + d[k+1]);
//        }
    }
    
    int[] d = {0, 1, 0, -1, 0};


    private void bfs(char[][] grid, int i, int j) {
        int m = grid.length;
        int n = grid[0].length;

        grid[i][j] = '0';       // 设0

        Queue<Integer> q = new LinkedList<>();
        q.offer(i * n + j);                         // 展开成一维的index

        while (!q.isEmpty()) {
            int idx = q.poll();
            int x = idx / n;
            int y = idx % n;

            if (x > 0 && grid[x - 1][y] == '1') {
                q.offer((x - 1) * n + y);
                grid[x - 1][y] = '0';
            }
            if (x < m - 1 && grid[x + 1][y] == '1') {
                q.offer((x + 1) * n + y);
                grid[x + 1][y] = '0';
            }
            if (y > 0 && grid[x][y - 1] == '1') {
                q.offer(x * n + y - 1);
                grid[x][y - 1] = '0';
            }
            if (y < n - 1 && grid[x][y + 1] == '1') {
                q.offer(x * n + y + 1);
                grid[x][y + 1] = '0';
            }
        }

    }
    
    /**
     * 305. Number of Islands II - Union Find
     * m rows and n columns 。在positions里每个[]就addLand()，算每次操作后有几个island
     * Given m = 3, n = 3, positions = [[0,0], [0,1], [1,2], [2,1]]. 最后return [1, 1, 2, 3]
     * @param m
     * @param n
     * @param positions
     * @return
     * 并查集Union-Find
     * 1. 初始化root[] 都为-1。 count在外面初始化0
     * 2. for，每次添加island后对应的root就是id自己
     * 3. 对于每个now点，首先count++, 再查四周邻居是否能连起来
     * 	  3.1 如果root[nei]=-1就是为水，可以skip
     *    3.2 也是island的话，看root[nei] == now有没连起来？没的话要union起来，接着才count--
     *      !!!!!记得要union。有可能已经连成周围一圈，但是没union不知道是否已经连起来了，那会count--太多次导致负数
     *    
     * 因为find压缩了，可以平均为O(1), 复杂度为 初始root m*n, 循环k次（position），总复杂度是 O(mn) + O(k)
     * 
     * count理论上每次加land都要++，所以初始为0在最外面。。但如果下面相连就--
     * 
     * int now = p[0] * cols + p[1];  二维转一维
     * 
     * roots[]初始化有2种情况：
     * a. roots[]初始化为-1，不需要matrix。这样之后加land更新root以后，就不是-1了。有了这个，就知道如果neighbor是-1，就是水，那就跳过..
     * 		需要find(root,nei)的点,都是本来就是land的，所以roots[id]=roots[[roots[id]]不会越界，不会出现-1的情况..
     * b. 如果正常的有matrix的话，那么land就变为1，这样判断neighbor时看是不是1就行。
     * 		所以这样的话，roots[]初始化为i本身，能判断now和nei的root是否相同
     * 
     */
    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        List<Integer> ans = new ArrayList<>();
        if (m == 0 || positions.length == 0) {
            return ans;
        }
        
        // initialize
        int[] root = new int[m * n];
        Arrays.fill(root, -1);
        
        int count = 0;					
   //     int[] d = {0, 1, 0, -1, 0};
            
        for (int[] p : positions) {
            int now = p[0] * n + p[1];
            if (root[now] == -1) {          // 防止新加的pos跟之前一样 多count一次
                root[now] = now;    //update the now's root
                count++;                //count理论上每次加land都要++，但如果下面相连就--

                //check 4 neighbors
                for (int k = 0; k < 4; k++) {
                    int a = p[0] + d[k];
                    int b = p[1] + d[k + 1];
                    if (a < 0 || b < 0 || a >= m || b >= n)
                        continue;

                    int nei = a * n + b;

                    if (root[nei] == -1)        // 说明是water
                        continue;

                    int neiRoot = find(root, nei);    //因为上面now已经是island的id,所以不用再找它的root

                    // union
                    if (now != neiRoot) {
                        // 记得要union。有可能已经连成周围一圈，但是没union不知道，那会count--太多次导致负数
                        root[neiRoot] = now;
                        count--;
                    }
                }
            }
            ans.add(count);
        }
        return ans;
        
    }
    
    //这个跟上面差不多，但是多构建了一个grid，判断是否为1就有island。比上面花空间
    public List<Integer> numIslands2HasGrid(int m, int n, int[][] positions) {
        List<Integer> ans = new ArrayList<>();
        if (m == 0 || positions.length == 0) {
            return ans;
        }
        
        // initialize
        int[] root = new int[m*n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                root[i*n+j] = i * n + j;    //2d -> 1d
            }
        }
        
        int[][] grid = new int[m][n];
        int[] d = {0, 1, 0, -1, 0};
        int count = 0;
        for (int i = 0; i < positions.length; i++) {
            int x = positions[i][0];
            int y = positions[i][1];
            grid[x][y] = 1;
            count++;
            
            // check if 4 neighbors is 1
            for (int k = 0; k < 4; k++) {
                int a = x + d[k];
                int b = y + d[k+1];
                if (a < 0 || b < 0 || a >= m || b >= n) {
                    continue;       //index outof bound
                }
                if (grid[a][b] == 1) {
                    // find root
                    int nowRoot = find(root, x * n + y);
                    int neiRoot = find(root, a * n + b);
                    if (nowRoot != neiRoot) {
                        // union
                        root[nowRoot] = neiRoot;
                        count--;
                    }
                }
            }
            
            ans.add(count);
        }
        return ans;
    }


    /**
     * 695. Max Area of Island
     * return连成1的island最大面积
     * @param grid
     * @return
     */
    public int maxAreaOfIsland(int[][] grid) {
        if (grid == null || grid.length == 0)
            return 0;

        int max = 0;

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    max = Math.max(max, dfsMaxArea(grid, i, j));
                }
            }
        }
        return max;
    }

    private int dfsMaxArea(int[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length)
            return 0;

        if (grid[i][j] == 0)
            return 0;

        int area = 1;
        grid[i][j] = 0;

        area += dfsMaxArea(grid, i - 1, j);
        area += dfsMaxArea(grid, i + 1, j);
        area += dfsMaxArea(grid, i, j - 1);
        area += dfsMaxArea(grid, i, j + 1);

        return area;
    }


    /**
     * 694. Number of Distinct Islands
     * 返回形状不同的islands.. 形状一样的话不另外算
     *
     * a. 用offset来算.. 根据刚开始的点来算offset
     * b. 用direction 上下左右表示
     *
     */
    public int numDistinctIslands(int[][] grid) {
        Set<String> islands = new HashSet<>();

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    StringBuilder sb = new StringBuilder();
                    // dfsOffset(grid, i, j, i, j, sb);   // use offset
                    dfs(grid, i, j, sb, 's');      // use direction, 's' is start
                    islands.add(sb.toString());
                }
            }
        }
        return islands.size();
    }

    int[][] moves = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};

    private void dfsOffset(int[][] grid, int i0, int j0, int i, int j, StringBuilder sb) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] == 0)
            return;

        grid[i][j] = 0;

        // offset
        sb.append("[" + (i - i0) + "," + (j - j0) + "]");

        for (int[] move : moves) {
            // i0, j0 remain the same
            dfsOffset(grid, i0, j0, i + move[0], j + move[1], sb);
        }
    }

    private void dfs(int[][] grid, int i, int j, StringBuilder sb, char cur) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] == 0)
            return;

        grid[i][j] = 0;

        sb.append(cur);

        dfs(grid, i - 1, j, sb, 'u');
        dfs(grid, i + 1, j, sb, 'd');
        dfs(grid, i, j - 1, sb, 'l');
        dfs(grid, i, j + 1, sb, 'r');

        sb.append('#');
    }


    /**
     * 711. Number of Distinct Islands II
     * 如果islands形状可以 rotate或者reflect完一样的话，那也算same..
     */
    public int numDistinctIslands2(int[][] grid) {
        // this.grid = grid;
        boolean[][] seen = new boolean[grid.length][grid[0].length];
        Set<String> shapes = new HashSet<String>();

        for (int r = 0; r < grid.length; ++r) {
            for (int c = 0; c < grid[0].length; ++c) {
                List<Integer> shape = new ArrayList();
                explore(grid, seen, r, c, shape);
                if (!shape.isEmpty()) {
                    shapes.add(canonical(grid, shape));
                }
            }
        }

        return shapes.size();
    }

    private void explore(int[][] grid, boolean[][] seen, int r, int c, List<Integer> shape) {
        if (r < 0 || r >= grid.length || c < 0 || c >= grid[0].length || grid[r][c] == 0 || seen[r][c])
            return;

        seen[r][c] = true;
        shape.add(r * grid[0].length + c);

        explore(grid, seen, r + 1, c, shape);
        explore(grid, seen, r - 1, c, shape);
        explore(grid, seen, r, c + 1, shape);
        explore(grid, seen, r, c - 1, shape);
    }

    private String canonical(int[][] grid, List<Integer> shape) {
        String ans = "";
        int lift = grid.length + grid[0].length;
        int[] out = new int[shape.size()];
        int[] xs = new int[shape.size()];
        int[] ys = new int[shape.size()];

        for (int c = 0; c < 8; ++c) {
            int t = 0;
            for (int z: shape) {
                int x = z / grid[0].length;
                int y = z % grid[0].length;
                //x y, x -y, -x y, -x -y
                //y x, y -x, -y x, -y -x
                xs[t] = c<=1 ? x : c<=3 ? -x : c<=5 ? y : -y;
                ys[t++] = c<=3 ? (c%2==0 ? y : -y) : (c%2==0 ? x : -x);
            }

            int mx = xs[0], my = ys[0];
            for (int x: xs) mx = Math.min(mx, x);
            for (int y: ys) my = Math.min(my, y);

            for (int j = 0; j < shape.size(); ++j) {
                out[j] = (xs[j] - mx) * lift + (ys[j] - my);
            }
            Arrays.sort(out);
            String candidate = Arrays.toString(out);
            if (ans.compareTo(candidate) < 0) ans = candidate;
        }
        return ans;
    }
    

    /**
     * 302. Smallest Rectangle Enclosing Black Pixels
     * 0 as a white pixel and 1 as a black. 1的点都是连在一起的，所以只有一个black region
     * 找能enclose所有1的最小rectangle area
     * [  "0010",
		  "0110",
		  "0100"  ]  并且给出任意1的点，比如 (0,2). 返回最少需要area为6
     * @param image
     * @param x
     * @param y
     * @return
     * 最直接的方法就是dfs找
     * 
     * !!注意 visit过变成0，可以不用再变回来. (要变回来的基本是backtracking, 比如排列组合)
     * 
     * 这题有binary search的方法，看Solution.java (主要因为给了x,y就能用BS, 否则只能dfs)
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
    
    
    public void dfs(char[][] image, int[] bounds, int x, int y) {
        if (x < 0 || y < 0 || x >= image.length || y >= image[0].length || image[x][y] == '0')
            return;
        
        image[x][y] = '0';			//记得标为visit
        
        // 更新bounds
        if (x < bounds[0])   bounds[0] = x;
        if (x > bounds[1])   bounds[1] = x;
        if (y < bounds[2])   bounds[2] = y;
        if (y > bounds[3])   bounds[3] = y;
        
        for (int i = 0; i < 4; i++) {
            dfs(image, bounds, x + d[i], y + d[i + 1]);
        }
    }
    
    
    
    /**
     * 417. Pacific Atlantic Water Flow
     * 左上角是Pacific，右下角是Atlantic。中间矩阵的cell代表高度，求 水能流到2个海洋的 pair数
     * 高的可以流到 相同或矮的那
     *
     * 跟trapping rain water II类似
     * 要从外墙/ 海洋边缘开始搜。 否则一个个搜十字列才费劲了
     * 
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
        
        Queue<int[]> pq = new LinkedList<>();		//只有bfs才需要
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
    
    public void dfs(int[][] matrix, boolean[][] visited, int i, int j) {
        int m = matrix.length;
        int n = matrix[0].length;
        visited[i][j] = true;
        for (int k = 0; k < 4; k++) {
            int x = i + d[k];
            int y = j + d[k+1];						//新的邻居m[x][y]要比cur[i][j]高，这样[x][y]才可以往外流
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
        
//        PriorityQueue<Cell> minHeap = new PriorityQueue<Cell>(1, new Comparator<Cell>() {
//            public int compare (Cell a, Cell b) {
//                return a.h - b.h;
//            }
//        });
        PriorityQueue<Cell> minHeap = new PriorityQueue<>((a, b) -> a.h - b.h);
        
        int m = heights.length;
        int n = heights[0].length;
        boolean[][] visited = new boolean[m][n];
        // 要用visited[][], 不能再Cell类里加visited值，否则while里找4周时判断不了是否visited，因为没建Cell
        
      //left - right 建外围墙
        for (int i = 0; i < n; i++) {
            minHeap.offer(new Cell(0, i, heights[0][i]));
            minHeap.offer(new Cell(m-1, i, heights[m-1][i]));
            visited[0][i] = true;
            visited[m-1][i] = true;
        }
        
        //top - bottom 建外围墙
        for (int j = 0; j < m; j++) {
            minHeap.offer(new Cell(j, 0, heights[j][0]));
            minHeap.offer(new Cell(j, n-1, heights[j][n-1]));
            visited[j][0] = true;
            visited[j][n-1] = true;
        }
        
        int[] d = {0, 1, 0, -1, 0};
        int water = 0;
        
        // BFS, 每次Poll出来最low的墙
        while (!minHeap.isEmpty()) {
            Cell cell = minHeap.poll();   //cell作为visited的外墙里最小的，所以到时没访问的邻居若小于cell，可以直接fill到cell.h
            for (int i = 0; i < 4; i++) {
                int x = cell.x + d[i];
                int y = cell.y + d[i + 1];
                if (x >= 0 && x < m && y >= 0 && y < n && !visited[x][y]) {     //not visited
                    visited[x][y] = true;
                    water += Math.max(0, cell.h - heights[x][y]);	//若高于邻居就fill水
                    //put the new wall(higher) 更新外墙. 如果邻居矮的话要fill水，且fill到跟cell一样高。否则直接更新最高的
                    minHeap.offer(new Cell(x, y, Math.max(cell.h, heights[x][y])));
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
    
    
    
    /**
     * 261. Graph Valid Tree - 用的Union Find方法
     * 给edges数组和n，看是否是valid tree
     * @param n
     * @param edges
     * @return
     * 1. edges.len == n-1，否则就不是valid树
     * 2. 看每条edge的2个端点的root是否相同，如果==那就有环，FALSE。否则，就再union
     */
    public boolean validTree(int n, int[][] edges) {
        if (edges.length != n - 1) {
            return false;   //tree should have n-1 edges
        }
        
        int[] roots = new int[n];
        for (int i = 0; i < n; i++) {
            roots[i] = i;
        }
        
        for (int[] e : edges) {
            int a = find(roots, e[0]);
            int b = find(roots, e[1]);
            if (a == b) {       //edge两端的点的root一样，说明之前已经连过，有环
                return false;
            }
            roots[a] = b;
            
        }
        return true;
    }
    
    // 这个适用于 Arrays.fill(roots, -1);的情况 
    public int findValid(int[] roots, int id) {
        if (roots[id] == -1) {
            return id;
        }           
        return findValid(roots, roots[id]);
    }
    
    
  
    /** 261. Graph Valid Tree - dfs
     * 也可以用stack，跟BFS一样
     * @param n
     * @param edges
     * @return
     * 因为是tree，所以只从一个点开始调用一次dfs就行。
     * 否则course schedule或普通图，需要for循环来调用几次dfs
     */
    public boolean validTreeDFS(int n, int[][] edges) {
        if (edges.length != n - 1) {
            return false;   //tree should have n-1 edges
        }
        
        HashMap<Integer, List<Integer>> graph = new HashMap<>();   
        // initialize
        for (int i = 0; i < n; i++) {
        	graph.put(i, new ArrayList<Integer>());   
        }
        
        // 把edges放到HashMap里形成graph. 建图
        for (int[] e : edges) {
        	graph.get(e[0]).add(e[1]);
        	graph.get(e[1]).add(e[0]);
        }
        
        boolean[] visited = new boolean[n];
        
        // 这里只用call一次hasCycle(dfs), 因为树只有一个顶点
        if (hasCycle(graph, visited, 0, -1)) {				//DFS
            return false;
        }
        
        // see if all vertices are connected看是否所有点都访问了
        for (boolean vis : visited) {
            if (!vis) 
            	return false;
        }
        
        return true;
    }
    
    //这个比2简化点
    public boolean hasCycle(HashMap<Integer, List<Integer>> graph, boolean[] visited, int v, int parent) {
        if (visited[v]) {
            return true;
        }
        visited[v] = true;
        for (int nei : graph.get(v)) {
        	// 如果nei == parent, 那就是2->3, 3->2，因为是无向图,这样是可以的，所以OK就跳过，不会再check if后面的dfs()
        	// 而且visted[parent]是TRUE，所以nei==parent的话也是true，但这种不应该算是hasCycle..所以就跳过
            if (nei != parent && hasCycle(graph, visited, nei, v)) {
                return true;
            }
        }
        return false;
    }
    
    public boolean hasCycle2(List<List<Integer>> adjList, int v, boolean[] visited, int parent) {
        visited[v] = true;
        for (int nei : adjList.get(v)) {
            // 这个把visited[v] 放进来. 上面的方法是放到最开始
            if ((visited[nei] && parent != v) || (!visited[nei] && hasCycle2(adjList, nei, visited, v)))
                return true;
        }
        return false;
    }
    

    //BFS --- 在node放neighbor时，在neighbor的map里去掉当前node
    public boolean validTreeBFS(int n, int[][] edges) {
        if (edges.length != n - 1) {
            return false;   //tree should have n-1 edges
        }
        
        HashMap<Integer, List<Integer>> map = new HashMap<>();   
        // initialize
        for (int i = 0; i < n; i++) {
            map.put(i, new ArrayList<Integer>());   
        }
        
        // put neighbors in map
        for (int[] e : edges) {
            map.get(e[0]).add(e[1]);
            map.get(e[1]).add(e[0]);
        }
        
        boolean[] visited = new boolean[n];
        
        Queue<Integer> q = new LinkedList<>();			// 开始BFS, 跟dfs不同的地方
        q.add(0);

        while (!q.isEmpty()) {
            int node = q.poll();
            if (visited[node]) {
                return false;
            }
            visited[node] = true;
            for (int nei : map.get(node)) {
                q.offer(nei);
                map.get(nei).remove((Integer) node);    //remove current node from nei为了去掉parent
            }											// 跟dfs里 nei != parent一样跳过, 防止重复循环
        }
        
        for (boolean vis : visited) {
            if (!vis) return false;
        }
        
        return true;
    }
    
    
    
    /**207. Course Schedule  --- DFS
     * 给n个课0 ~ n-1 有些课是前提（0，1）的话，要上0的话，需要先上1
     * 
     * 主要看有没有环cycle，有的话就FALSE了
     * 
     * 根据prerequisites来建图, build edges, 这样就能dfs看这些点会不会有cycle
     * 需要for循环所有course先，因为不保证它们是connected
     * 
     * boolean[] visisted 存all看过的点.. 比如第一次看了 2->1->0, 之后还有点 3->2，就不用继续判断2->1->0了。剪枝
     * Boolean[] onpath 这次dfs path里visited过的点。记得dfs最后结束时Reset onpath[v]=false
     * 
     */
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        if (numCourses <= 0 || prerequisites == null) {
            return false;
        }
        
        Map<Integer, List<Integer>> graph = new HashMap<>();
        // initialize
        for (int i = 0; i < numCourses; i++) {
            graph.put(i, new ArrayList<>());
        }
        
        // put edges in graph
        for (int[] e : prerequisites) {
            graph.get(e[1]).add(e[0]);		//先上c[1]的课, 再上c[0]
        }
        
        boolean[] visited = new boolean[numCourses];    //all visited nodes
        boolean[] onpath = new boolean[numCourses];    //visited nodes in current dfs visit
       
        // 这要for循环，因为有n个顶点，不是所以课程都connect起来，所以要for all. 跟上面的valid tree那题不一样!!!!
        for (int i = 0; i < numCourses; i++) {
            // skip already visited nodes 这跟dfs里刚开始的判断一样，但省时
            if (!visited[i] && dfsHasCycle(graph, visited, onpath, i)) {      //dfs
        //    if (!dfs(graph, track, i)) {		//track = int[] 用一个track代表2个boolean[]
                return false;
            }
        }
        return true;
    }
    
    public boolean dfsHasCycle(Map<Integer, List<Integer>> graph, boolean[] visited, boolean[] onpath, int v) {
        if (visited[v]) {
            return false;   //这条路visited过, just skip 省时prune!!
        }
        visited[v] = onpath[v] = true;
        for (int nei : graph.get(v)) {
        	//在这次dfs path上已visit过，那就是有环
            if (onpath[nei] || dfsHasCycle(graph, visited, onpath, nei)) {
                return true;
            }
        }
        onpath[v] = false;   //记得reset onpath to false after this dfs visited
        return false;
    }
    
    // 可以用一个数组的2个状态表示all visited / visited in this dfs loop (onpath)..
    // 默认状态是0, 也是初始值..说明没visit过
    public boolean dfs(HashMap<Integer, List<Integer>> graph, int[] track, int v) {
        if (track[v] == 2)  return true;    //visited
        if (track[v] == 1)  return false;   // if already visited in this dfs loop (onpath[])
        
        track[v] = 1;       //start this dfs visit
        
        for (int nei : graph.get(v)) {
            if (!dfs(graph, track, nei)) {
                return false;
            }
        }
        track[v] = 2;   //set to visited (in all)
        // stack.push(v);	如果要打印出结果的话
        return true;
    } 
    
    
    
    
    /** 207. Course Schedule  --- BFS 拓扑排序
     * 有些课是前提（0，1）的话，要上0的话，需要先上1, 所以是1 -> 0
     * @param numCourses
     * @param prerequisites
     * @return
     *
     * 1. 找到indegree = 0 的作为顶点，放进Q里
     * 2. 每个顶点里找neighbor, 并把neighboor的indegree-1, 若nei的=0，放进q里
     * 3. 看是否q里排好的数 == 课程数  !!!! 注意查。有可能里面有些环，indegree无法减到0，不能放q里，导致错
     */
    public boolean canFinishBFS(int numCourses, int[][] prerequisites) {
        List<Integer>[] adj = new List[numCourses];
        int[] indegree = new int[numCourses];
        
        for (int[] p : prerequisites) {
            List<Integer> list = adj[p[1]];		//也可以用普通的HashMap,就是会比array慢点
            if (list == null) {
                list = new ArrayList<>();
                adj[p[1]] = list;
            }
            list.add(p[0]);
            indegree[p[0]]++;
        }
        
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {     //put courses with no indegree in q first
                q.add(i);
            }
        }
        
        int count = 0;
        while (!q.isEmpty()) {
            int v = q.poll();
            count++;
            if (adj[v] == null) continue;		//记得判断null
            for (int nei : adj[v]) {
            	// 下面2句可以合并成一句  if (--indegree[nei] == 0) {
                indegree[nei]--;
                if (indegree[nei] == 0) {
                    q.add(nei);
                }
            }
        }
        
        return count == numCourses;     // 有可能里面有些环，indegree无法减到0，不能放q里，导致错
    }
    
    
    
    /** 210. Course Schedule II - BFS
     * 输出拓扑排序结果
     * @param numCourses
     * @param prerequisites
     * @return
     *
     * !!!!!!! 记得最后要判断result的个数是否==numCourses. 如果不是的话说明有环..
     *
     * 比较general的用hashmap的方法在下面
     *
     * 除了普通的queue方法，还能用two pointer来省去queue
     * 1. 最开始找indegree=0 的顶点，放进结果的arr里，并且k++
     * 2. 设置j=0 开始，找顶点(最开始indegree=0) 的neighbor往后放
     * 3. j = k时说明有环。因为正常会有新加的课程，k会++。而j往后的速度小。 k要顺利到n后才成功
     */
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        List<Integer>[] adj = new List[numCourses];
        int[] indegree = new int[numCourses];
        
        for (int i = 0 ; i < numCourses; i++) {
            adj[i] = new ArrayList<>();
        }
        
        for (int[] p : prerequisites) {
            adj[p[1]].add(p[0]);
            indegree[p[0]]++;
        }
        
        int[] result = new int[numCourses];
        int k = 0;
  //      Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {     //put courses with no indegree in q first
       //         q.add(i);
                result[k++] = i;
            }
        }
        
        /* //用普通的queue来存  比下面的方法慢
        while (!q.isEmpty()) {
            int v = q.poll();
            result[k++] = v;
            for (int nei : adj[v]) {
                indegree[nei]--;
                if (indegree[nei] == 0) {
                    q.add(nei);
                }
            }
        }
        return k == numCourses ? result : new int[0];
        */
        
        if (k == 0) 
            return new int[0];
            
        int j = 0;      						//two pointer
        while (k < numCourses) {
            for (int nei : adj[result[j++]]) {	 //j is the earlier courses that indegree=0
                if (--indegree[nei] == 0) {
                    result[k++] = nei;
                }
            }
            if (j == k) {               // 说明有环
                return new int[0];
            }
        }
        return result;
        
    }

    // BFS. general HashMap
    public int[] findOrderBFS(int numCourses, int[][] prerequisites) {
        if (numCourses <= 0 || prerequisites == null) {
            return new int[0];
        }

        Map<Integer, List<Integer>> graph = new HashMap<>();
        int[] indegree = new int[numCourses];

        for (int[] e : prerequisites) {
            if (!graph.containsKey(e[1])) {
                graph.put(e[1], new ArrayList<>());
            }
            graph.get(e[1]).add(e[0]);
            indegree[e[0]]++;        // 先上e[1]的课，再上e[0]
        }

        Queue<Integer> q = new LinkedList<>();

        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                q.offer(i);
            }
        }

        if (q.isEmpty())        // 说明有环
            return new int[0];

        int[] result = new int[numCourses];
        int idx = 0;

        while (!q.isEmpty()) {
            int course = q.poll();
            result[idx++] = course;

            List<Integer> list = graph.get(course);
            if (list == null)
                continue;
            for (int nei : list) {
                indegree[nei]--;
                if (indegree[nei] == 0) {
                    q.add(nei);
                }
            }
        }

        return idx == numCourses ? result : new int[0];     // 有可能里面有些环，indegree无法减到0，不能放q里，导致错
    }


    /**
     * Topological Sort for dependency schedule - 拓扑排序
     * 这是比较general的.. 某个package依赖于别的，需要先build 后者.. 跟course schedule II 一样，但是更general，需要注意几点
     *
     * A: [B, C]   A depends on B & C
     * B: [C]
     * D:
     *
     * !!!!!! 需要注意的
     * a. 最后记得判断result size 是否 跟出现的package总数一样, 少于说明有cycle.. 错的
     *    有环的话 indegree不会变0放进queue..所以最后queue里剩有环的
     *
     * b. input可能没有提供完全的dependency，比如上面例子，没有C的dependency...
     *    所以需要在for dependency时 也加上dependency的indegree. 可以设成0，这样结果才会打印出C.. 否则不会打印C
     *    如果设成0，那么会先出现在最开始.. 其实可以放在任意位置也可以最后，但这时就不能设0了.. 看具体要求
     */
    public List<String> printDependency(Map<String, List<String>> dependencies) {
        List<String> result = new ArrayList<>();

        if (dependencies == null || dependencies.size() == 0) {
            return result;          // validate input and return empty result
        }

        Map<String, List<String>> graph = new HashMap<>();
        Map<String, Integer> indegree = new HashMap<>();

        // build graph
        for (Map.Entry<String, List<String>> entry : dependencies.entrySet()) {
            String str = entry.getKey();
            List<String> list = entry.getValue();
            for (String dependency : list) {
                if (!graph.containsKey(dependency)) {
                    graph.put(dependency, new ArrayList<>());
                }
                graph.get(dependency).add(str);    // B -> A

                // 同时，如果input里没出现C的dependency的话，也要把这个C的indegree设成0，否则之后会漏掉它
                // 不过设0的话，这个会首先出现.. 其实可以出现在最后或者任意位置.. 这时看面试官想要什么
                if (!indegree.containsKey(dependency)) {
                    indegree.put(dependency, 0);
                }
            }

            indegree.put(str, indegree.getOrDefault(str, 0) + list.size());
        }

        System.out.println("indegree  " + indegree);
        System.out.println("graph  " + graph);

        Queue<String> q = new LinkedList<>();

        // find the elements with 0 indegree
        for (Map.Entry<String, Integer> entry : indegree.entrySet()) {
            if (entry.getValue() == 0) {
                q.add(entry.getKey());
            }
        }

        // iterate through the graph and output the result in order
        while (!q.isEmpty()) {
            String cur = q.poll();
            result.add(cur);

            if (graph.containsKey(cur)) {
                for (String neighbor : graph.get(cur)) {
                    int count = indegree.get(neighbor);
                    indegree.put(neighbor, count - 1);
                    if (count - 1 == 0) {
                        q.offer(neighbor);
                    }
                }
            }
        }

        // 记得最后这个  要check result  size...!!!!!   有环的话 indegree不会变0放进queue..所以最后queue里剩有环的
        return result.size() < indegree.size() ? new ArrayList<>() : result;

        // 怎么写unit test.. 如何让 result predictable... 我说treeMap... 他说python有 OrderDict.. 可能也是treemap
    }


    /** 210. Course Schedule II - DFS
     * @param numCourses
     * @param prerequisites
     * @return
     *
     * 用stack存value, 因为最先进去的是最后面的叶子（没有depend的课了，不需要上别的课）
     * 最晚放stack其实是最先在result里要上的课.。 indegree为0
     */
    public int[] findOrderDFS(int numCourses, int[][] prerequisites) {
        List<Integer>[] adj = new List[numCourses];
        
        for (int i = 0 ; i < numCourses; i++) {
            adj[i] = new ArrayList<>();
        }
        
        for (int[] p : prerequisites) {
            adj[p[1]].add(p[0]);
        }
        
        boolean[] visited = new boolean[numCourses];    //all visited nodes
        boolean[] onpath = new boolean[numCourses];    //visited nodes in current dfs visit
        Stack<Integer> s = new Stack<>();
        // 也可以用 int[] track, 然后 if (!dfs(adj, track, s, i)) 更快
        for (int i = 0; i < numCourses; i++) {
            if (!visited[i] && !dfs2(adj, visited, onpath, s, i)) { 
                return new int[0];
            }
        }
        int[] result = new int[numCourses];
        for (int i = 0; i < numCourses; i++) {
            result[i] = s.pop();
        }
        return result;
    }
    
    public boolean dfs2(List<Integer>[] adj, boolean[] visited, boolean[] onpath, Stack<Integer> s, int v) {
    	if (onpath[v])     return false;
        if (visited[v])     return true;
        
        visited[v] = onpath[v] = true;
        for (int nei : adj[v]) {
            if (!dfs2(adj, visited, onpath, s, nei)) {
                return false;
            }
        }
        onpath[v] = false;

        s.push(v);           // 放结果，最开始先是把没有neighbor/别的dependency的放进去，也就是indegree为0. 后面才放depend on它的课

        return true;
    }
    
    
    
    /** 269. Alien Dictionary
     * words are sorted lexicographically
     *   "wrt",
		  "wrf",
		  "er",
		  "ett",
		  "rftt"  返回 "wertf"  . 如果是"z","x","z" 那就是空，因为顺序错了
     * @param words
     * @return
     * 拓扑排序.. 跟course schedule II 很像
     *
     * 1. for循环，把出现的char的indegree记为0，并count++ 算出现的char个数
     * 2. 建关系图。 for整个words, 上下两个词对比 & 建adj关系表   只比到char不一样的地方
     *      2.1 如果adj有环，有错 ""
     *      2.2 把c2加到c1的邻接表里，并且c2 indegree++
     * 3. 把indegree为0的放到queue里
     * 4. while q，加到result里。 只当邻居的indegree为0 时才放q.. 否则只是indegree--
     *
     * !!!!!!!!  上下垂直对比  !!!!!!!!!!!
     * 建立indegree表，记得顺便count一下总共几个char
     * 
     * 同时初始化出现的char的indegree为0. 提前初始化0，不要在后面再初始化，否则 >minLen的char会被忽略
     * 
     * ！！注意如果 两个char不相等，就要开始比较，而且只比较一次!!!!!
     * 只有当char前面是相同的时候，才能进行这一对char的比较
     * za, zb, ca 前面z时a,b已经算过.. zb和ca 比a,b的话会出错
     * 
     * 如果c2之前出现在c1的set里，那就不要重复加否则indegree会多加一次
     * 
     * 记得最后看 sb 是否跟count 一样。有可能minLen后面的有环，没算，也没加进q
     */
    public String alienOrder(String[] words) {
        if (words == null || words.length == 0)
            return "";
        
        Map<Character, Set<Character>> map = new HashMap<>();
        int[] indegree = new int[26];
        Arrays.fill(indegree, -1);
        int count = 0;
        
        // 先把出现过的indegree设为0. 不要在后面minLen之类的加，否则 >minLen的char会被忽略
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
                    if (map.containsKey(w2[j]) && map.get(w2[j]).contains(w1[j])) {  	//可有可无
                        return "";          //说明有环
                    }
                    if (!map.containsKey(w1[j])) {
                        map.put(w1[j], new HashSet<Character>());
                    }
                    // 注意不要重复加
                    if (map.get(w1[j]).add(w2[j])) {
                        indegree[w2[j] - 'a']++;
                    }
                    break;		// 记得比完一次不同的就要break，否则后面会错！！
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
        
        if (sb.length() != count)   return "";		// 有可能minLen后面的有环，没算，也没加进q
        return sb.toString();
        
    }
    
    
    
    /**
     * 310. Minimum Height Trees
     * 给n nodes which are labeled from 0 to n - 1
     * Given n = 6, edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]， 返回[3, 4] 它俩可以作为root。
     * 只能有1-2个MHT符合条件
     * @param n
     * @param edges
     * @return
     * 类似剥洋葱的方法，一层一层的褪去leaf，最后剩下的1 或 2个节点就是MHT的根节点
     * 做法类似于course schedule的拓扑排序。或者吹掉叶子那题
     * 找到degree为1的所有leaf，放进list里。每次for邻居时把node去掉（褪去leaf），剩下的degree为1再放进list里
     */
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 1) 
            return Collections.singletonList(0);	//省掉3句（建list，加0，return）
            
        //build graph
        Set<Integer>[] adj = new HashSet[n];
        for (int i = 0; i < n; i++) {
            adj[i] = new HashSet<>();
        }
        
        for (int[] e : edges) {
            adj[e[0]].add(e[1]);
            adj[e[1]].add(e[0]);

//            degree[e[0]]++;
//            degree[e[1]]++;
        }
        
        LinkedList<Integer> leavesQ = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (adj[i].size() == 1) {      //put leaf in leaves when 1
                leavesQ.add(i);
            }
        }
        
        // recursively put leaf in leaves and remove node
        while (n > 2) {
            int numLeaf = leavesQ.size();		//记得提前赋值，否则for里直接用.size()会变
            n -= numLeaf;
            // height++;				//如果求height的话，初始化为0
            for (int i = 0; i < numLeaf; i++) {
                int leaf = leavesQ.poll();
                int nei = adj[leaf].iterator().next();     //only one neighbor, because l is leaf
                adj[nei].remove(leaf);   		//每层去掉叶子 from neighbor
                if (adj[nei].size() == 1) {
                    leavesQ.add(nei);			//加新的叶子。跟拓扑排序挺像
                }

                /*  // 或者... 其实也是只有一个neighbor
                for (int oneNei : adj[leaf]) {
                    degree[oneNei]--;           // 用degree来查，而不是remove neighbor叶子
                    if (degree[oneNei] == 1) {
                        leavesQ.add(oneNei);
                    }
                }
                 */
            }
        }
        
        // height = n == 2 ? height + 1 : height;  //最后分两种情况
        
        return leavesQ;
    }
    
    
    
    /**
     * 332. Reconstruct Itinerary
     * 给机票，输出itinerary。要求城市以字母顺序排.. && 要用完手里的机票
     * tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
		Return ["JFK","ATL","JFK","SFO","ATL","SFO"].
     * @param tickets
     * @return
     * 
     * 这是Eulerian path，每个点只访问一次。用 Hierholzer's algorithm
     * 每次访问一个vertex的边，把outdegree的边去掉.. 直到所有边都visit过，再list.add(0, v)
     * 
     * Eulerian path有2种情况：
     * 1. 所有点入/出度都为1.  
     * 2. 只有一个点out - in = 1 (起点), 只有一个点in - out = 1(终点), 其他出入度都为1.. （这题就是itinerary）
     * 
     * 这题最重要的是，用完手里的机票.. 有的地方出现次数可能是odd，所以只能作为终点
     * 比如最后要JFK -> A -> C -> D -> B -> C -> JFK -> D -> A... 
     * 第一次时，JFK -> A -> C -> D -> A，然后就停了.. 但还有其他ticket没用，所以要把A先记录下来变终点
     * 第二次再走别的路...
     * 
     * 因为这个，需要dfs里面while完所有线路后（每次删掉路线），再加result(0, city)
     */
    public List<String> findItinerary(String[][] tickets) {
        List<String> result = new ArrayList<String>();
        // build graph
        Map<String, PriorityQueue<String>> graph = new HashMap<>();

        for (String[] t : tickets) {
            if (!graph.containsKey(t[0])) {
                graph.put(t[0], new PriorityQueue<String>());	// 用minHeap最好..能排序，也能删掉
            }
            graph.get(t[0]).add(t[1]);
        }
        
        dfs(graph, result, "JFK");
        Collections.reverse(result);
        
        return result;
    }
    
    public void dfs(Map<String, PriorityQueue<String>> graph, List<String> result, String start) {
        PriorityQueue<String> arrivals = graph.get(start);
        while (arrivals != null && !arrivals.isEmpty()) {
            dfs(graph, result, arrivals.poll());			// poll也删除visit过的边
        }
        result.add(start);			// 注意这里.. while完了再加到first位置
    }
    
    
    
    //iterative方法，放stack就行
    public List<String> findItineraryIterative(String[][] tickets) {
        List<String> result = new LinkedList<>();
        
        // build graph
        Map<String, PriorityQueue<String>> graph = new HashMap<>();
        for (String[] t : tickets) {
            if (!graph.containsKey(t[0])) {
                graph.put(t[0], new PriorityQueue<String>());
            }
            graph.get(t[0]).add(t[1]);
        }
        
   //     dfs(graph, result, "JFK");
        
        Stack<String> stack = new Stack<>();
        stack.push("JFK");
        while (!stack.isEmpty()) {
            String next = stack.peek();
            if (graph.containsKey(next) && !graph.get(next).isEmpty()) {
                stack.push(graph.get(next).poll());
            } else {
                result.add(tack.pop());
            }
        }
        
        return result;
    }
    
    
    
    /**
     * 286. Walls and Gates
     * 给个matrix，-1 是A wall or an obstacle.0 是 gate.。INF是空地 int max.
     * 把空地填满，里面填成distances to nearest gate.  无法到gate的还是保留成INF
     * @param rooms
     * 找最短距离都是用BFS.
     * 这里要注意的是，从GATE 0 出发！！！ 而不是空地，否则会超时。因为GATE远小于空地
     *
     * 因为先把GATE 0都push进去了.. 所以其实每次都是 等所有GATE周围visit完后再进行下一层的
     */
    public void wallsAndGates(int[][] rooms) {
        Queue<Integer> qx = new LinkedList<Integer>();
        Queue<Integer> qy = new LinkedList<Integer>();
        
        int[] dx = {0, 0, 1, -1};
        int[] dy = {1, -1, 0, 0};
        if (rooms.length == 0) {
            return;
        }
        
        
        for (int i = 0; i < rooms.length; i++) {
            for (int j = 0; j < rooms[0].length; j++){
                if (rooms[i][j] == 0){
                    qx.offer(i);
                    qy.offer(j);
                }
            }
        }
        
        while(!qx.isEmpty()) {
            int nx = qx.poll();
            int ny = qy.poll();
            
            for (int i = 0; i < 4; i++) {
                int newx = nx + dx[i];
                int newy = ny + dy[i];
                
                if (newx >= 0 && newy >= 0 && newx < rooms.length && newy < rooms[0].length 
                   && rooms[newx][newy] == Integer.MAX_VALUE) {
                    qx.offer(newx);
                    qy.offer(newy);
                    rooms[newx][newy] = rooms[nx][ny] + 1;
                }
            }
        }
        
    }
   }
    
    /**
     * 317. Shortest Distance from All Buildings
     * 0是空地，1是building，不能pass。2是障碍物也不能pass
     * @param grid
     * @return
     * 这题跟上面的wall & gates很像。也是要从building出发找，否则太慢
     * 
     * int[][] distance记录每个bld到其他空地的距离
     * for循环一遍所有bld, distance除了加这一层的level，还要在上一次的bld算出来之上加
     * int[][] canReach 查每个空地是否能到bld..有时可能被阻碍到，不能到。所以需要看canReach是否== blds个数
     *  
     * 每个bld扫时，需要新建一个visit[][]防止重复访问。记得要新建k次（bld数）
     * 
     * 时间复杂度 O(k*m*n)
     *
     * 下面有 省空间的优化
     */
    public int shortestDistance(int[][] grid) {
        if (grid == null || grid.length == 0)   return -1;
        
        int m = grid.length;
        int n = grid[0].length;
        int[] d = new int[] {0, -1, 0, 1, 0};
        int[][] distance = new int[m][n];
        int[][] canReach = new int[m][n];   //check if empty place can be reach to every building
        int bldNum = 0;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {       //is building
                    bldNum++;
                    boolean[][] visited = new boolean[m][n];        //每个bld都要建新的visit
                    Queue<int[]> q = new LinkedList<>();
                    q.add(new int[]{i,j});
                    int level = 1;
                    
                    while (!q.isEmpty()) {
                        int qsize = q.size();
                        for (int t = 0; t < qsize; t++) {  		//记得在这for一下qsize.这样最后level++才对    
                            int[] p = q.poll();
                            int x = p[0];
                            int y = p[1];
                            visited[x][y] = true;
                        
                            for (int k = 0; k < 4; k++) {
                                int r = x + d[k];
                                int c = y + d[k+1];
                                if (r < 0 || c < 0 || r >= m || c >= n || grid[r][c] != 0 || visited[r][c]) {
                                    continue;			
                                }
                                distance[r][c] += level;		//还要加上之前bld的值
                                canReach[r][c]++;           //finally should == bldNum
                                q.add(new int[]{r,c});
                                visited[r][c] = true;		//记得四周也要变visited
                            }
                        }
                        level++;		//每层完了再++level
                    }
                }
            }
        }
        
        
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0 && canReach[i][j] == bldNum) {   //要保证空地能到达所有blds, 所以要记录bldNum
                    min = Math.min(min, distance[i][j]);
                }
            }
        }
            
        return min == Integer.MAX_VALUE ? -1 : min;          
    }
    
    
    /**
     * 317. Shortest Distance from All Buildings
     * 0是空地，1是building，不能pass。2是障碍物也不能pass
     * @param grid
     * @return
     * 是上面的优化。
     * 
     * 合并 空地 && visit
     * 1. 判断空地。把空地从0变成-1，变成-2这样，每次换bld时walk--. 而且邻居grid[][]-- . 所以只要grid[i][j]==walk，就证明是空地。
     * 2. 不用visit[][]。因为会把grid[i][j]-- 从0变成-1. 但对于这个bld来说空地还是用0. 所以就略去了visited过的
     * 
     * 下一个方法是把中间一大段BFS抽出来，更清楚
     */
    public int shortestDistanceBetter(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[] d = new int[] {0, -1, 0, 1, 0};
        int[][] distance = new int[m][n];
        
        int walk = 0;   //empty place
        int min = -1;
         
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {       //is building
                    Queue<int[]> q = new LinkedList<>();
                    q.add(new int[]{i,j});
                    int level = 1;
                    min = -1;				//每次min要重新置-1
                    
                    while (!q.isEmpty()) {
                         int qsize = q.size();
                         for (int t = 0; t < qsize; t++) {       
                            int[] p = q.poll();
                            int x = p[0];
                            int y = p[1];
                        
                            for (int k = 0; k < 4; k++) {
                                int r = x + d[k];
                                int c = y + d[k+1];
                                			// g[r][c]==walk代表空地 && g[r][c]已经-1 变得 != walk, 所以也算visited过
                                if (r >= 0 && r < m && c >= 0 && c < n && grid[r][c] == walk) {
                                    grid[r][c]--;
                                    distance[r][c] += level;
                                    q.add(new int[]{r,c});
                                    if (min < 0 || min > distance[r][c]) {
                                        min = distance[r][c];
                                    }
                                }
                            }
                         }
                         level++;		//for完这一层后再++
                     }
                     walk--;		//每个bld结束，才--
                }
            }
        }
        return min;          
    }
    
    //单独把中间的方法抽出来BFS
    public int shortestDistanceBFS(int[][] grid) {
        if (grid == null || grid.length == 0)   return -1;
        
        int m = grid.length;
        int n = grid[0].length;
        
        int[][] distance = new int[m][n];
        int walk = 0;   //empty place
        int min = -1;
         
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {       //is building 循环k个building
                    min = bfs(grid, distance, walk--, i, j, -1);    // 每遍历完一个bld,那么walk--
                }
            }
        }
        
        return min;          
    }
    
    private int bfs(int[][] grid, int[][] distance, int walk, int i, int j, int min) {
        Queue<int[]> q = new LinkedList<>();
        q.add(new int[]{i,j});
        int level = 0;			//如果level放在开始的话，初始化要0.
        min = -1;
        
        while (!q.isEmpty()) {
             int qsize = q.size();
             level++;
             for (int t = 0; t < qsize; t++) {       
                int[] p = q.poll();
                int x = p[0];
                int y = p[1];
            
                for (int k = 0; k < 4; k++) {
                    int r = x + d[k];
                    int c = y + d[k+1];
                    if (r >= 0 && r < grid.length && c >= 0 && c < grid[0].length && grid[r][c] == walk) {
                        grid[r][c]--;
                        distance[r][c] += level;
                        q.add(new int[]{r,c});
                        if (min < 0 || min > distance[r][c]) {
                            min = distance[r][c];
                        }
                    }
                }
             }
         }
         
         return min;
    }


    /**
     * 296. Best Meeting Point
     * 给2D矩阵，1代表人，0代表空地，找个meeting point 使距离所有人最短
     * @param grid
     * @return
     * 这种如果是人 做的话，肯定会选中点。
     * 所以，要看是选中位数median 还是 平均数mean。 假设左边很多1，右边只有一个1，那平均数的话就不行。所以要选中位数
     * 1. 如果1D list是奇数，那3/2=1正好在中间。 如果是偶数，偏左偏右都可以，因为对称。所以4/2=2也行，这里会偏右
     * 2. 先看1D的怎么选，然后2D的话就是先选rows, 再选cols的中位数即可
     * 3. 得到rows & cols的中位数后，分别遍历这些list算出里面的1与median的距离是多少
     *
     * 为了保持O(m*n)，在挑有1的cols时，先遍历j. 这样就能sort保持顺序，到时直接取median即可
     *
     *  1 - 0 - 0 - 0 - 1
        |   |   |   |   |
        0 - 0 - 0 - 0 - 0
        |   |   |   |   |
        0 - 0 - 1 - 0 - 0  最近的点是(0,2)
     *  这里rows得到的是(0, 0, 2).  cols得到(0, 2, 4). 所有取中点，加上与median的距离，就是答案
     */
    public int minTotalDistance(int[][] grid) {
        if (grid == null || grid.length == 0)   return 0;

        List<Integer> rows = getRows(grid);     //get all 1 in all rows / cols
        List<Integer> cols = getCols(grid);

        int medRow = rows.get(rows.size() / 2);    //find the median
        int medCol = cols.get(cols.size() / 2);    //find the median
        return minDistance(rows, medRow) + minDistance(cols, medCol);
    }

    // 法二 双指针
    public int minTotalDistanceTwoPointer(int[][] grid) {
        if (grid == null || grid.length == 0)   return 0;

        List<Integer> rows = getRows(grid);     //get all 1 in all rows / cols
        List<Integer> cols = getCols(grid);

        return getMinDist(rows) + getMinDist(cols);		//或者不需要getMedian，直接双指针
    }

    private int getMinDist(List<Integer> points) {
        int sum = 0;
        int i = 0;
        int j = points.size() - 1;
        while (i < j) {         // another way to get median
            sum += points.get(j--) - points.get(i++);   //combine i++,j--
        }
        return sum;
    }

    private int minDistance(List<Integer> points, int median) {
        int minDist = 0;
        for (int p : points) {
            minDist += Math.abs(p - median);
        }
        return minDist;
    }

    private List<Integer> getRows(int[][] grid) {
        List<Integer> rows = new ArrayList<>();
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    rows.add(i);
                }
            }
        }
        return rows;
    }

    private List<Integer> getCols(int[][] grid) {
        List<Integer> cols = new ArrayList<>();
        for (int j = 0; j < grid[0].length; j++) {      // j first to sort cols
            for (int i = 0; i < grid.length; i++) {
                if (grid[i][j] == 1) {
                    cols.add(j);
                }
            }
        }
        return cols;
    }


    /** 296. Best Meeting Point2 - Naive BFS
     * 这太naive，复杂度高 O(m^​2 *​ n^​2​​ ).. 跟上面的Shortest Distance from All Buildings一样解法..
     * 但是这题没有障碍物.. 所以不需要复杂度这么高的
     * @param grid
     * @return
     */
    public int minTotalDistanceBFS(int[][] grid) {
        if (grid == null || grid.length == 0)   return 0;

        int m = grid.length;
        int n = grid[0].length;
        int[] d = new int[] {0, -1, 0, 1, 0};
        int[][] distance = new int[m][n];

        int min = Integer.MAX_VALUE;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    Queue<int[]> q = new LinkedList<>();
                    q.add(new int[] {i, j});
                    boolean[][] visit = new boolean[m][n];

                    while (!q.isEmpty()) {
                        int[] p = q.poll();
                        int x = p[0];
                        int y = p[1];
                        for (int k = 0; k < 4; k++) {
                            int r = x + d[k];
                            int c = y + d[k+1];
                            if (r >= 0 && r < m && c >= 0 && c < n && !visit[r][c]) {
                                distance[r][c] += Math.abs(i-r) + Math.abs(j-c);
                                visit[r][c] = true;
                                q.add(new int[] {r, c});
                            }
                        }
                    }
                }
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                min = Math.min(min, distance[i][j]);
            }
        }

        return min;
    }



    /**
     * 542. 01 Matrix
     * 0 & 1 的矩阵里，find the distance of the nearest 0 for each cell.
     * 本身是0的话就还是0，当1的时候就找最近的0的距离多少
     * @param matrix
     * @return
     *
     * 找最近距离，那就用BFS.。
     *
     * 这里有点要注意，是要从0开始扩散..
     * 尽管直觉是对每个1进行BFS找0，但这样是brute force.. 要搜好多次，每次搜完只更新1一个位置，其他1的位置又重复搜了..
     * 所以需要从0开始搜，这样搜的过程碰到1就可以更新他们的距离，一举n得
     *
     * 那么就要把 0 的坐标放进queue里面来BFS。。
     *
     * 另外，第一次遍历时，把 1 的坐标设成max..这样到时更新距离时比较方便更新，不用每次都更新，而是新的距离比当前大的要小 才更新
     *
     * 下面有种更快的方法
     */
    public int[][] updateMatrix(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;

        int[][] dist = new int[m][n];

        int max = m * n;

        Queue<int[]> q = new LinkedList<>();

        // 因为我们从 0 开始扩散，看到1就更新距离，这样比从1扩散要快。因为1开始的话只能更新自己，0开始的话一次能更新多个1的位置
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    q.add(new int[]{i, j});     // 只放 0 的坐标..
                } else {
                    dist[i][j] = max;
                }
            }
        }

        int[] dir = {0, -1, 0, 1, 0};

        while (!q.isEmpty()) {
            int[] pos = q.poll();
            int x = pos[0];
            int y = pos[1];
            for (int k = 0; k < 4; k++) {
                int newX = x + dir[k];
                int newY = y + dir[k + 1];
                if (newX < 0 || newX >= m || newY < 0 || newY >= n ||
                        dist[newX][newY] <= dist[x][y] + 1)
                    continue;

                q.add(new int[]{newX, newY});
                dist[newX][newY] = dist[x][y] + 1;
            }
        }
        return dist;
    }


    /**
     * 542. 01 Matrix - faster
     *
     * 这个分 two pass..
     * 第一次 从 左上开始扫，所以只看 left & up
     * 第二次 从 右下往上扫，看right & down.. 和现有的 dist[i][j] 看哪个大
     */
    public int[][] updateMatrixFaster(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;

        int[][] dist = new int[m][n];

        int max = m * n;

        // 第一遍看 left & up cells
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 1) {
                    int upCell = i > 0 ? dist[i - 1][j] : max;
                    int leftCell = j > 0 ? dist[i][j - 1] : max;
                    dist[i][j] = Math.min(upCell, leftCell) + 1;
                }
            }
        }

        // 第二遍看 right & down cells
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                if (matrix[i][j] == 1) {
                    int downCell = i + 1 < m ? dist[i + 1][j] : max;
                    int rightCell = j + 1 < n ? dist[i][j + 1] : max;
                    // 看是第一次已经算完的 左上小，还是这次的右下小
                    dist[i][j] = Math.min(dist[i][j], Math.min(downCell, rightCell) + 1);
                }
            }
        }
        return dist;
    }


    /**
     * 01 Matrix Walking Problem
     * 从左上角 到 右下角.. 1是wall 0是空地，问如果把一个1变成0，能否走到end..有的话，最少需要多少步
     *
     * Approach: BFS
     * 求最短路径，因此可以想到使用 BFS 来解决这道问题。
     * 我们需要求：
     *  从 左上角 到 右下角 不经过障碍点的最短距离
     *  从 右下角 到 左上角 不经过障碍点的最短距离
     *  修改每个障碍点之后，到左上角和右上角的距离之和。
     * 然后在这些值中取最小值即可。
     *
     * 那么其实可以算 从每个障碍点出发，到左上角 & 右下角分别的最短距离..
     * 这样到时就能在这些里面取min
     *
     * Note:
     *  本题的难点就是在于图的布局是可变的，但是我们不能对每个可变的点都进行一次 BFS.
     *  因为这样时间复杂度肯定会超时的，所以我们可以利用一个 matrix 来存储计算好的结果。
     *  也就是 空间换时间 的做法。
     *
     * 时间复杂度：O(MN)
     * 空间复杂度：O(MN)
     */
    public int getBestRoad(int[][] grid) {
        int rows = grid.length;
        int cols = grid[0].length;
        // 从top left出发，各个障碍点到左上角顶点的距离(包括右下角顶点)
        int[][] disToUL = new int[rows][cols];
        // 从bottom right出发，各个障碍点到右下角顶点的距离(包括左上角顶点)
        int[][] disToLR = new int[rows][cols];

        bfs(disToUL, grid, new int[]{0, 0}, new int[]{rows - 1, cols - 1});
        bfs(disToLR, grid, new int[]{rows - 1, cols - 1}, new int[]{0, 0});

        int minDistance = Integer.MAX_VALUE;
        if (disToUL[rows - 1][cols - 1] != 0) {
            minDistance = Math.min(minDistance, disToUL[rows - 1][cols - 1]);
        }
        if (disToLR[0][0] != 0) {
            minDistance = Math.min(minDistance, disToLR[0][0]);
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 1 && disToUL[i][j] != 0 && disToLR[i][j] != 0) {
                    minDistance = Math.min(minDistance, disToUL[i][j] + disToLR[i][j]);
                }
            }
        }

        return minDistance == Integer.MAX_VALUE ? -1 : minDistance;
    }

    private void bfs(int[][] distance, int[][] grid, int[] start, int[] end) {
        int rows = grid.length;
        int cols = grid[0].length;

        Queue<int[]> queue = new LinkedList<>();
        boolean[][] visited = new boolean[rows][cols];
        queue.offer(new int[]{start[0], start[1]});
        visited[start[0]][start[1]] = true;

        int step = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] curr = queue.poll();
                int row = curr[0], col = curr[1];
                // 如果当前顶点为 1，则可以进行一次修改使得当前顶点是可达的。然后 continue
                // 如果是结束的目标位置，同样需要更新步数（距离）值
                if (grid[row][col] == 1 || (row == end[0] && col == end[1])) {
                    distance[row][col] = step;
                    continue;           // 记得continue，因为已经是wall了，只能换一次0，不能再往后试
                }

                for (int[] dir : DIRS) {
                    int nextRow = row + dir[0];
                    int nextCol = col + dir[1];
                    if (nextRow < 0 || nextRow >= rows || nextCol < 0 || nextCol >= cols
                            || visited[nextRow][nextCol]) {
                        continue;
                    }
                    queue.offer(new int[]{nextRow, nextCol});
                    visited[nextRow][nextCol] = true;
                }
            }
            step++;
        }
    }

    public static final int[][] DIRS = new int[][]{{0, -1}, {0, 1}, {-1, 0}, {1, 0}};




    /**
     * 361. Bomb Enemy - Naive
     * 2D grid, each cell is either a wall 'W', an enemy 'E' or empty '0' (the number zero)
     * 如果在0的空地放炸弹，可以炸十字行列，但墙后面的炸不了。 return最多炸几个敌人
     * @param grid
     * @return
     * 这是Naive方法，for循环每个空地，每次扫4个方向看十字行列有多少个enemy
     * Attention：4次行列时，记得把x,y reset成当前i,j, 否则值变了导致错误
     * O(m*n * (m+n))
     * 看下面的better
     */
    public int maxKilledEnemies(char[][] grid) {
        if (grid == null || grid.length == 0)   return 0;
        
        int m = grid.length;
        int n = grid[0].length;
        int max = 0;
        int enemy = 0;
                    
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '0') {        // 只有empty place才能放炸弹
                    enemy = 0;

                    for (int x = i + 1; x < m && grid[x][j] != 'W'; x++) {
                        if (grid[x][j] == 'E') {
                            enemy++;
                        }
                    }

                    for (int x = i - 1; x >= 0 && grid[x][j] != 'W'; x--) {
                        if (grid[x][j] == 'E') {
                            enemy++;
                        }
                    }

                    for (int y = j + 1; y < n && grid[i][y] != 'W'; y++) {
                        if (grid[i][y] == 'E') {
                            enemy++;
                        }
                    }

                    for (int y = j - 1; y >= 0 && grid[i][y] != 'W'; y--) {
                        if (grid[i][y] == 'E') {
                            enemy++;
                        }
                    }

                    max = Math.max(max, enemy);

                    /*
                    int x = i;
                    int y = j;
                    
                    while (x > 0) {     //up
                        x--;
                        if (grid[x][j] == 'W')    break;
                        if (grid[x][j] == 'E')    enemy++;
                    }
                    x = i;					//记得变回i !!!!!!!!!!!
                    while (x < m - 1) {     //down
                        x++;
                        if (grid[x][j] == 'W')    break;
                        if (grid[x][j] == 'E')    enemy++;
                    }
                        
                    while (y > 0) {     //left
                        y--;
                        if (grid[i][y] == 'W')    break;
                        if (grid[i][y] == 'E')    enemy++;
                    }
                     y = j;   		//记得变回j
                    while (y < n - 1) {     //right
                        y++;
                        if (grid[i][y] == 'W')    break;
                        if (grid[i][y] == 'E')    enemy++;
                    }
                    */
                }
            }
        }
        
        return max;       
    }
    
    
    
    /**
     * 361. Bomb Enemy
     * 上一中方法太naive，每次都要重复算每行每列有多少enemy。如何利用已经搜索过的信息就是本题的考点
     * 所以就要想到用memory cache来存储 这些enemy
     * @param grid
     * @return
     *
     * 算每行每列有多少敌人。在2中情况需要重新算：
     * 1. 当每行每列开始时 i, j = 0
     * 2. 当前一个是wall时，需要重新算敌人
     * 
     * 算了敌人后就能根据 rowCount 和 colCoun[j]求出enemy结果
     *
     * rowCount是int就行，因为从上到下 i在外层，不会来回扫，只用int即可
     */
    public int maxKilledEnemiesBetter(char[][] grid) {
        if (grid == null || grid.length == 0)   return 0;
        
        int m = grid.length;
        int n = grid[0].length;
        int max = 0;

        int rowCount = 0;       // enemies in a row 因为从上到下 i在外层，不会来回扫，只用int即可
        int[] colCount = new int[n];    //enemies in a col 需要来回扫，所以要int[]
                    
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
            	// 算每行row有多少敌人
                // only need to recalculate rowCount when start from 0, 或左边是墙
                if (j == 0 || grid[i][j - 1] == 'W') {
                    rowCount = 0;         // !!!! reset

                    for (int k = j; k < n && grid[i][k] != 'W'; k++) {
                        if (grid[i][k] == 'E') {
                            rowCount += 1;
                        }
                    }
                }
                
                // 算每列col多少敌人
                // only re-count when start or up level is wall 或上面是墙
                if (i == 0 || grid[i - 1][j] == 'W') {
                    colCount[j] = 0;         // !!!! reset

                    for (int k = i; k < m && grid[k][j] != 'W'; k++) {
                        if (grid[k][j] == 'E') {
                            colCount[j] += 1;
                        }
                    }
                }
                
                if (grid[i][j] == '0') {                         //if empty place
                    max = Math.max(max, rowCount + colCount[j]);
                }
                
            }
        }
        
        return max;       
    }
    
    

    
    /** 399. Evaluate Division  - Google   -  DFS
     * Given a / b = 2.0, b / c = 3.0. 
		queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? . 
		return [6.0, 0.5, -1.0, 1.0, -1.0 ].
     * @param equations
     * @param values
     * @param queries
     * @return
     * 其实这些表示的是graph，只要DFS找是否有路径从 start 到end
     * dfs里，每次会改变start和val
     * 且记得用hashset来存visit过的edge，否则会循环算
     * 
     * 后面还有用union find更好
     */
    public double[] calcEquation(String[][] equations, double[] values, String[][] queries) {
        Map<String, List<Edge>> graph = new HashMap<>();
        //建图
        for (int i = 0; i < equations.length; i++) {		
            String[] e = equations[i];
            double val = values[i];
            if (!graph.containsKey(e[0]))	graph.put(e[0], new ArrayList<Edge>());
            if (!graph.containsKey(e[1]))	graph.put(e[1], new ArrayList<Edge>());
            
            graph.get(e[0]).add(new Edge(e[1], val));
            graph.get(e[1]).add(new Edge(e[0], 1.0 / val));
        }
        
        double[] result = new double[queries.length];
        
        // scan queris
        for (int i = 0; i < queries.length; i++) {
            String[] q = queries[i];
            result[i] = dfs(graph, q[0], q[1], new HashSet<>(), 1.0);
        }
        
        return result;
    }
    
    
    public double dfs(Map<String, List<Edge>> graph, String node, String end, Set<String> visited, double val) {
        if (visited.contains(node))    return -1.0;     //防止来回重复会stack overflow
        if (!graph.containsKey(node))  return -1.0;
        if (node.equals(end))          return val;
        
        visited.add(node);			//start每次都会变
        
        for (Edge edge : graph.get(node)) {
            double result = dfs(graph, edge.v, end, visited, val * edge.w);		
            if (result != -1.0)
                return result;
        }
        return -1.0;
    }
    
    
    class Edge {
        String v;
        double w;
        
        public Edge(String v, double weight) {
            this.v = v;
            this.w = weight;
        }
    }
    
    
    /**
     * 399. Evaluate Division  - Google
     * 用BFS, 比较慢
     * @param equations
     * @param values
     * @param queries
     * @return
     */
    public double[] calcEquation1(String[][] equations, double[] values, String[][] queries) {
        Map<String, List<String[]>> map = new HashMap<>();
        for (int i = 0; i < equations.length; i++) {
            String[] e = equations[i];
            if (!map.containsKey(e[0])) {
                map.put(e[0], new ArrayList<String[]>());
            }
            map.get(e[0]).add(new String[]{e[1], String.valueOf(values[i])});
            
            // put for other direction
            if (!map.containsKey(e[1])) {
                map.put(e[1], new ArrayList<String[]>());
            }
            map.get(e[1]).add(new String[]{e[0],String.valueOf(1.0 / values[i])});  
        }
        
        double[] result = new double[queries.length];
        
        for (int i = 0; i < queries.length; i++) {
            String[] q = queries[i];
            if (!map.containsKey(q[0]) && !map.containsKey(q[1])) {
                result[i] = -1.0;
            } else {
                result[i] = bfs(map, q[0], q[1]);
            }
        }
        return result;
    }
    
    // BFS 比较麻烦
    private double bfs(Map<String, List<String[]>> map, String s, String target) {
        Queue<String> q = new LinkedList<>();
        q.add(s);
        
        Queue<Double> resultQ = new LinkedList<>();
        resultQ.add(1.0);
        
        Set<String> used = new HashSet<>();
        
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                String a = q.poll();
                double result = resultQ.poll();
                if (map.containsKey(a)) {
                    for (String[] cand : map.get(a)) {
                        if (used.contains(a))
                            continue;
                        
                        double val = Double.parseDouble(cand[1]);
                        if (cand[0].equals(target)) {
                            return result * val;
                        } else {
                            q.add(cand[0]);
                            resultQ.add(result * val);
                            used.add(a);
                        }
                    }
                }
            }
        }
        return -1.0;
    }

    
    /**
     * 399. Evaluate Division  - Union Find 如果操作多次的话
     * 这个不用额外空间
     * 
     * 把 a->b (parent), b->c (ancestor) 连起来 生成最终的 a->c
     *
     * 1. for循环equations
     *   - findRoot（其中包括建root）
     *   - union起来，把 s1 -> s2 所对应的roots 连起来union
     *
     * 2. for循环queries，找root, 再看root是否一样，就可以得出result
     * 
     * union 那里，除了root.put(root1, root2) 为何不放 root.put(s1, root2) ?
     * answer：因为union只会union root，而不会把之前所有更早的点union起来比如s1->root2. 否则这样太慢了
     * 之后只需要在queries里findRoot就行.. 
     */
    public double[] calcEquationUnionFind(String[][] equations, double[] values, String[][] queries) {
        Map<String, String> root = new HashMap<>();
        Map<String, Double> distance = new HashMap<>(); // distance from orig to root
        
        // union equations
        for (int i = 0; i < equations.length; i++) {
            union(root, distance, equations[i][0], equations[i][1], values[i]);
        }
        
        double[] result = new double[queries.length];
        
        // get result for queries
        for (int i = 0; i < queries.length; i++) {
            String[] q = queries[i];
            if (root.containsKey(q[0]) && root.containsKey(q[1])) {
                // 要 find root. 不能直接查 rootMap. 因为有的没连起来
                String r1 = findRoot(root, distance, q[0]);
                String r2 = findRoot(root, distance, q[1]);
                if (r1.equals(r2)) {
                    result[i] = distance.get(q[0]) / distance.get(q[1]);
                    continue;
                }
            }
            result[i] = -1.0;
        }
        return result;
    }
    
    // 把 s1 -> s2 所对应的roots 连起来union 
    private void union(Map<String, String> root, Map<String, Double> distance, String s1, String s2, double val) {
        String root1 = findRoot(root, distance, s1);
        String root2 = findRoot(root, distance, s2);
        root.put(root1, root2);     // 把这2个root连起来
        // s1->r1->s2->r2 : s1 -> s2 (val) * s2 -> r2 / 去掉 s1 -> r1 这段，因为只更新 r1 开始的距离
        distance.put(root1, val * distance.get(s2) / distance.get(s1));
    }
    
    private String findRoot(Map<String, String> root, Map<String, Double> distance, String s) {
        // no root build for s, need to build for itself. 
        if (!root.containsKey(s)) {
            root.put(s, s);
            distance.put(s, 1.0);
            return s;
        }
        
        // 自己是自己的root
        if (root.get(s).equals(s)) {
            return s;
        }
        
        // 把 a->b (parent), b->c (ancestor) 连起来 生成最终的 a->c. compressed压缩路径 能快点
        String parent = root.get(s);
        String ancestor = findRoot(root, distance, parent);
        root.put(s, ancestor);
        distance.put(s, distance.get(s) * distance.get(parent));
        
        return ancestor;
    }
    
    
    /**
     * 547. Friend Circles  看多少个朋友圈 - DFS
     * 简单，像connected component.. 但是由邻接matrix组成，所以还是有点不一样
     * M[i][j]=1 表示 i和j是朋友，也会连成更多1如果有间接朋友的话
     * 
     * 注意，主函数里先for(i), 接着call dfs()时, dfs里面 for(j)。 而不是跟别的一样for(i) for(j)一起
     * 因为其实这里只是N个student..
     *
     * 其实像其他题的for(点)一样 course之类的
     * @param M
     * @return
     */
    public int findCircleNum(int[][] M) {
        int n = M.length;
        
        boolean[] visited = new boolean[n];
        
        int count = 0;
        for (int i = 0; i < n; i++) {		// 先看某人的，其实像其他的for(顶点)一样
            if (!visited[i]) {
                dfs(M, visited, i);
                count++;
            }
        }
        return count;
    }
    
    private void dfs(int[][] M, boolean[] visited, int i) {
        // 找neighbors
        for (int j = 0; j < M.length; j++) {
            if (!visited[j] && M[i][j] == 1) {
                visited[j] = true;              // 把相连的friend都标成visited
                dfs(M, visited, j);     // dfs its friend
            }
        }
    }

    // 或者visited往前 放 一样的
    private void dfs1(int[][] m, boolean[] visited, int i) {
        if (visited[i])     return;

        visited[i] = true;
        for (int j = 0; j < m.length; j++) {
            if (m[i][j] == 1) {
                dfs(m, visited, j);
            }
        }
    }

    
    /**
     * 547. Friend Circles  看多少个朋友圈 - Union Find
     * 跟Number of Connected Components in an Undirected Graph一样
     * @param M
     * @return
     */
    public int findCircleNum1(int[][] M) {
        int n = M.length;
        
        int[] roots = new int[n];
        for (int i = 0; i < n; i++) {
            roots[i] = i;
        }
        
        int count = n;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {	// j = i+1 就行，快点
                // union
                if (M[i][j] == 1) {
                    int r1 = find(roots, i);
                    int r2 = find(roots, j);
                    if (r1 != r2) {
                        roots[r1] = r2;
                        count--;			// count--
                    }
                }
            }
        }
        return count;
    }


    /**
     * 734. Sentence Similarity
     * 看2个单词是否same或者similar(出现在pair里)
     * 给words1 = ["great", "acting", "skills"] and words2 = ["fine", "drama", "talent"] are similar,
     * 如果有后面这些关系pair[["great", "good"], ["fine", "good"], ["acting","drama"], ["skills","talent"]]
     *
     * 关系不 transitive
     */
    public boolean areSentencesSimilar(String[] words1, String[] words2, String[][] pairs) {
        if (words1.length != words2.length) return false;

        Map<String, Set<String>> map = new HashMap<>();
        for (String[] p : pairs) {
            map.putIfAbsent(p[0], new HashSet<>());
            map.putIfAbsent(p[1], new HashSet<>());
            map.get(p[0]).add(p[1]);
            map.get(p[1]).add(p[0]);
        }

        for (int i = 0; i < words1.length; i++) {
            if (words1[i].equals(words2[i])) continue;
            if (!map.containsKey(words1[i])) return false;
            if (!map.get(words1[i]).contains(words2[i])) return false;
        }

        return true;
    }
    
    
    /**
     * 737. Sentence Similarity II
     * 给words1 = ["great", "acting", "skills"] and words2 = ["fine", "drama", "talent"] are similar, 
     * 如果有后面这些关系pair[["great", "good"], ["fine", "good"], ["acting","drama"], ["skills","talent"]]
     *
     * 这些关系是transitive的.. 那就用DFS， union find
     */
    public boolean areSentencesSimilarTwoDFS(String[] words1, String[] words2, String[][] pairs) {
        if (words1.length != words2.length)
            return false;
        
        Map<String, Set<String>> graph = new HashMap<>();
        for (String[] p : pairs) {
            graph.putIfAbsent(p[0], new HashSet<>());
            graph.putIfAbsent(p[1], new HashSet<>());
            graph.get(p[0]).add(p[1]);
            graph.get(p[1]).add(p[0]);
        }
        
        for (int i = 0; i < words1.length; i++) {
            if (words1[i].equals(words2[i]))
                continue;
            
            if (!dfs(graph, words1[i], words2[i], new HashSet<String>())) {
                return false;
            }
        }
        return true;
    }
    
    private boolean dfs(Map<String, Set<String>> graph, String word, String target, Set<String> used) {
        if (!graph.containsKey(word))
            return false;
        
        if (graph.get(word).contains(target))
            return true;
        
        used.add(word);
        
        for (String nei : graph.get(word)) {
            if (!used.contains(nei) && dfs(graph, nei, target, used)) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * 737. Sentence Similarity II - Union Find
     * 给words1 = ["great", "acting", "skills"] and words2 = ["fine", "drama", "talent"] are similar, 
     * 如果有后面这些关系pair[["great", "good"], ["fine", "good"], ["acting","drama"], ["skills","talent"]]
     *
     * 这些关系是transitive的.. 那就用DFS， union find
     */
    public boolean areSentencesSimilarTwo(String[] words1, String[] words2, String[][] pairs) {
        if (words1.length != words2.length)
            return false;
        
        Map<String, String> roots = new HashMap<>();
        for (String[] p : pairs) {
            union(roots, p[0], p[1]);
        }
        
        for (int i = 0; i < words1.length; i++) {
            if (words1[i].equals(words2[i]))
                continue;
            
            String r1 = findRoot(roots, words1[i]);
            String r2 = findRoot(roots, words2[i]);
            if (!r1.equals(r2)) {
                return false;
            }
        }
        return true;
    }
    
    private void union(Map<String, String> roots, String s1, String s2) {
        String r1 = findRoot(roots, s1);
        String r2 = findRoot(roots, s2);
        if (roots.get(r1) != roots.get(r2)) {
            roots.put(r1, r2);
        }
    }
    
    private String findRoot(Map<String, String> roots, String s) {
        if (!roots.containsKey(s)) {
            roots.put(s, s);
            return s;
        }
        
        if (roots.get(s).equals(s))
            return s;
        
        // 有下面这个能快些.. 相当于roots[id] = roots[roots[id]]
//         String parent = roots.get(s);
//         String ancestor = findRoot(roots, parent);
//         roots.put(s, ancestor);
        
//         return ancestor;
        
        return findRoot(roots, roots.get(s));
    }


    /**
     * 721. Accounts Merge - DFS   - Google
     *
     * account[0]是名字，后面的是对应的Email
     * 
     * a. 有的人开了几个账号，有的email可能相同，这时需要把这些accounts merge起来
     *   !!!!! 注意这些email可以transitive，所以要想到DFS或union find找connected emails
     *   
     * b. 但有人是same name, different person, different email，所以分开独立
     * 最后每个account的email要sort一下
     * 
     * 这题其实是按照email来找.. 同一个account内的几个email可以组成edge, 然后跟别的账号看是否connect能merge
     * 1. 建graph。一个account里，email i-1 & i 都放到graph里（只用前后i-1 & i连，无需所有，反正之后都能connect）
     * 2. for(graph.keySet())，遍历email，DFS把所有connected email连起来放list里加到result
     * 
     * @param accounts
     * @return
     */
    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        List<List<String>> result = new ArrayList<>();
        
        Map<String, Set<String>> graph = new HashMap<>();     // email to emails within same account
        Map<String, String> emailToName = new HashMap<>();    // email to account name
        
        // 建图
        for (List<String> account : accounts) {
            String name = account.get(0);
            for (int i = 1; i < account.size(); i++) {
                emailToName.put(account.get(i), name);
                
                if (!graph.containsKey(account.get(i))) {
                    graph.put(account.get(i), new HashSet<String>());
                }
                
                if (i == 1)     // 先加第一个，后面再加前一个i-1
                    continue;
                
                // connect 2 emails. 只用connect前后2个Email，这样到时就可以link all了
                graph.get(account.get(i)).add(account.get(i - 1));
                graph.get(account.get(i - 1)).add(account.get(i));
            }
        }
        
        Set<String> visited = new HashSet<>();
        
        // 扫描
        for (String email : graph.keySet()) {
            List<String> list = new ArrayList<>();
            if (!visited.contains(email)) {
                dfs(graph, email, list, visited);
                
                Collections.sort(list);
                list.add(0, emailToName.get(email));
                result.add(list);
            }
        }
        return result;
    }
    
    private void dfs(Map<String, Set<String>> graph, String email, List<String> list, Set<String> visited) {
        visited.add(email);
        list.add(email);
        
        for (String nei : graph.get(email)) {
            if (!visited.contains(nei)) {
                dfs(graph, nei, list, visited);
            }
        }
    }
    
    /**
     * 721. Accounts Merge - Union Find
     * account[0]是名字，后面的是对应的Email
     * 
     * @param accounts
     * @return
     */
    public List<List<String>> accountsMergeUF(List<List<String>> accounts) {
        List<List<String>> result = new ArrayList<>();
        
        Map<String, String> roots = new HashMap<>();
        Map<String, String> emailToName = new HashMap<>();
        
        for (List<String> account : accounts) {
            String name = account.get(0);
            for (int i = 1; i < account.size(); i++) {
                String email = account.get(i);
                emailToName.put(email, name);      // 记得放account
                
                if (!roots.containsKey(email)) {	// 要判断，否则会覆盖掉
                    roots.put(email, email);
                }
                
                if (i == 1)    continue;
                
                // union 前后
                union(roots, email, account.get(i - 1));
            }
        }
        
        // 一个root email，对应其他email
        Map<String, Set<String>> emailsWithSameRoot = new HashMap<>();
        for (String email : roots.keySet()) {
            String root = findRoot1(roots, email);
            if (!emailsWithSameRoot.containsKey(root)) {
                emailsWithSameRoot.put(root, new HashSet<String>());
            }
            emailsWithSameRoot.get(root).add(email);
        }
        
        for (String root : emailsWithSameRoot.keySet()) {
            List<String> list = new ArrayList<>(emailsWithSameRoot.get(root));
            Collections.sort(list);
            list.add(0, emailToName.get(root));
            result.add(list);
        }
        return result;
    }

    private String findRoot1(Map<String, String> roots, String s) {
        // if (!roots.containsKey(s)) {		// 因为前面union之前判断过了，无需再来
        //     roots.put(s, s);
        //     return s;
        // }
        
        if (roots.get(s).equals(s)) {
            return s;
        }
        
        return findRoot(roots, roots.get(s));
    }


    
    /**
     * 684. Redundant Connection - dfs   - Google
     *
     * 题目假设只有一条多余的边
     * undirected edge无向的
     *
     * 这其实是个tree，找多余的边（能组成环的），并返回最后出现的多余的边undirected..with one additional directed edge added.
     *
     * 每条边for (edges)来dfs搜是否有cycle O(n^2) 慢
     * @param edges
     * @return
     */
    public int[] findRedundantConnection(int[][] edges) {
        List<Integer>[] adjList = new ArrayList[edges.length + 1];
        for(int i = 0; i < edges.length + 1; i++){
            adjList[i] = new ArrayList<>();
        }
        
        // 每条边都DFS下，看是否相连，是的话就有环找到return。否则就顺便建图
        for (int[] e : edges) {
            int u = e[0];
            int v = e[1];
            Set<Integer> visited = new HashSet<>();
            
            if (dfsHasCycle(adjList, visited, u, v)) {
                return e;
            }
            // build adjList
            adjList[u].add(v);
            adjList[v].add(u);
        }
        
        return null;
    }
    
    private boolean dfsHasCycle(List<Integer>[] adjList, Set<Integer> visited, int u, int target) {
        if (u == target)         // cycle
            return true;        
        
        visited.add(u);
        
        for (int nei : adjList[u]) {
            if (!visited.contains(nei) && dfsHasCycle(adjList, visited, nei, target)) {
                return true;   
            }
        }
        return false;
    }
    
    /**
     * 684. Redundant Connection - Union Find
     * 这其实是个tree，找多余的边（能组成环的），并返回最后出现的多余的边
     * 
     * for(edges), 只要看到root相同，就说明connect找到了，否则就连起来
     */
    public int[] findRedundantConnectionUF(int[][] edges) {
        int[] roots = new int[edges.length + 1];
        // 记得要initialize
        for (int i = 1; i < roots.length; i++) {
            roots[i] = i;
        }
        
        for (int[] e : edges) {
            int r1 = find(roots, e[0]);
            int r2 = find(roots, e[1]);
            if (r1 != r2) {
                roots[r1] = r2;
            } else {        // already connected, found
                return e;
            }
        }
        
        return new int[2];      
    }
    
    
    /**
     * 685. Redundant Connection II
     * 跟上面一样，但是 有向图 directed.. u(parent) -> v(child)
     * 题目假设只有一条多余的边
     * 
     * 这里可能的情况是 A: 有奇葩点（2个parent）B:有环loop..组成不同情况
     * 
     * code里面这样区分：
     * 1. 先找有没一个node有2个parent的，如果有的话把这2条edge分别放到 cand1 & cand2
     * 	  找到的话，顺便把edge2 去掉边，这样使得 cand2可能去掉loop
     *
     * 2. 再for(edges) union find
     * 	a. 如果现在valid 无环了，直接return cand2 , 因为1里去掉了那条边就没有loop
     *  b. 没奇葩点，单纯只有环 return edge
     *  c. 有奇葩点（2个parent），并且有环, 那就return cand1
     * @param edges
     * @return
     */
    public int[] findRedundantDirectedConnection(int[][] edges) {
        // 根据node找parent. 通常是一一对应，如果已经有这node key，说明他有多个parent，有错
        Map<Integer, Integer> parentMap = new HashMap<>();
        
        // 有问题的边 (有2个parent)
        int[] cand1 = new int[2];
        int[] cand2 = new int[2];
        
        // 每条边都遍历下，加parent，看是否相连，是的话就有环找到return。否则就顺便建图
        for (int[] e : edges) {
            int parent = e[0];
            int child = e[1];
            
            // build parent & 找看有没 2个 parent的
            if(parentMap.containsKey(child)) {
                cand1[0] = parentMap.get(child);        // 要么之前那条edge有问题
                cand1[1] = child;

                cand2 = new int[]{e[0], e[1]};          // 要么是现在这条edge有问题
                
                e[1] = 0;   // 断掉第二条有问题的edge，这样没有loop的话就是cand2 了
                break;
            } else {
                parentMap.put(child, parent);
            }
        }
        
        int[] roots = new int[edges.length + 1];
        // 记得要initialize
        for (int i = 1; i < roots.length; i++) {
            roots[i] = i;
        }

        // union find  看有没环
        for (int[] e : edges) {
            if (e[1] == 0)      // cand2 断了的边，跳过
                continue;
            
            int r1 = find(roots, e[0]);
            int r2 = find(roots, e[1]);
            
            // 有loop..
            if (r1 == r2) {
                if (cand1[0] == 0) {    // 情况b：只有环，没有 2个parent的奇葩点
                    return e;
                } else {
                    return cand1;		// 情况c：有环，有奇葩点
                }
            } else {
                roots[r1] = r2;     // union
            }
        }
        
        // 情况a：如果上面 union时都没发现loop, 那就证明前面去掉的cand2 边是结果
        return cand2;
    }


    /**
     * 685. Redundant Connection II
     * 比上面简短 & 快，只for edges一遍..
     * parents[] 是放 edges的idx
     *
     * first & second 表示edges point to the same node
     *
     * 没太弄懂
     * https://leetcode.com/problems/redundant-connection-ii/discuss/108058/one-pass-disjoint-set-solution-with-explain
     */
    public int[] findRedundantDirectedConnectionII(int[][] edges) {
        int n = edges.length;

        int[] parents = new int[n + 1];
        Arrays.fill(parents, -1);

        int[] roots = new int[n + 1];
        for (int i = 1; i < roots.length; i++) {
            roots[i] = i;
        }

        int first = -1;     // first & second表示edges point to the same node
        int second = -1;
        int cycle = -1;      // loop

        // for loop all edges
        for (int i = 0; i < n; i++) {
            int parent = edges[i][0];
            int child = edges[i][1];

            if (parents[child] != -1) {     // 2 edges point to same node
                first = parents[child];
                second = i;
                continue;
            }

            parents[child] = i;         // 注意是放 edges的idx

            int parentRoot = find(roots, parent);
            if (parentRoot == child) {          // 有环
                cycle = i;
            } else {
                roots[child] = parentRoot;
            }
        }

        if (cycle == -1)    return edges[second];
        if (second == -1)   return edges[cycle];
        return edges[first];
    }



    /**
     * 785. Is Graph Bipartite?
     * 看这个图是否可以分成2个set.. 每个set里的node之间不能有edge.. 只能有到对方set的edge
     * graph[i] is a list of indexes j for which the edge between nodes i and j exists.
     * Input: [[1,2,3], [0,2], [0,1,3], [0,2]]
     * Output: false
     *
     * BFS.. 需要for graph，因为有可能 图是 disconnect的，需要遍历所有漏掉的点
     * 然后就used[]数组记录 visited.. 哪个set.. 用1 、2表示
     */
    public boolean isBipartite(int[][] graph) {
        // 0 - not visited; 1-set1; 2-set2
        int[] used = new int[graph.length];

        for (int i = 0; i < graph.length; i++) {
            if (used[i] == 0) {
                used[i] = 1;
                Queue<Integer> q = new LinkedList<>();
                q.add(i);

                while (!q.isEmpty()) {
                    int cur = q.poll();
                    for (int nei : graph[cur]) {
                        if (used[nei] == 0) {
                            used[nei] = used[cur] == 1 ? 2 : 1;
                            q.add(nei);
                        } else if (used[nei] == used[cur]) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }


    /**
     * 785. Is Graph Bipartite?
     *
     * 正常的 DFS set的话用 1 和 -1 表示.. 没visit过的就是0
     *
     * @param graph
     * @return
     */
    public boolean isBipartiteDFS(int[][] graph) {
        // 0 - not visited; 1: set1; -1: set2
        int[] used = new int[graph.length];

        for (int i = 0; i < graph.length; i++) {
            if (used[i] == 0 && !validDFS(graph, used, 1, i))
                return false;
        }
        return true;
    }

    private boolean validDFS(int[][] graph, int[] used , int color, int node) {
        if (used[node] != 0) {
            return used[node] == color;
        }

        // 记得set color
        used[node] = color;

        for (int nei : graph[node]) {
            if (!validDFS(graph, used, -color, nei))    // 设相反的-1
                return false;
        }
        return true;
    }


    /**
     * 886. Possible Bipartition
     * 有N个人，想把他们分成2组。。 dislike[i]=[a, b]说明a, b不能在一组
     *
     * 跟上面一样.. 只是这里外面一层变成for (i 每个人)
     *
     * TODO 还可以用dfs 和 union find
     */
    public boolean possibleBipartition(int N, int[][] dislikes) {
        Map<Integer, List<Integer>> graph = new HashMap<>();

        for (int[] d : dislikes) {
            if (!graph.containsKey(d[0]))   graph.put(d[0], new ArrayList<>());
            if (!graph.containsKey(d[1]))   graph.put(d[1], new ArrayList<>());
            graph.get(d[0]).add(d[1]);
            graph.get(d[1]).add(d[0]);
        }

        int[] used = new int[N + 1];        // 0 - not visited; 1-set1; 2-set2

        for (int i = 0; i < N; i++) {
            if (used[i] == 0) {
                used[i] = 1;
                Queue<Integer> q = new LinkedList<>();
                q.add(i);

                while (!q.isEmpty()) {
                    int cur = q.poll();
                    if (!graph.containsKey(cur))
                        continue;
                    for (int nei : graph.get(cur)) {
                        if (used[nei] == 0) {
                            used[nei] = used[cur] == 1 ? 2 : 1;
                            q.add(nei);
                        } else if (used[nei] == used[cur]) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }


    /**
     * 787. Cheapest Flights Within K Stops
     *
     * Dijkstra - 在PQ里放src到当前路径的cost总和.. 每次挑min poll出来
     *
     * 1. 建图，按照flight[3] 那些来
     * 2. 建PriorityQueue 里面放 int[cost, dst_city, stop]
     * 3. while循环pq.. 当stop < K 时，就可以把neighbor下一站放进去，并更新 src到cur所有cost总和
     *
     * ！！！！！注意 放进pq时，要更新cost为src到cur所有cost总和，否则如果只是某一段的cost那会错的
     *
     * Follow-up
     * 如果要print path的话，那就在 Flight类里加上 String path即可
     *
     * 有可能条件 是k次flights （不是k stops）k次航班..就是k-1次stops
     *
     * Follow up 1 : What if I want you to compute maximum cities you could visit with certain amount of money in hand?
     *      每个Node出发，做BFS，到下一个Node记录当前到达这个Node所visit过得城市数目和当前还剩下的cost，
     *      如果visit过得数目大于全局最大，更新。这题不需要写code，说一说思路就行。
     *
     * Follow up 2: Will adding negative 负数 cost to the original problem still work using existing code?
     *      第一个方法用PQ, Dijkstra在有负数权重的情况下会 不work 不行的!!!!，因为 不会因为有负边而去更新已经计算过的路径
     *      比如  A --3-- B
     *            \      /
     *             2   -3
     *              \  /
     *               C
     *
     *      src是A，dst是C。Dijikstra返回结果是 A->C 为2，但其实有A->B->C 为0 的更短.. 因为放进pq后不会再trigger更新..贪心
     *
     *      第二个方法Bellman Ford可以，因为总是在更新
     */
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
        // src,          dest,     price
        Map<Integer, Map<Integer, Integer>> graph = new HashMap<>();
        // build the graph
        for (int[] f : flights) {
            if (!graph.containsKey(f[0])) {
                graph.put(f[0], new HashMap<>());
            }
            graph.get(f[0]).put(f[1], f[2]);
        }

        // costs, city, stop
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        pq.add(new int[]{0, src, -1});

        while (!pq.isEmpty()) {
            int[] cur = pq.poll();
            int costs = cur[0];
            int city = cur[1];
            int stops = cur[2];

            if (city == dst)
                return costs;

            if (stops < K) {
                if (graph.containsKey(city)) {
                    for (Map.Entry<Integer, Integer> e : graph.get(city).entrySet()) {
                        //  记得 这里的 cost要加上之前的 !!!!  这样才能保证下一次poll是截止src到cur的路径和的min
                        pq.add(new int[]{costs + e.getValue(), e.getKey(), stops + 1});
                    }
                }
            }
        }
        return -1;
    }


    /**
     * Bellman Ford方法 - O(E * K)  E是flights的总长度
     *
     * 通常的Bellman Ford：循环v-1次 (v是顶点数)，每一轮更新dist[] 当前 dist[u] + weight 跟 dist[v]下一个node比，小的话就更新
     * curDist[v] = Math.min(curDist[v], dist[u] + w)
     *
     * 在这题里，我们最多K次stop，所以可以飞 k+1 次，所以我们只用循环 k+1 次即可
     * 每次循环会更新整个dist[]数组
     *
     * 其实这个核心思想就是DP..
     * dp[k][i] 表示 最多飞k次到达i 所需的min cost.. 所以最外层是 for(k), 里面每次更新dp[k][i]
     *
     * for (k = 0 <= K) {
     *     dp[k][src] = 0;
     *
     *     for(all flights) {
     *          dp[k][to] = min(dp[k][to], dp[k-1][from] + cost)
     *     }
     * }
     *
     * 然后变成一维的就成了Bellman Ford方法
     *
     * https://www.cnblogs.com/grandyang/p/9109981.html
     * https://www.youtube.com/watch?v=obWXjtg0L64
     */
    public int findCheapestPriceBF(int n, int[][] flights, int src, int dst, int K) {
        int MAX_VALUE = 100001;
        int[] dist = new int[n];
        Arrays.fill(dist, MAX_VALUE);
        dist[src] = 0;

        for (int i = 0; i <= K; i++) {
            // 需要保留上一次的，这样pre[from] + cost才对，不能用这次的，否则这次都更新了（本来应该下一轮K才能更新）
            int[] curDist = dist.clone();

            for (int[] f : flights) {
                int from = f[0], to = f[1], cost = f[2];
                curDist[to] = Math.min(curDist[to], dist[from] + cost);
            }
            dist = curDist;
        }
        return dist[dst] == MAX_VALUE ? -1 : dist[dst];     // 记得返回的是 d[destination].. 而不是d[n-1]
    }



    /**
     * Ten Wizards - 有权有向图的最短路径   airbnb
     *
     * There are 10 wizards, 0-9, you are given a list that each entry is a list of wizards known by wizard.
     * The cost between wizards and wizard as square of different of i and j. index的距离
     * To find the min cost between 0 and 9.
     *
     * wizard[0] list: 1, 4, 5 
     *
     * wizard[4] list: 9
     *
     *  wizard 0 to 9 min distance is (0-4)^2+(4-9)^2 = 41 (wizard[0] -> wizard[4] -> wizard[9])
     * 法1 - BFS
     * 记得只在 新的dist <= minDist时才加到queue里，剪枝.. 超过了就没必要加
     */
    public List<Integer> getShortestPath1(List<List<Integer>> wizards, int source, int destination) {
        Queue<Wizard> q = new LinkedList<>();
        q.offer(new Wizard(source, 0));
        int minDist = Integer.MAX_VALUE;

        int n = wizards.size();
        int[] parents = new int[n];        // follow-up 打印出最短路径的path

        while (!q.isEmpty()) {
            Wizard cur = q.poll();
            if (cur.id == destination) {
                minDist = Math.min(minDist, cur.distance);
            }

            for (int nei : wizards.get(cur.id)) {
                int dist = cur.distance + (nei - cur.id) * (nei - cur.id);
                if (dist <= minDist) {               // 只加符合条件的，cost超过现在那就别想了不会是结果
                    q.offer(new Wizard(nei, dist));
                    parents[nei] = cur.id;          // if follow-up to print path, 放nei的parent为cur.id
                }
            }
        }

        System.out.println("min cost is " + minDist);

        // 打印shortest path 路径
        List<Integer> result = new ArrayList<>();
        int idx = destination;
        while (idx != source) {     // !!! 跟source比
            result.add(idx);
            idx = parents[idx];     // 向上找parent
        }
        result.add(source);

        // 记得reverse, 或者LinkedList.addFirst()
        Collections.reverse(result);

        System.out.println("path : " + result);

        return result;
    }

    class Wizard implements Comparable<Wizard> {
        int id;
        int distance;

        Wizard(int id, int distance) {
            this.id = id;
            this.distance = distance;
        }

        @Override
        public int compareTo(Wizard other) {
            return this.distance - other.distance;
        }
    }



    /**
     * Ten Wizards - 有权有向图的最短路径   airbnb
     * 法2 - Dijkstra
     * 用PriorityQueue, 遇到destination了就可以找到minDist可以break了。剩下的跟上面一样
     *
     * !!!!! 记得用 visited来防止cycle..
     * 理论上int distance的话没问题，但如果打印path的话，之前valid的parent map的值会被后面循环cycle的值overwrite导致出错
     */
    public List<Integer> getShortestPath2(List<List<Integer>> wizards, int source, int destination) {
        PriorityQueue<Wizard> pq = new PriorityQueue<>();
        pq.offer(new Wizard(source, 0));
        int minDist = Integer.MAX_VALUE;

        int n = wizards.size();
        Set<Integer> visited = new HashSet<>();

        int[] parents = new int[n];        // follow-up 打印出最短路径的path

        while (!pq.isEmpty()) {
            Wizard cur = pq.poll();
            if (cur.id == destination) {
                minDist = Math.min(minDist, cur.distance);
                break;
            }

            // !!!! 记住加这个check visited, 否则有环，打印path会错，因为前面正确的parent map被后面cycle的overwrite了
            if (visited.contains(cur.id)) {
                continue;
            }

            visited.add(cur.id);

            for (int nei : wizards.get(cur.id)) {
                int dist = cur.distance + (nei - cur.id) * (nei - cur.id);
                pq.offer(new Wizard(nei, dist));
                if (dist <= minDist) {             // 剪枝，可以加快
                    parents[nei] = cur.id;          // if follow-up to print path, 放nei的parent为cur.id
                }
            }
        }

        // 打印shortest path 路径
        LinkedList<Integer> result = new LinkedList<>();
        int idx = destination;
        while (idx != source) {     // !!! 跟source比
            result.addFirst(idx);
            idx = parents[idx];     // 向上找parent
        }
        result.addFirst(source);

        System.out.println("path : " + result);

        return result;
    }




    /**
     * 815. Bus Routes
     * 给多条 bus routes.. 和 start, target站点，看中途at least需要换乘多少辆bus
     * Input:
     * routes = [[1, 2, 7], [3, 6, 7]]
     * S = 1
     * T = 6
     * Output: 2
     *
     * 这题的关键key：把 ***** bus route 作为node和edge 来考虑******，而不是stop
     * start对应多个bus, target也是，那么要找 经过多少层，换到终点的bus..
     * 而且题目问的是，最少要换乘几次，也就是按照bus route层数来
     *
     * 用hashset保存访问过的bus.. 因为每层是bus个数来算
     *
     * 注意，如果build graph时按照stop来，那会很复杂，也更多可能性，不好记录换乘几次bus
     *
     * https://leetcode.com/articles/bus-routes/  这有个求intersection的好像还能加快速度.. 但是没看
     */
    public int numBusesToDestination(int[][] routes, int start, int target) {
        if (start == target)
            return 0;

        Map<Integer, Set<Integer>> stopToRoutes = new HashMap<>();

        // 要把 bus route作为node来考虑...
        // build graph key is stop, value is list of routes(bus) that pass the stop key
        for (int bus = 0; bus < routes.length; bus++) {
            for (int stop : routes[bus]) {
                if (!stopToRoutes.containsKey(stop)) {
                    stopToRoutes.put(stop, new HashSet<>());
                }
                stopToRoutes.get(stop).add(bus);
            }
        }

        Set<Integer> visitedBus = new HashSet<>();
        Queue<Integer> stopQ = new LinkedList<>();
        stopQ.offer(start);

        int minBus = 0;

        while (!stopQ.isEmpty()) {
            minBus++;                   // 最小的层数就是bus数
            int size = stopQ.size();
            for (int i = 0; i < size; i++) {
                int stop = stopQ.poll();
                for (int bus : stopToRoutes.get(stop)) {
                    if (!visitedBus.contains(bus)) {
                        visitedBus.add(bus);
                        // go through this bus route to check every stops
                        for (int nextStop : routes[bus]) {
                            if (nextStop == target) {
                                return minBus;
                            }
                            stopQ.offer(nextStop);
                        }
                    }
                }
            }
        }
        return -1;
    }


    /**
     * 465. Optimal Account Balancing - Google
     * 每个数组有3位 [0, 2, 10] 表示 people 0 欠 people 2  $10 块钱，问这几个人中，最后 最少需要多少次交易把钱还清 settle the debt
     * @param transactions
     * @return
     *
     * 这题关键是，有个account[] 记录每个人最后的amount.. （根据transaction来算 + / - amount)
     * ignore掉account是0的人.. 然后for循环 dfs 看哪个transaction最min.. 直到最后所有account为0
     *
     * 记得要backtracking回溯..
     * account[i] += account[pos];                 // 记得  dfs(pos + 1) !!!! 而非i+1
     * minTransaction = min(minTransaction, dfsTransaction(account, pos + 1) + 1);    // 这次交易算一次 所以 + 1
     * account[i] -= account[pos];
     *
     * 另外可以有些优化剪枝..preprocess时sort一下，快速skip掉0的情况，还能找到左右两边和为0的情况
     * 而且dfs时，需要 正负 数再dfs这样比较小的min
     */
    public int minTransfers(int[][] transactions) {
        Map<Integer, Integer> map = new HashMap<>();        // <people, amount>

        for (int[] t : transactions) {
            map.put(t[0], map.getOrDefault(t[0], 0) + t[2]);
            map.put(t[1], map.getOrDefault(t[1], 0) - t[2]);
        }

        int[] account = new int[map.size()];
        int idx = 0;

        // 把最后算好的账 总数 都放到account里
        for (int amount : map.values()) {     // 不care是谁要pay，只care金额，因为题目只care transaction
            account[idx++] = amount;
        }


        // 小优化 提速： 1.去掉已经是0的值，其实在dfs()中while也可以， 2.统计可以直接消除的两idx和为0的情况
        Arrays.sort(account);

        int preProcessResult = 0;
        int left = 0, right = account.length - 1;

        while (left < right) {
            if (account[left] == 0) {
                left++;
            } else if (account[left] + account[right] == 0) {
                preProcessResult++;
                account[left++] = 0;
                account[right--] = 0;
            } else if (account[left] + account[right] < 0) {
                left++;
            } else {
                right--;
            }
        }
        // ************* end of optimization *********************

        // optimize后 加上preprocess结果
        return preProcessResult + dfsTransaction(account, 0);
        // return dfsTransaction(account, 0);
    }

    private int dfsTransaction(int[] account, int pos) {
        int len = account.length;

        // 跳过0的，这些不需要交易
        while (pos < len && account[pos] == 0) {
            pos++;
        }

        if (pos == len)
            return 0;

        int minTransaction = Integer.MAX_VALUE;

        for (int i = pos + 1; i < len; i++) {
            // 只有正负数时，我们才抵消 这样最min transaction
            if (account[i] * account[pos] < 0) {
                account[i] += account[pos];           // 记住是 当前 d[i] + 这个参照物 d[pos], 因为for(i) 只能改i 不能改参照物
                minTransaction = Math.min(minTransaction, dfsTransaction(account, pos + 1) + 1);    // 这次交易算一次 所以 + 1
                account[i] -= account[pos];
            }
        }
        return minTransaction;
    }


    /**
     * 490. The Maze
     * 0是空地，1是墙..一个球从start开始，看能否滑到destination.
     *
     * BFS
     * 4个方向，每个方向要一直while直到碰到墙为止..
     *
     * 时间复杂度 O( m * n * max(m,n)) 可能会遍历整个矩阵，同时在每个位置有可能会要遍历最长的长度
     *
     * 也可以DFS一样的
     */
    public boolean hasPath(int[][] maze, int[] start, int[] destination) {
        int m = maze.length;
        int n = maze[0].length;

        boolean[][] visited = new boolean[m][n];
        visited[start[0]][start[1]] = true;

        Queue<int[]> q = new LinkedList<>();
        q.offer(start);

        while (!q.isEmpty()) {
            int[] cur = q.poll();

            if (cur[0] == destination[0] && cur[1] == destination[1])
                return true;

            for (int k = 0; k < 4; k++) {
                int nx = cur[0];
                int ny = cur[1];

                // 某个方向一直到墙
                while (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx][ny] == 0) {
                    nx += d[k];
                    ny += d[k + 1];
                }

                nx -= d[k];         // 记得往后退回一格
                ny -= d[k + 1];

                if (!visited[nx][ny]) {
                    visited[nx][ny] = true;
                    q.offer(new int[]{nx, ny});
                }
            }
        }
        return false;
    }


    /**
     * 505. The Maze II
     * 0是空地，1是墙..一个球从start开始。
     * find the shortest distance for the ball to stop at the destination
     * distance按照start(exclude)到dest(include)遇到的所有empty空格算
     *
     * BFS
     * queue + dist[][]
     * 需要dist[][]才知道坐标的dist, 这样新的可以跟原先的比。只有newDist < old dist[][]才push进queue
     * 同时也就不需要visited了，也不用判断是否到dest
     *
     */

    public int shortestDistance(int[][] maze, int[] start, int[] destination) {
        int m = maze.length;
        int n = maze[0].length;

        Queue<int[]> q = new LinkedList<>();
        q.offer(start);

        // 要保持每个坐标distance，这样新的distance跟旧的比..如果小于旧的那就放q里..不需要visited[][]
        int[][] dist = new int[m][n];
        for (int[] row : dist) {
            Arrays.fill(row, Integer.MAX_VALUE);
        }
        dist[start[0]][start[1]] = 0;

        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int y = cur[1];

            for (int k = 0; k < 4; k++) {
                int nx = cur[0];
                int ny = cur[1];
                int newDist = dist[x][y];

                // 某个方向一直到墙
                while (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx][ny] == 0) {
                    nx += d[k];
                    ny += d[k + 1];
                    newDist++;
                }

                nx -= d[k];         // 记得往后退回一格
                ny -= d[k + 1];
                newDist--;

                if (newDist < dist[nx][ny]) {
                    q.offer(new int[]{nx, ny});
                    dist[nx][ny] = newDist;
                }
            }
        }
        return dist[destination[0]][destination[1]] == Integer.MAX_VALUE ? -1 : dist[destination[0]][destination[1]];
    }


    /**
     * 505. The Maze II
     *
     * PQ, 比较慢
     *
     * PQ + visited[][]  同时遇到end返回
     *
     * 如果用PQ的话，保证每次都是最短的路径出来..
     * 还要保持每个坐标distance，这样新的distance跟旧的比
     */
    public int shortestDistancePQ(int[][] maze, int[] start, int[] destination) {
        int m = maze.length;
        int n = maze[0].length;

        boolean[][] visited = new boolean[m][n];

        PriorityQueue<Point> q = new PriorityQueue<>((a, b) -> a.dist - b.dist);
        // 如果用PQ的话，保证每次都是最短的路径出来..
        // 还要保持每个坐标distance，这样新的distance跟旧的比
        q.offer(new Point(start[0], start[1], 0));

        while (!q.isEmpty()) {
            Point cur = q.poll();
            int x = cur.x;
            int y = cur.y;

            if (x == destination[0] && y == destination[1])
                return cur.dist;

            if (visited[x][y])
                continue;

            visited[x][y] = true;

            for (int k = 0; k < 4; k++) {
                int nx = x;
                int ny = y;
                int newDist = cur.dist;

                // 某个方向一直到墙
                while (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx][ny] == 0) {
                    nx += d[k];
                    ny += d[k + 1];
                    newDist++;
                }

                nx -= d[k];         // 记得往后退回一格
                ny -= d[k + 1];
                newDist--;

                q.offer(new Point(nx, ny, newDist));
            }
        }
        return -1;
    }


    /**
     * 499. The Maze III
     * 打印出一条最短路径.. 如果有多条，按字母顺序返回第一条
     *
     * 那就在Point类里加条path
     *
     * 也是 PQ + visited[][]
     */
    public String findShortestWay(int[][] maze, int[] start, int[] dest) {
        int m = maze.length;
        int n = maze[0].length;

        boolean[][] visited = new boolean[m][n];

        // distance相同的话按字母顺序排
        PriorityQueue<Point> pq = new PriorityQueue<>((a, b) -> a.dist != b.dist ? a.dist - b.dist : a.path.compareTo(b.path));
        pq.offer(new Point(start[0], start[1], 0, ""));

        int[][] d = {{1, 0}, {0, -1}, {0, 1}, {-1, 0}};
        char[] moves = {'d', 'l', 'r', 'u'};

        while (!pq.isEmpty()) {
            Point cur = pq.poll();
            int x = cur.x;
            int y = cur.y;

            if (x == dest[0] && y == dest[1])
                return cur.path;

            if (visited[x][y])
                continue;

            visited[x][y] = true;

            for (int k = 0; k < 4; k++) {
                int nx = x;
                int ny = y;
                int newDist = cur.dist;

                // 某个方向一直到墙, 同时现在destination hole可能在中间，所以要检查
                while (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx][ny] == 0) {
                    nx += d[k][0];
                    ny += d[k][1];
                    newDist++;

                    if (nx == dest[0] && ny == dest[1])     // 看有没hole
                        break;
                }

                if (nx != dest[0] || ny != dest[1]) {
                    nx -= d[k][0];         // 记得往后退回一格
                    ny -= d[k][1];
                    newDist--;
                }
                            // 在这里加上 move letter
                pq.offer(new Point(nx, ny, newDist, cur.path + moves[k]));
            }
        }
        return "impossible";
    }


    class Point {
        int x;
        int y;
        int dist;
        String path;

        public Point(int x, int y, int dist) {
            this.x = x;
            this.y = y;
            this.dist = dist;
        }

        public Point(int x, int y, int dist, String path) {
            this.x = x;
            this.y = y;
            this.dist = dist;
            this.path = path;
        }
    }


    public static void main(String[] args) {
    	Graph sol = new Graph();
    	char[][] m = new char[][]{{'0','E','0','0'},{'E','0','W','E'},{'0','E','0','0'}};
    	int[][] c = new int[][]{{1,0,0,0,1},{0,0,0,0,0},{0,0,1,0,0}};
    //    System.out.println(sol.minTotalDistance(c));
    	
    	List<Integer> list1 = Arrays.asList(1, 2);
    	List<Integer> list2 = Arrays.asList(2,3);
    	List<Integer> list3 = Arrays.asList(2, 1);
    	List<List<Integer>> list = new ArrayList<>();
    	list.add(list1);
    	list.add(list2);
    	list.add(list3);
    	
    	String[][] equations = new String[][]{{"a", "b"}, {"b", "c"}};
    	double[] values = new double[]{2.0, 3.0};
    	String[][] queries = new String[][]{{"a", "c"}, {"b", "a"}};
    	
    	double[] results = sol.calcEquation1(equations, values, queries);
    	for (double d : results) {
    		System.out.println(d);
    	}



        Map<String, List<String>> map = new HashMap<>();

//        map.put("A", Arrays.asList("B", "C"));
//        map.put("B", Arrays.asList("C"));
////        map.put("C", Arrays.asList("A"));
//        map.put("D", new ArrayList<>());

        map.put("A", Arrays.asList("C"));
        map.put("C", Arrays.asList("A", "B"));

        List<String> result = sol.printDependency(map);
        System.out.println(result);
    }
    
}
