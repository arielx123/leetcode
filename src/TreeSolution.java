
import java.util.*;


class TreeNode {
	int val;
	TreeNode left;
	TreeNode right;

	TreeNode(int x) {
		val = x;
	}
}

class TreeLinkNode {
	int val;
	TreeLinkNode left;
	TreeLinkNode right;
	TreeLinkNode next;
}


/**
 * @author Yan
 *
 */
public class TreeSolution {

	/** Preorder
	 * @param root
	 * @return
	 */
	// divide & conquer
	public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
   
        if (root == null)
            return result;
            
        result.add(root.val);
        List<Integer> left = preorderTraversal(root.left);
        List<Integer> right = preorderTraversal(root.right);
        result.addAll(left);
        result.addAll(right);
        
        //     dfspreorder(result, root);
        
        return result;
    }
    
	// dfs recursion
    public void dfspreorder(List<Integer> result, TreeNode node) {
        if (node == null)
            return;
        
        result.add(node.val);
        dfspreorder(result, node.left);
        dfspreorder(result, node.right);
    }
    
    
	public List<Integer> preorderTraversalIter(TreeNode root) {
		List<Integer> preOrder = new ArrayList<>();
		Stack<TreeNode> stack = new Stack<>();

		if (root == null) {
			return preOrder;
		}

		TreeNode n = root;
		//只push父亲 。也可用与inorder
		while (n != null || !stack.isEmpty()) {
			if (n != null) {			//找到叶子为止。这样可以用while, 那下面else可以省略
				preOrder.add(n.val);	
				stack.push(n);
				n = n.left;
			} else {
				n = stack.pop();
				n = n.right;
			}
		}
		/*
		 // 简单，但是较慢
		 stack.push(root); 
		 while (!stack.isEmpty()) { 
		 	TreeNode n = stack.pop(); 
		 	preOrder.add(n.val); 
		 	if (n.right != null)	stack.push(n.right); 
		 	if (n.left != null)		stack.push(n.left); 
		 }
		 *
		
		// 只把right放进stack里
        while (node != null) {
            result.add(node.val);
            
            if (node.right != null)     //放右
                stack.push(node.right);
            
            node = node.left;
            
            if (node == null && !stack.isEmpty())   //左null了才pop右
                node = stack.pop();
        }
        */
		return preOrder;
	}
	
	/**
	 * Inorder Traversal. 跟preorder差不多
	 * 
	 * @param root
	 * @return
	 */
	public ArrayList<Integer> inorderTraversal(TreeNode root) {
		ArrayList<Integer> result = new ArrayList<Integer>();

		if (root == null) {
			return result;
		}

		Stack<TreeNode> stack = new Stack<TreeNode>();
		TreeNode node = root;
		
        while (node != null || !stack.isEmpty()) {
            while (node != null) {
                stack.push(node);		//只push父节点
                node = node.left;
            }
            node = stack.pop();		 //只先pop一次找右边，再继续找左边了
            result.add(node.val);
            node = node.right;
        }

		return result;
	}
	
	
	/** Inorder traversal - Morris.
	 * 对于每个当前node, 如果能找到node之前的一个点 那就不用回溯，这个点是 left子树的最right点。
	 * 如果left为空，print cur, 然后往right走
	 * 如果left非空，就找inorder里root的前驱节点predecessor（left里的最右）
	 * 				如果predecessor.right为空，那么让它指向root, predecessor.right = root。建thread
	 * 				如果.........非空，证明visit过了，这时就可以去掉thread线，print cur，并往右走
	 * 
	 * 这种空间复杂度是O(1).因为只需要predecessor，且没有递归没有stack。
	 * 时间复杂度是O(n), 好像是 2*n 因为右边的点会走2次，construct thread & delete thread
	 * @param root
	 * @return
	 */
	public List<Integer> inorderTraversalMorris(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null)    return result;
        
        TreeNode cur = root;
        while (cur != null) {
            if (cur.left == null) {             //如果没有left
                result.add(cur.val);        //打印
                cur = cur.right;
            } else {
                TreeNode predecessor = cur.left;
                while (predecessor.right != null && predecessor.right != cur) {
                    predecessor = predecessor.right;            //predecessor从left开始一直往右找max右
                }
                
                if (predecessor.right == null) {        //construct the thread
                    predecessor.right = cur;            // 左子树的最right 指向 current node
              // result.add(cur.val);   如果是preorder
                    cur = cur.left;                 //然后cur就继续往left走
                } else {
                    predecessor.right = null;       //thread已经存在，要Reset回null
                    result.add(cur.val);                //打印
                    cur = cur.right;
                }
            }
        }
        return result;
    }
	
	
	public List<Integer> postorderTraversalEasy(TreeNode root) {
        LinkedList<Integer> result = new LinkedList<>();        //需要是linkedlist
   
        if (root == null)     return result;
        
        Stack<TreeNode> stack = new Stack<TreeNode>();
		
		TreeNode node = root;
		while (node != null || !stack.isEmpty()) {
		    while (node != null) {
		        result.addFirst(node.val);			//这里可以addFirst，需要LinkedList才行
		        stack.push(node);
		        node = node.right;
		    }
		    node = stack.pop();
		    node = node.left;
		}
    //    Collections.reverse(result);      //如果只是result.add,而不是addFirst的话
        
		/*  慢一点的，但简单
		stack.push(root);
		
		while (!stack.isEmpty()) {
		    TreeNode node = stack.pop();
		    result.addFirst(node.val);
		    if (node.left != null)  stack.push(node.left);
		    if (node.right != null)  stack.push(node.right);
		}
		*/
		
        return result;
    }
	

	/** PostOrder Traversal
	 * 复杂。有一个prev记录访问过的点, 并且用一个stack记录根
	 * 分3情况：上往下，左下往上，右下往上（都遍历完可以加result并pop）
	 * 
	 * @param root
	 * @return
	 */
	public ArrayList<Integer> postorderTraversal(TreeNode root) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		Stack<TreeNode> stack = new Stack<TreeNode>();
		TreeNode pre = null; // previously traversed node
		TreeNode cur = root;
		stack.push(root);

		if (root == null) {
			return result;
		}

		while (!stack.isEmpty()) {
			cur = stack.peek();
			// traverse down the tree 正常：push左或右 (当左空才push右）
			if (pre == null || pre.left == cur || pre.right == cur) {
				if (cur.left != null) {
					stack.push(cur.left);
				} else if (cur.right != null) {
					stack.push(cur.right);
				}
			} else if (cur.left == pre) { // traverse up from left
				if (cur.right != null) { // push右
					stack.push(cur.right);
				}
			} else { // traverse up from right
				result.add(cur.val); // 都遍历完，加结果，并pop
				stack.pop();
			}

			pre = cur;
		}
		return result;
	}

	public ArrayList<Integer> postorderTraversalII(TreeNode root) {
		ArrayList<Integer> result = new ArrayList<Integer>();

		if (root == null) {
			return result;
		}

		result.addAll(postorderTraversal(root.left));
		result.addAll(postorderTraversal(root.right));
		result.add(root.val); // preOrder就放在最前面，inorder放中间

		return result;
	}

	
	
	/** 102. Binary Tree Level Order Traversal 用queue - 典型BFS
	 * @param root
	 * @return
	 */
	public List<List<Integer>> levelOrder(TreeNode root) {
		List<List<Integer>> result = new ArrayList<>();
		if (root == null)
			return result;
		Queue<TreeNode> queue = new LinkedList<>();
		queue.offer(root);

		while (!queue.isEmpty()) {
			List<Integer> level = new ArrayList<>();
			int size = queue.size(); // need to assign size int here
			for (int i = 0; i < size; i++) { // otherwise size() will change in loop
				TreeNode cur = queue.poll();
				level.add(cur.val);
				if (cur.left != null)   queue.offer(cur.left);
				if (cur.right != null)  queue.offer(cur.right);
			}
			result.add(level); // 如果是bottom-up就result.add(0, level)
								// 且result是LinkedList会更快
		}
		return result;
	}

	// Level Order Traversal 用dfs方法. 跟其他dfs recursion的方法很像
	public List<List<Integer>> levelOrderDFS(TreeNode root) {
		List<List<Integer>> res = new ArrayList<>();
		dfs(res, root, 0);
		return res;
	}

	public void dfs(List<List<Integer>> list, TreeNode node, int level) {
		if (node == null)
			return;

		if (list.size() == level) 			 	//到了新的一层，就要新建一个空集
			list.add(new LinkedList<Integer>());
		

		// 如果是zigzag那题 if (deep % 2 == 0) 就正常加
		list.get(level).add(node.val);			//在这层加val
		// else {....add(0, node.val);

		dfs(list, node.left, level + 1);
		dfs(list, node.right, level + 1);
	}

	
	/** 107. Binary Tree bottom-up Level Order Traversal II 
	 * 左右子树还是一样，但从bottom开始排 
	 * @param root
	 * @return
	 * 只改一句
	 * result.get(result.size() - level - 1).add(n.val); 
	 * 在前面的层里加。不能直接0。否则下面几层的加了以后，result变大了，第0个不再是之前那层
	 * 
	 * BFS的方法跟I一样，就是在result.add(0, levels)
	 */
	public List<List<Integer>> levelOrderBottom(TreeNode root) {
		List<List<Integer>> result = new LinkedList<>();
		dfsBottom(result, root, 0);
		return result;
	}

	public void dfsBottom(List<List<Integer>> result, TreeNode n, int level) {
		if (n == null) {
			return;
		}

		if (level == result.size()) { // 也可以==
			result.add(0, new LinkedList<Integer>()); // 在result最前面加 也可addFirst(v)
		}

		result.get(result.size() - level - 1).add(n.val); // 在前面的层里加。不能直接0

		dfsBottom(result, n.left, level + 1);
		dfsBottom(result, n.right, level + 1);
	}

	
	/** 103. Binary Tree Zigzag Level Order Traversal 
	 * 其实用list.add(0, cur.val)来颠倒顺序放比较好，比stack快
	 * 
	 * boolean toggle = false 来表示 result.size() % 2 == 0来对调
	 * 每层结束后result.add(list);	toggle = !toggle;
	 * 
	 * @param root
	 * 看cur要直接放level list里还是stack（之后再reverse放）但加孩子时还是放queue，因为不能变
	 */
	public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
		List<List<Integer>> result = new ArrayList<>();
		if (root == null) {
			return result;
		}

		Queue<TreeNode> q = new LinkedList<>();
		q.offer(root);
		boolean toggle = false; // 用Boolean来表示更方便

		while (!q.isEmpty()) {
			List<Integer> list = new LinkedList<>();
			int size = q.size();
			for (int i = 0; i < size; i++) {
				TreeNode cur = q.poll();
				if (!toggle) {
					list.add(cur.val);
				} else {
					list.add(0, cur.val); // reverse sequence 在这里代替stack
				}
				if (cur.left != null)  q.add(cur.left);
		        if (cur.right != null)  q.add(cur.right);
			}
			result.add(list);
			toggle = !toggle;
		}

		return result;
	}

	public ArrayList<ArrayList<Integer>> zigzagLevelOrderStack(TreeNode root) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (root == null) {
			return result;
		}
		Queue<TreeNode> queue = new LinkedList<TreeNode>();
		queue.offer(root);
		int count = 0;

		while (!queue.isEmpty()) {
			ArrayList<Integer> level = new ArrayList<Integer>();
			Stack<Integer> stack = new Stack<Integer>();
			int size = queue.size();
			for (int i = 0; i < size; i++) {
				TreeNode cur = queue.poll();
				if (count % 2 == 0) {
					level.add(cur.val);
				} else {
					stack.push(cur.val); // 放倒转顺序
				}
				// still should remain original order
				if (cur.left != null)
					queue.offer(cur.left);
				if (cur.right != null)
					queue.offer(cur.right);
			}
			if (count % 2 != 0) {
				while (!stack.isEmpty()) {
					level.add(stack.pop());
				}
			}
			result.add(level);
			count++;
		}
		return result;
	}


	/**
	 * 515. Find Largest Value in Each Tree Row - easy
	 * 普通的BFS即可
	 */
	public List<Integer> largestValues(TreeNode root) {
		List<Integer> result = new ArrayList<>();
		if (root == null)
			return result;

		Queue<TreeNode> q = new LinkedList<>();
		q.add(root);

		while (!q.isEmpty()) {
			int size = q.size();
			int max = Integer.MIN_VALUE;
			for (int i = 0; i < size; i++) {
				TreeNode node = q.poll();
				max = Math.max(max, node.val);

				if (node.left != null)  q.add(node.left);
				if (node.right != null)  q.add(node.right);
			}
			result.add(max);
		}
		return result;
	}

	// DFS
	public List<Integer> largestValuesDFS(TreeNode root) {
		List<Integer> result = new ArrayList<>();
		if (root == null)
			return result;

		dfsHelper(result, root, 0);
		return result;
	}

	private void dfsHelper(List<Integer> result, TreeNode node, int level) {
		if (node == null)
			return;

		if (level == result.size()) {			// 新加一层
			result.add(node.val);
		}

		if (node.val > result.get(level)) {		// 对比这一层
			result.set(level, node.val);
		}

		dfsHelper(result, node.left, level + 1);
		dfsHelper(result, node.right, level + 1);
	}

	
	
	/**
	 * 314. Binary Tree Vertical Order Traversal
	 * 垂直地一列列打出来。  其实对于一个node, 如果当前所在列是col, 那left就在col-1, right在col+1
	 * @param root
	 * @return
	 * 用queue来存，因为先后顺序matter，先进先出。
	 * 记得把node和对应的col同时放到queue里。然后用hashMap存col, 已经对应同一列里的node val
	 * 
	 * 1. 初始时把root和0(col)同时放queue里
	 * 2. while queue有东西，Poll出来，如果map里有这一列col, 那就加进去。
	 * 3. 然后把left，right放进去
	 * 4. 同时用min, max来记录col的左右范围
	 * 5. 最后for[min, max]， 吧map里对应的col list都加到result里
	 */
	public List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null)   return result;
        
        // 同时存node和它的col.. 如果可以写成一个TreeColNode也可以只用一个queue
        Queue<TreeNode> q = new LinkedList<>();         
        Queue<Integer> colq = new LinkedList<>();
        
        q.add(root);
        colq.add(0);       //刚开始是0，左就col-1, 右col+1
        
        int min = 0, max = 0;       //cols的左右范围
        Map<Integer, List<Integer>> map = new HashMap<>();      //存某个col对应的所有node值
        
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            int col = colq.poll();
            if (!map.containsKey(col))
                map.put(col, new ArrayList<Integer>());
            map.get(col).add(node.val);         //把node加进第col列
            
            if (node.left != null) {
                q.add(node.left);
                colq.add(col - 1);
                min = Math.min(min, col - 1);
            }
            if (node.right != null) {
                q.add(node.right);
                colq.add(col + 1);
                max = Math.max(max, col + 1);
            }
        }

        // 需要 max.. 不能按照map.size()判断
        for (int i = min; i <= max; i++) {
            result.add(map.get(i));
        }
        return result;
    }


    // 这个大部分可以，但是 不能保证 上层row 先出现..
	// 因为dfs, 可能下面的子树先放list里，之后上面的row再放，这样顺序就错了
	public List<List<Integer>> verticalOrderDFS(TreeNode root) {
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		Map<Integer, List<Integer>> map  = new HashMap<>();

		dfsVertical(map, root, 0);

		List<Integer> cols = new ArrayList<>(map.keySet());
		Collections.sort(cols);

		for (int col : cols) {
			result.add(map.get(col));
		}

		return result;
	}

	private void dfsVertical(Map<Integer, List<Integer>> map, TreeNode node, int col) {
		if (node == null)
			return;

		if (!map.containsKey(col)) {
			map.put(col, new ArrayList<>());
		}
		map.get(col).add(node.val);

		dfsVertical(map, node.left, col - 1);
		dfsVertical(map, node.right, col + 1);
	}


	/**
	 * 987. Vertical Order Traversal of a Binary Tree
	 * 跟上面那题很像，唯一区别是：
	 * 		a. 上一题 同x,y的情况下，先left再right先后顺序
	 * 	    b. 这题， 同x,y的情况下，按node.val最小的排起
	 *
	 *
	 * 所以 需要用comparator.. 要有另外的Point记录 x, y, val 这样好比较..
	 * 1. 把所有node都DFS, 变成Point放list里
	 * 2. sort这些points
	 * 3. for循环这些points，加入result即可..  prevCol变时就加new list
	 */
	public List<List<Integer>> verticalTraversal(TreeNode root) {
		List<List<Integer>> result = new ArrayList<>();
		if (root == null)
			return result;

		List<Point> points = new ArrayList<>();
		// put points in list & sort the list
		dfsPutPoints(root, 0, 0, points);
		Collections.sort(points);

		Integer prevCol = null;

		for (Point p : points) {
			if (prevCol == null || p.x != prevCol) {		// 其实也是 p.x > prevCol，新的一col, 加新list
				result.add(new ArrayList<>());
				prevCol = p.x;
			}

			result.get(result.size() - 1).add(p.val);
		}

		return result;
	}

	private void dfsPutPoints(TreeNode node, int x, int y, List<Point> list) {
		if (node == null)
			return;

		list.add(new Point(x, y, node.val));

		dfsPutPoints(node.left, x - 1, y + 1, list);
		dfsPutPoints(node.right, x + 1, y + 1, list);
	}


	class Point implements Comparable<Point>{
		int x, y, val;

		public Point(int x, int y, int val) {
			this.x = x;
			this.y = y;
			this.val = val;
		}

		@Override
		public int compareTo(Point other) {
			if (this.x != other.x) {
				return Integer.compare(this.x, other.x);
			} else if (this.y != other.y) {
				return Integer.compare(this.y, other.y);
			} else {
				return Integer.compare(this.val, other.val);
			}
		}
	}


	/**
	 * BFS 解法.. 跟上一题一样BFS.. 只是需要Point来记录x, y 这样 之后放result 前可以比较
	 * @param root
	 * @return
	 */
	public List<List<Integer>> verticalTraversalBFS(TreeNode root) {
		List<List<Integer>> result = new ArrayList<>();
		if (root == null)
			return result;

		Map<Integer, List<Point2>> map  = new HashMap<>();
		Queue<Point2> q = new LinkedList<>();
		q.add(new Point2(root, 0, 0));

		int minCol = 0, maxCol = 0;

		while (!q.isEmpty()) {
			Point2 point = q.poll();
			int col = point.x;
			int row = point.y;
			minCol = Math.min(minCol, col);         // 也可以放在left里比，稍微快点
			maxCol = Math.max(maxCol, col);

			if (!map.containsKey(col)) {
				map.put(col, new ArrayList<>());
			}

			map.get(col).add(point);

			if (point.node.left != null) {
				q.add(new Point2(point.node.left, col - 1, row - 1));
			}
			if (point.node.right != null) {
				q.add(new Point2(point.node.right, col + 1, row - 1));
			}
		}

		int idx = 0;
		for (int i = minCol; i <= maxCol ; i++) {
			// result.add(map.get(i));

			Collections.sort(map.get(i), (a, b) -> a.y == b.y ? a.node.val - b.node.val : b.y - a.y);
			result.add(new ArrayList<>());
			for (Point2 p : map.get(i)) {
				result.get(idx).add(p.node.val);
			}

			idx++;
		}
		return result;
	}

	class Point2 {
		TreeNode node;
		int x;
		int y;

		public Point2(TreeNode node, int x, int y) {
			this.node = node;
			this.x = x;
			this.y = y;
		}
	}


	/**
	 * 655. Print Binary Tree  - DFS
	 * root不会和之后的重叠.. 需要留出足够的空位.. 不是简单的col-1 之类的
	 *
	 *      1
	 *     / \
	 *    2   3
	 *     \
	 *      4
	 * Output:
	 * [["", "", "", "1", "", "", ""],
	 *  ["", "2", "", "", "", "3", ""],
	 *  ["", "", "4", "", "", "", ""]]
	 *
	 * @param root
	 * @return
	 *
	 * 需要算height。返回的值，cols看完整leaf个数：2^height - 1
	 *
	 * 然后 每次填result时， 当前row，而col是 (left + right) / 2 .. 左右边界的mid中点
	 * arr[row][(l + r) / 2] = node.val
	 *
	 * 其他地方都初始化为""即可
	 */
	public List<List<String>> printTree(TreeNode root) {
		int height = maxDepth(root);
		// 高度height，宽度是最多leaf数，2^height - 1
		String[][] arr = new String[height][(1 << height) - 1];
		for (String[] strs : arr) {
			Arrays.fill(strs, "");
		}

		dfsFill(arr, root, 0, 0, arr[0].length - 1);        // left bound & right bound

		List<List<String>> result = new ArrayList<>();
		for (String[] strs : arr) {
			result.add(Arrays.asList(strs));
		}
		return result;
	}

	private void dfsFill(String[][] arr, TreeNode node, int row, int l, int r) {
		if (node == null)
			return;

		arr[row][(l + r) / 2] = String.valueOf(node.val);

		dfsFill(arr, node.left, row + 1, l, (l + r) / 2);
		dfsFill(arr, node.right, row + 1, (l + r) / 2 + 1, r);
	}


	/**
	 * 655. Print Binary Tree - BFS
	 * 但是也要先算 height.. 才能知道叶子数和result的size
	 * 那么queue里的Param要知道l, r 边界和level
	 * @param root
	 * @return
	 */
	public List<List<String>> printTreeBFS(TreeNode root) {
        List<List<String>> result = new ArrayList<>();
        if (root == null)
            return result;

        int rows = maxDepth(root);
        int cols = (int)Math.pow(2, rows) - 1;       // 2^height - 1

        List<String> row = new ArrayList<>();
        for (int i = 0; i < cols; i++) {
            row.add("");
        }
        for (int i = 0; i < rows; i++) {
            result.add(new ArrayList<>(row));
        }

        Queue<Param> q = new LinkedList<>();
        q.add(new Param(root, 0, 0, cols));

        while (!q.isEmpty()) {
            Param p = q.poll();
            int midPos = (p.l + p.r) / 2;
            result.get(p.row).set(midPos, String.valueOf(p.node.val));

            if (p.node.left != null) {
                q.add(new Param(p.node.left, p.row + 1, p.l, midPos));
            }
            if (p.node.right != null) {
                q.add(new Param(p.node.right, p.row + 1, midPos + 1, p.r));
            }
        }

        // 如果要打印出来... string有空格的话
		/*
		while (!q.isEmpty()) {
			int[] level = new int[cols];			// 每层新建一个level
			int size = q.size();
			for (int i = 0; i < size; i++) {
				Param p = q.poll();
				int midPos = (p.l + p.r) / 2;
				result.get(p.row).set(midPos, String.valueOf(p.node.val));

				level[midPos] = p.node.val;			// 存level

				if (p.node.left != null) {
					q.add(new Param(p.node.left, p.row + 1, p.l, midPos - 1));
				}
				if (p.node.right != null) {
					q.add(new Param(p.node.right, p.row + 1, midPos + 1, p.r));
				}
			}
			// 这里打把每层的level打出来
			 for (int i = 0; i < level.length; i++) {
			     System.out.print(level[i] != 0 ? level[i] : " ");
			 }
			 System.out.println();
		}
		*/

        return result;
    }

    class Param {
        TreeNode node;
        int l, r;
        int row;

        public Param(TreeNode node, int row, int l, int r) {
            this.node = node;
            this.row = row;
            this.l = l;
            this.r = r;
        }
    }

    
	
	/** 104. Maximum Depth of Binary Tree root到leaf的最深层
	 * @param root
	 * @return
	 * Divide & Conquer
	 */
	public int maxDepth(TreeNode root) {
		if (root == null) { // leaf then return 0
			return 0;
		}
		int left = maxDepth(root.left); // if return 0 then left=0..
		int right = maxDepth(root.right);

		return Math.max(left, right) + 1; // plus root
	}
	
	// 或者模板的BFS
	public int maxDepthBFS(TreeNode root) {
        if (root == null)   return 0;
        
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        int level = 0;
        
        while (!q.isEmpty()) {
            int size = q.size();
            level++;				//每层就level++ 
            while (size-- > 0) {
                TreeNode node = q.poll();
                
                /* 求min depth的话，遇到leaf就提前return
                if (node.left == null && node.right == null) 
                    return level;
                */
                if (node.left != null)      q.add(node.left);
                if (node.right != null)     q.add(node.right);
            }
        }
        return level;
    }

	
	/** 111. Minimum Depth of Binary Tree - root到leaf的最短层
	 * @param root
	 * @return
	 * 这个比max麻烦一点，如果root的left类似于一条linkedlist, 没有right孩子，那算的是left的长度。
	 * 所以要注意除非root是叶子，才return 1， 否则要分别讨论
	 */
	public int minDepth(TreeNode root) {
		if (root == null) { // leaf
			return 0;
		}
		int leftMin = minDepth(root.left);
		int rightMin = minDepth(root.right);

		//一句话搞定的话
	//  return (left == 0 || right == 0) ? left + right + 1: Math.min(left,right) + 1;
		
		//正常情况就继续recursion..  
		if (leftMin != 0 && rightMin != 0)  return Math.min(leftMin, rightMin) + 1;
        if (leftMin != 0)                   return leftMin + 1;
        if (rightMin != 0)                  return rightMin + 1;
        return 1;               //叶子
	}
	
	

	/** 110. Balanced Binary Tree  - Bottom Up
	 * 高度差<1
	 * @param root
	 * @return 
	 * 这方法跟上面的max height 差不多  O(n)
	 * 注意需要返回max depth
	 */
	public boolean isBalanced1(TreeNode root) {
		return getDepth(root) != -1;
	}

	private int getDepth(TreeNode root) {
		if (root == null) {
			return 0;
		}
		int left = getDepth(root.left);
		int right = getDepth(root.right);

		if (left == -1 || right == -1 || Math.abs(left - right) > 1) {
			return -1;
		}
		return Math.max(left, right) + 1;
	}
	
	
	
	/** 110. Balanced Binary Tree - Top Down方法
	 * 这种要慢一点，因为每一层调用depth()都要遍历所有node，大概有logn层，所以是O(NlogN)
	 * @param root
	 * @return
	 * isBalance() 主函数里，除了call getDepth()以外，还另外call自己 isBalance()。就很慢
	 */
	public boolean isBalanced2(TreeNode root) {
		if (root == null)
			return true;
		int left = getDepth(root.left);
        int right = getDepth(root.right);
        
        //除了左右差要<=1, 还要看是不是所有node都balance
        // 有可能左左和右右balance，但中间没有分支，比如没有左右，右左
        return Math.abs(left - right) <= 1 && isBalanced2(root.left) && isBalanced2(root.right);
	}

	private int depth(TreeNode root) {
		if (root == null)
			return 0;
		return Math.max(depth(root.left), depth(root.right)) + 1;
	}
	
	

	
	/** Same Tree 
	 * 跟下面symmetric差不多，但是不用helper，因为一来就是2个node传进来。 
	 * @param p
	 * @param q
	 * @return
	 */
	public boolean isSameTree(TreeNode p, TreeNode q) {
		if (p == null && q == null)     return true;
        if (p == null || q == null)     return false;
        if (p.val != q.val)             return false;
        
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
	}
	
	
	/** Sub Tree
	 * @return
	 * 一直找相同的情况，当root.val == sub.val, 那就调用 sameTree
	 */
	public boolean isSubTree(TreeNode root, TreeNode sub) {
		if (sub == null)	return true;		//sub可为空
		if (root == null)	return false;
		
		if (root.val == sub.val)
			return isSameTree(root, sub);		//这里调用 sameTree
		
		return isSubTree(root.left, sub) || isSubTree(root.right, sub);
	}

	
	/** Symmetric Tree 
	 * Given a binary tree, check whether it is a mirror of
	 * itself (ie, symmetric around its center).
	 * 			1
	 *         / \
	 *        2   2
	 *       / \  /\
	 *      3  4  4 3
	 * @param root
	 * @return
	 * 跟same非常像.. 不同是是最后一句
	 * 
	 * 记得 最左与最右比，中间左右比
	 */
	public boolean isSymmetric2(TreeNode root) {
		if (root == null)	return true;
		return sameVal(root.left, root.right);
	}
	
	public boolean sameVal(TreeNode n1, TreeNode n2) {
        if (n1 == null && n2 == null)   return true;
        if (n1 == null || n2 == null)   return false;
        if (n1.val != n2.val)			return false;
        
        return sameVal(n1.left, n2.right) && sameVal(n1.right, n2.left);	// 最左与最右比，中间左右比
    }

	
	// Iterative 用 queue
	public boolean isSymmetric(TreeNode root) {
		if (root == null) 	return true;
		
		Queue<TreeNode> q = new LinkedList<>();
		q.offer(root);
		q.offer(root);

		while (!q.isEmpty()) {
			TreeNode t1 = q.poll();		//一次Poll 2个一起比
	        TreeNode t2 = q.poll();
	        if (t1 == null && t2 == null) continue;
	        if (t1 == null || t2 == null) return false;
	        if (t1.val != t2.val) return false;
	        q.add(t1.left);			//注意不同顺序
	        q.add(t2.right);
	        q.add(t1.right);
	        q.add(t2.left);	
		}
		return true;
	}


	/**
	 * 951. Flip Equivalent Binary Trees
	 * 看两个树是不是翻转相同.. 其实就跟上面的symmetric一样
	 */
	public boolean flipEquiv(TreeNode root1, TreeNode root2) {
		if (root1 == null && root2 == null)     return true;
		if (root1 == null || root2 == null)     return false;
		if (root1.val != root2.val)             return false;

		return (flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right))
			|| (flipEquiv(root1.left, root2.right) && flipEquiv(root1.right, root2.left));
	}
	
	
	
	/** 226. Invert Binary Tree
	 * @param root
	 * @return
	 */
	public TreeNode invertTree(TreeNode root) {
        if (root == null)   return null;
        
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        
        /* 或者swap那种
        TreeNode tempRight = root.right;
        root.right = invertTree(root.left);
        root.left = invertTree(tempRight);
         */
        
        return root;
    }
	
	
	// 可以用queue或stack，放left和right的顺序随便。 先invert，再把孩子push进去
	public TreeNode invertTreeIterative(TreeNode root) {
        if (root == null)   return null;
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            TreeNode tmpLeft = node.left;
            node.left = node.right;
            node.right = tmpLeft;
            
            if (node.left != null)      q.add(node.left);
            if (node.right != null)     q.add(node.right);
        }
        return root;
    }
	
	


	/** 112. Path Sum 
	 * determine if the tree has a root-to-leaf path 是否加起来等于sum
	 * @param root
	 * @param sum
	 * @return
	 * 记得，leaf时才可以return
	 */
	public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null)   return false;
        
        if (root.left == null && root.right == null)        //叶子
            return root.val == sum;
        
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }
	

	// 用while循环，2个queue. 分别存node和目前为止的path sum.. 
	// 跟vertical order或 Min Depth有点像，要判断leaf
	public boolean hasPathSum2(TreeNode root, int sum) {
		if (root == null)
			return false;

		Queue<TreeNode> q = new LinkedList<>();
        Queue<Integer> sumq = new LinkedList<>();
        q.add(root);
        sumq.add(root.val);
        
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            int tmp = sumq.poll();
            if (node.left == null && node.right == null) {
                if (tmp == sum)     return true;
            }
            if (node.left != null) {
                q.add(node.left);
                sumq.add(tmp + node.left.val);
            }
            if (node.right != null) {
                q.add(node.right);
                sumq.add(tmp + node.right.val);
            }
        }
		return false;
	}
	

	/** 113. Path Sum II 
	 * 打印出ArrayList结果 （有几条paths符合） 用backtracking的recursion
	 * @param root
	 * @param sum
	 * @return
	 * 记得这种path可能有左右邻居，符合target时不要return, 要去掉这个leaf回溯，可能右边邻居也行
	 */
	public List<List<Integer>> pathSumII(TreeNode root, int sum) {
		List<List<Integer>> result = new ArrayList<>();

		sumHelper(root, sum, result, new ArrayList<Integer>());

		return result;
	}

	private void sumHelper(TreeNode n, int sum, List<List<Integer>> result,
			List<Integer> path) {
		if (n == null) {
			return;
		}

		path.add(n.val);

		if (n.left == null && n.right == null && sum == n.val) {
			result.add(new ArrayList<Integer>(path));
			path.remove(path.size() - 1);				// 记得要去掉这个leaf回溯，可能右边邻居也行
			return;
		}

		sumHelper(n.left, sum - n.val, result, path);
		sumHelper(n.right, sum - n.val, result, path);

		path.remove(path.size() - 1);

	}

	
	/** 257. Binary Tree Paths
	 * 打印出所有path.. 比上面更简单
	 * @param root
	 * @return
	 */
	public List<String> binaryTreePaths(TreeNode root) {
		List<String> list = new ArrayList<>();
		treePaths(list, root, "");
		return list;
	}

	private void treePaths(List<String> list, TreeNode n, String str) {
		if (n == null) {
			return;
		}

		if (n.left == null && n.right == null) { // leaf
			list.add(str + n.val);
			return;
		}

		treePaths(list, n.left, str + n.val + "->");
		treePaths(list, n.right, str + n.val + "->");
	}

	// 用Stringbuilder写。记得要用delete才能修改sb. substring不能in place修改sb值，只能返回
	private void treePathSB(List<String> list, TreeNode n, StringBuilder sb) {
		if (n == null) {
			return;
		}

		int len = sb.length();
		if (n.left == null && n.right == null) { // leaf
			list.add(sb.append(n.val).toString());
			sb.delete(len, sb.length());
			return;
		}
		sb.append(n.val + "->");
		treePathSB(list, n.left, sb);
		treePathSB(list, n.right, sb);
		sb.setLength(len);					//跟用delete差不多
	}
	
	
	
	/**
	 * 437. Path Sum III - node to node
	 * 算出所有和为sum的path个数，不需要从root到leaf，可以任意，但是只能从上往下走
	 * @param root
	 * @param sum
	 * @return
	 * 注意  helper(root)   + pathSumIII(left)     + pathSumIII(right)
	 * 调helper来算sum，而left和right孩子需要另外再算
	 */
	public int pathSumIII(TreeNode root, int sum) {
        if(root == null)
            return 0;
        // 注意  helper(root)   + pathSumIII(left)     + pathSumIII(right)
        return helper(root, sum) + pathSumIII(root.left, sum) + pathSumIII(root.right, sum);
    }
    
    public int helper(TreeNode root, int sum) {
        if (root == null)
            return 0;

		int res = 0;

        if (sum == root.val)
            res++;
        
        res += helper(root.left, sum - root.val);
        res += helper(root.right, sum - root.val);
        
        return res;
    }

	/**
	 * 437. Path Sum III    O(n)
	 *
	 * 这里用prefix sum的原理
	 * The sum[node, cur] (target) = the difference: sum[root, cur] - prefix sum [root, node]. (node是某个点开始)
	 *
	 * 每次看map里有没存在 sum - target 这段，有的话 证明这prefixSum对应的target段存在。 看多少种方法result
	 *
	 * https://leetcode.com/problems/path-sum-iii/discuss/91878/17-ms-O(n)-java-Prefix-sum-method
	 */
	public int pathSumIII2(TreeNode root, int sum) {
		if (root == null)
			return 0;

		Map<Integer, Integer> preSum = new HashMap<>();        // <sum, ways>  prefix sum

		// 记住要有这个
		preSum.put(0, 1);

		return helper(root, 0, sum, preSum);
	}

	private int helper(TreeNode node, int sum, int target, Map<Integer, Integer> preSum) {
		if (node == null)
			return 0;

		sum += node.val;

		// sum - target 是 rootToCurSum - nodeToCurTarget = rootToNodePrefixSum
		// 如果rootToCurSum - target存在于map，证明存在 中间一段sum == target
		int result = preSum.getOrDefault(sum - target, 0);

		// 另外这里 map也要记录下 root到当前node的 prefixSum
		preSum.put(sum, preSum.getOrDefault(sum, 0) + 1);

		result += helper(node.left, sum, target, preSum);
		result += helper(node.right, sum, target, preSum);

		preSum.put(sum, preSum.get(sum) - 1);           // 记得要 backtrack !!!!

		return result;
	}
    
    
    
    /**
	 * Path Sum IV - 输出和为target的paths.. any to any node
     * 只能dfs, 像graph一样。同时要有parent，这样才能往上走.. 而且要判断 visit过没
     * @param root
     * @param target
     * @return
     * 也是要 对当前node, 先findSum。同时要pathSumIV求left和right。跟上面的pathSum III很像
     */
    public List<List<Integer>> binaryTreePathSumIV(ParentTreeNode root, int target) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        pathSumIV(result, target, root);
        return result;
    }
    
    public void pathSumIV(List<List<Integer>> result, int target, ParentTreeNode node) {
    	if (node == null)	return;
    	
    	findSum(result, target, new ArrayList<Integer>(), node, null);
    	
    	pathSumIV(result, target, node.left);
    	pathSumIV(result, target, node.right);
    }
    
    public void findSum(List<List<Integer>> result, int target, List<Integer> path, ParentTreeNode node, ParentTreeNode prev) {
    	if (node == null)	return;
    	
    	path.add(node.val);
    	target -= node.val;
    	
    	if (target == 0) {
    		result.add(new ArrayList<Integer>(path));
    	}
    	
    	if (node.parent != null && node.parent != prev) 
    		findSum(result, target, path, node.parent, node);
    	if (node.left != null && node.left != prev) 
    		findSum(result, target, path, node.left, node);
    	if (node.right != null && node.right != prev) 
    		findSum(result, target, path, node.right, node);
    	
    	path.remove(path.size() - 1);
    }
    
	
    class ParentTreeNode {
    	public int val;
    	public ParentTreeNode parent, left, right;
    }
    
    

	/** 129. Sum Root to Leaf Numbers 
	 * 每条路连起来变成数， 再加。 12 + 13 = 25，root是1，左右孩子2，3
	 * @param root
	 * @return
	 * 这个算是分治法
	 *
	 * !!!!注意!!!! 需要判断叶子...  不能叶子往下再node为null，否则会算2次，重复了
	 */
	public int sumNumbersBetter(TreeNode root) {
		return dfs(root, 0);
	}

	private int dfs(TreeNode root, int sum) {
		if (root == null)
			return 0;			// 返回0

		sum = sum * 10 + root.val;

		// 注意在这里判断叶子.. 不能 进到下一步node为null时返回sum，这样left和right会算2次sum，错的
		if (root.left == null && root.right == null)
			return sum;

		return dfs(root.left, sum) + dfs(root.right, sum); // 相加  Divide & Conquer
	}

	
	// backtracking. 慢点   如果需要打印 all叶子的数
	public int sumNumbersII(TreeNode root) {
		ArrayList<Integer> paths = new ArrayList<>();
		int sum = 0;
		pathRec(paths, root, 0);
		for (int i : paths) {
			sum += i;
		}
		return sum;
	}

	private void pathRec(ArrayList<Integer> paths, TreeNode node, int sum) {
		if (node == null) 	return;
		
		sum = sum * 10 + node.val; // 先加，再判断是不是leaf. 再加paths

		if (node.left == null && node.right == null) {
			paths.add(sum);
			return;
		}

		pathRec(paths, node.left, sum);
		pathRec(paths, node.right, sum);
	}
	

	/** Sum Root to Leaf Numbers 
	 * 用stack循环做, 模仿preorder. 但也比dfs慢
	 * @param root
	 * @return
	 * 但这个改变了每个node原先的val, 变成它以上的path sum
	 */
	public int sumNumbersIII(TreeNode root) {
		int sum = 0;
		Stack<TreeNode> stack = new Stack<>();
		stack.push(root);
		TreeNode cur;

		while (!stack.isEmpty()) {
			cur = stack.pop();
			if (cur.left != null) {
				cur.left.val = cur.val * 10 + cur.left.val; // 把之前的总和推到每个node里
				stack.push(cur.left);
			}
			if (cur.right != null) {
				cur.right.val = cur.val * 10 + cur.right.val; // update right
																// child val
				stack.push(cur.right);
			}
			if (cur.left == null && cur.right == null) { // leaf
				sum += cur.val;
			}
		}
		return sum;
	}


	/** Max path sum from root
	 * 采用max depth的算法
	 * @param root
	 * @return
	 */
	public int maxPathSumFromRoot(TreeNode root) {
		if (root == null) {
			return 0;
		}

		int left = maxPathSumFromRoot(root.left);
		int right = maxPathSumFromRoot(root.right);

		// if root to leaf
		// return Math.max(left, right) + root.val;

		// if root to any node
		return Math.max(0, Math.max(left, right)) + root.val;
		// 这个跟算max subArr一样，也是看是否大于0
	}
	
	
    
    /**
	 * 543. Diameter of Binary Tree - easy  (longest path)
     *  longest path between any two nodes in a tree. This path may or may not pass through the root.
     *  有可能某个子树很大，那就不会穿过root构成longest path
     * @param root
     * @return
     */
	int longest = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        if (root == null)   return 0;
        
        getMaxDepth(root);
        
        return longest;
    }
    
    private int getMaxDepth(TreeNode node) {
        if (node == null)   return 0;
        
        int left = getMaxDepth(node.left);
        int right = getMaxDepth(node.right);
        
        //这是longest结果的candidate，穿过root.. 正常要+1，但是题目给的是间距，不用+1
        longest = Math.max(longest, left + right);		
        
        return Math.max(left, right) + 1;
    }
    
    
    /** 不用全局变量，而用数组..
     * 也是差不多的思想
     */
    public int diameterOfBinaryTree1(TreeNode root) {
        if (root == null)
            return 0;
            
        int[] max = dfs(root);
        return max[0] > max[1] ? max[0] - 1 : max[1] - 1;
    }
    
    private int[] dfs(TreeNode node) {
        if (node == null) {
            return new int[2];
        }
        
        int[] max = new int[2];
        
        int[] leftMax = dfs(node.left);
        int[] rightMax = dfs(node.right);
        
        // m[0]存left或者right哪边的的最深深度，后面+1是加上当前node
        max[0] = Math.max(leftMax[0], rightMax[0]) + 1;      
        // m[1]存 longest path = left最大深度 + 当前node + right最大深度 或者 left/right子树
        max[1] = Math.max(leftMax[0] + rightMax[0] + 1, Math.max(leftMax[1], rightMax[1]));
        
        return max;
    }
    
    
    
    /**
	 * 543. Diameter of Binary Tree - easy   - any 2 any
     * any node之间的最长路径 ， 打印出其中一条最长的
     * @param node
     * @return
     * 全局变量有int longest & List<Integer> path
     * 
     * 只有走到最下面的leaf才知道哪个需要加到path里，所以在null时新建空list, 而不是像参数一样传进来
     */
    List<Integer> path = new ArrayList<>();
    private List<Integer> getMaxDepthPath(TreeNode node) {
        if (node == null)   
            return new ArrayList<>();
        
        List<Integer> lefts = getMaxDepthPath(node.left);
        List<Integer> rights = getMaxDepthPath(node.right);

		// 不能直接跟path.size()比，会错
        if (longest < lefts.size() + rights.size()) {
            longest = lefts.size() + rights.size();			// 需要有longest
            path.clear();
            path.addAll(lefts);
            path.add(node.val);
            path.addAll(rights);
        }
        
    //    return Math.max(left, right) + 1;
        if (lefts.size() > rights.size()) {
            lefts.add(node.val);
            return lefts;
        } else {
			rights.add(node.val);
			return rights;
		}
    }
    

	
	/**
	 * 124. Binary Tree Maximum Path Sum - any to any
	 * The path may start and end at any node in the tree.
	 * The path must contain at least one node
	 *
	 * 跟上面的diameter (longest path) 很像
	 *
	 * 注意要存 全局的max.. 这样来最后返回它。
	 * 
	 * rec时返回的是 单条path，是max(left, right) + root.val;  
	 * 因为对于每个点，需要知道left和right孩子的单条路径max。
	 * 
	 * 而结果candidate的话，是算出left+right+node 作为整个max的candidate
	 * 因为答案至少要有一个node，所以max candidate要加上自己
	 *   
	 * 不能skip root因为都要连一起
	 */
    int max = Integer.MIN_VALUE;
    
    public int maxPathSum(TreeNode root) {
        maxSinglePath(root);
        return max;
    }
    
    public int maxSinglePath(TreeNode node) {
        if (node == null)   return 0;
        
        int leftMax = Math.max(0, maxSinglePath(node.left));
        int rightMax = Math.max(0, maxSinglePath(node.right));
        
        //算结果max的candidate
        int localMax = leftMax + rightMax + node.val;
        max = Math.max(max, localMax);
        
        // return的是single path
        return Math.max(leftMax, rightMax) + node.val; 
    }
    
    
	/** 用数组存max，也是跟上面一样的做法
	 */
	public int maxPathSum1(TreeNode root) {
		int[] maxPath = new int[1]; 		// 用数组是pass by reference这样可以当全局变量
		maxPath[0] = Integer.MIN_VALUE; 	// 或者在函数外面定义int 当全局变量
		maxRec(root, maxPath); 
		return maxPath[0]; 					// 结果要max，不是rec返回值
	}

	private int maxRec(TreeNode root, int[] maxPath) {
		if (root == null)      return 0;
		
		int left = Math.max(0, maxRec(root.left, maxPath)); // 左子树single
		int right = Math.max(0, maxRec(root.right, maxPath));
		
		maxPath[0] = Math.max(maxPath[0], left + right + root.val);

		// 返回每条single path的最大，这样才能 分别 算出 从root开始的左边那条path，或者右边那条。而不是全局max[]
		return Math.max(left, right) + root.val; // 返回的单条path，而不是max!!!!
	}

	
	
	// 法2 用wrapper class
	public int maxPathSum2(TreeNode root) {
		ResultType result = helper(root);
		return result.maxPath;
	}

	private ResultType helper(TreeNode root) {
		if (root == null) {
			return new ResultType(0, Integer.MIN_VALUE);
			// single path可以不包含任何点，所以可以0；max可能负数
		}
		// divide
		ResultType left = helper(root.left);
		ResultType right = helper(root.right);

		// conquer
		// 上面传回left包含了singlePath值。在这还需加上当前root的值. 节点有可能为负数< 0，所以要单独判断
		int root2any = Math.max(0, Math.max(left.singlePath, right.singlePath) + root.val);

		int maxPath = Math.max(left.maxPath, right.maxPath); // 不加root自己的值
		// 考虑经过自己root的情况。这时要算singlePath与root相加，从而得出new maxPath
		maxPath = Math.max(maxPath, left.singlePath + right.singlePath + root.val);
		// 若让maxPath相加的话，可能是折线与折线相加，这样就会有重复计算了

		return new ResultType(root2any, maxPath);
	}

	private class ResultType {
		int singlePath, maxPath;

		ResultType(int singlePath, int maxPath) {
			this.singlePath = singlePath; // 从左或右边的一条直线的和, 可以不包括任何点。root2any
			this.maxPath = maxPath; // 可能是折线的和  any2any
		}
	}



	/**
	 * 653. Two Sum IV - Input is a BST - easy
	 *
	 * 这个也是时间 O(n)  空间是O(h)
	 * 如果取当前值作为other, 然后往后search val-other.val的node
	 *
	 * 或者可以用2个stack.. 像BST iterator一样
	 * 一个stack一直pushLeft，放最小；另一个一直pushRight放最大.. 分别从两个stack里挑值
	 *
	 */
	public boolean findTarget(TreeNode root, int k) {
		return dfs(root, root, k);
	}

	// 这种写法有点像 path sum...  取当前 那就 call后面的search()... 否则就考虑left或right的dfs当前方法
	private boolean dfs(TreeNode node, TreeNode other, int value) {
		if (other == null)
			return false;

		return search(node, other, value - other.val) ||
				   dfs(node, other.left, value) || dfs(node, other.right, value);
	}

	private boolean search(TreeNode node, TreeNode other, int value) {
		if (node == null)
			return false;

		return ((node.val == value) && node != other) ||
				   ((node.val < value) && search(node.right, other, value)) ||
				   ((node.val > value) && search(node.left, other, value));
	}


	// naive - O(n)  时间空间O(n).. 用set然后遍历，或者inorder放到list里再双指针
	public boolean findTargetEasy(TreeNode root, int k) {
		return find(root, k, new HashSet<>());
	}

	private boolean find(TreeNode node, int target, Set<Integer> set) {
		if (node == null)
			return false;

		if (set.contains(target - node.val))
			return true;

		set.add(node.val);

		return find(node.left, target, set) || find(node.right, target, set);
	}




	/** Maximum Subtree
     * Given a binary tree, every node has a int value, return the root node of subtree with the largest sum up value. 
     * @param root
     * @return
     */
    public TreeNode findSubtree(TreeNode root) {
        helper(root);
        return resultNode;
    }
    
    public TreeNode resultNode = null;
    public int maximum_weight = Integer.MIN_VALUE;


    public int maxSubHelper(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = maxSubHelper(root.left);
        int right = maxSubHelper(root.right);

        if (resultNode == null || left + right + root.val > maximum_weight) {
            maximum_weight = left + right + root.val;
            resultNode = root;
        }

        return left + right + root.val;
    }

	
	/** 235. Lowest Common Ancestor of a BST
	 * 判断是否都小于root，或都大于..其他情况就return，包括p<root<q, p是q的parent
	 * @param root
	 * @param p
	 * @param q
	 * @return
	 */
	public TreeNode lowestCommonAncestorBST(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == root || q == root)   return root;
        
        if (p.val < root.val && q.val < root.val)   return lowestCommonAncestor(root.left, p, q);
        if (p.val > root.val && q.val > root.val)   return lowestCommonAncestor(root.right, p, q);
        return root;		// 包含了 root在p, q中间，和 root == null || p == root || q == root 情况
    }
	
	
	// iteration
	public TreeNode lowestCommonAncestorBSTiteration(TreeNode root, TreeNode p, TreeNode q) {
		while (root != null) {
			if (p.val > root.val && q.val > root.val) {
				root = root.right;
			} else if (p.val < root.val && q.val < root.val) {
				root = root.left;
			} else {
				return root; // in else{},includes p is q's parent
			}
		}
		return null;
	}

	
	
	/** 236. Lowest Common Ancestor 
	 * 没有parent节点 // 在root为根的二叉树中找A,B的LCA
	 * @param root
	 * @param p
	 * @param q
	 * @return
	 * 如果找到了就返回这个LCA，刚好p, q分别在左右子树
	 * 如果只剩left子树，就返回left 
	 * 如果只剩right子树，就返回right
	 * 如果都没有，就返回null
	 *
	 *
	 * follow up, 如果a 或 b 不在tree里 怎么办？
	 *
	 * 在main() {} 里call 这方法后要判断一下return solution
	 * TreeNode sol = lowestCommonAncestor(root, a, b);
	 * if (sol != a && sol != b)	return sol;  这是OK的
	 * if (sol == a)	return lowestCommonAncestor(sol, b, b);  如果返回null就说明 b 不存在
	 * if (sol == b)	return lowestCommonAncestor(sol, a, a);  如果返回null就说明 a 不存在
	 * sol == null 那就都不在
	 *
	 * 因为 a/b == root 时返回自己，也就是说可能a或b为空..最后是返回非空的那个
	 * 那么就要查一下另一个是否null, 是就不存在
	 */
	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
		if (root == null)   return null;
        if (p == root || q == root)     return root;		//刚好其中之一是root，那就返回root
        
        // divide
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        
        // conquer
        if (left != null && right != null)  return root;
        if (left != null)                   return left;   
        if (right != null)                  return right;   
        
        return null;
	}

	
	/** lowest Common Ancestor 如果有 parent node
	 * @param node1
	 * @param node2
	 * @return
	 * 1. 分别打出各自的ancestor paths, 从n1 到 root的路径
	 * 2. 然后对比2个path list 从后往前扫，相同ancestor就继续往前，直到不同
	 */
	public ParentNode lowestCommonAncestorParent(ParentNode node1,
			ParentNode node2) {
		ArrayList<ParentNode> list1 = getPath2Root(node1);
		ArrayList<ParentNode> list2 = getPath2Root(node2);

		int i, j;
		for (i = list1.size() - 1, j = list2.size() - 1; i >= 0 && j >= 0; i--, j--) {
			if (list1.get(i) != list2.get(j)) {
				return list1.get(i).parent;
			}
		}
		return list1.get(i + 1);
	}

	private ArrayList<ParentNode> getPath2Root(ParentNode node) {
		ArrayList<ParentNode> list = new ArrayList<ParentNode>();
		while (node != null) {
			list.add(node);
			node = node.parent;
		}
		return list;
	}

	class ParentNode {
		int val;
		ParentNode left;
		ParentNode right;
		ParentNode parent;

		ParentNode(int x) {
			val = x;
		}
	}

	/** lowest Common Ancestor 用iterative 做 （没parent）
	 * 
	 * 主要思路就还是跟上面有parent值时一样， 得到ancestors的路径，然后p, q对比看看有没一样的
	 * 
	 * 但是这里没有parent值，所以需要hashmap存每个点对应的parent
	 * n1向上traverse直到root，用ancestor的set来存n1的祖先
	 * n2也再travel一遍, 看n2的parent 是否包含在n1's set里。这时就 while(n1's ancestors 不包含n2), 那n2就往上走	 * 
	 * @param root
	 * @param p
	 * @param q
	 * @return
	 */
	public TreeNode lowestCommonAncestorIter(TreeNode root, TreeNode p,
			TreeNode q) {
		Map<TreeNode, TreeNode> parent = new HashMap<>();
		Stack<TreeNode> stack = new Stack<>();

		parent.put(root, null);
		stack.push(root);

		// traverse the tree & put parents in map 帮助track parent节点, 直到都找到p & q为止
		while (!parent.containsKey(p) || !parent.containsKey(q)) {
			TreeNode n = stack.pop();
			if (n.left != null) {
				stack.push(n.left); 
				parent.put(n.left, n);		// 2nd value is parent
			}
			if (n.right != null) {
				stack.push(n.right);
				parent.put(n.right, n);
			}
		}

		// create a set of p's ancestors，相当于上面的整个ancestor list
		Set<TreeNode> ancestors = new HashSet<>();
		while (p != null) {
			ancestors.add(p); // p is its ancestor too,自己也是，所以先加进去
			p = parent.get(p);
		}

		// 看第2个点 q是否有同样的parent, 没的话就继续往上
		while (!ancestors.contains(q)) {
			q = parent.get(q);
		}

		return q;
	}

	
	/** 98. Validate Binary Search Tree 
	 * 记得要用min, max 这样才能确保subtree也对. 右subtree的左要<parent, 但也要大于root。这个要切记！！
	 * @param root
	 * @return 
	 * min, max可以Integer，也可以用TreeNode表示
	 */
	public boolean isValidBST(TreeNode root) {
		return isBST(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
	}

	private boolean isBST(TreeNode node, int min, int max) {
		if (node == null)   return true;
        
        if (node.val <= min || node.val >= max)     
            return false;
            
        return isBST(node.left, min, node.val) && isBST(node.right, node.val, max);
	}
	
	//如果节点是Integer.MIN_VALUE或者MAX的话，初始化时用null, 传入参数用Integer min(对象而不是int)
	public boolean isValid(TreeNode root, Integer min, Integer max) {
	    if (root == null) return true;

	    if (min != null && root.val <= min) return false;
	    if (max != null && root.val >= max) return false;
	    
	    return isValid(root.left, min, root.val) && isValid(root.right, root.val, max);
	}
	

	/** Validate Binary Search Tree 
	 * 用中序遍历traverse, recursion的dfs模板
	 * 用全局变量prev来记录之前的值，先left，然后比较prev，再right
	 * @param root
	 * @return
	 */
	TreeNode prev = null;	//要用全局变量，不能作为参数传进去. 否则rec里面改了，但跳回原先层，prev还是之前一样的没有变
	public boolean isValidBSTInorder(TreeNode root) {
		// inorder traversal
		if (root == null)
			return true;

		if (!isValidBST(root.left)) // traverse left first
			return false;

		if (prev != null && root.val <= prev.val) // compare with previous value
			return false;

		prev = root;

		return isValidBST(root.right);
	}

	
	/** Validate Binary Search Tree 
	 * 用中序遍历traverse, iterative, stack模板. 只push父亲节点（没有right）
	 * 思路跟上面差不多。先一直找到最左的，这样是最小的数，然后往外pop并比较pre的值
	 * @param root
	 * @return
	 * 不能把所有node都push进去，否则很难用pre来比较。
	 */
	public boolean isValidBSTInorderStack(TreeNode root) {
		// inorder traversal
		Stack<TreeNode> stack = new Stack<>();
		TreeNode cur = root;
		TreeNode pre = null;			//要有pre才能比较

		while (cur != null || !stack.isEmpty()) {
			while (cur != null) { // find the most left leaf
				stack.push(cur);
				cur = cur.left;
			}
			cur = stack.pop();
			if (pre != null && cur.val <= pre.val) {
				return false; // compare with the previous val
			}
			pre = cur;
			cur = cur.right;
		}
		return true;
	}
	
	
	/** 230. Kth Smallest Element in a BST
	 * 又是inorder iterative模板
	 * @param root
	 * @param k
	 * @return
	 */
	public int kthSmallest(TreeNode root, int k) {
        if (root == null || k <= 0) {
            return -1;
        }
        
        Stack<TreeNode> s = new Stack<>();
        TreeNode cur = root;
        while (cur != null || !s.isEmpty()) {
            while (cur != null) {
                s.push(cur);
                cur = cur.left;
            }
            cur = s.pop();
            if (--k == 0)	break;			//  之前inorder里是list.add(cur.val);
            cur = cur.right;
        }
        return cur.val;
    }
	
	
	int count = 0;
    int res = Integer.MIN_VALUE;
    
    public int kthSmallestRec(TreeNode root, int k) {
        count = k;
        traverse(root);
        return res;
    }
	
    public void traverse(TreeNode node) {
    	if (node == null)   return;
        
        inorder(node.left);
        if (--count == 0) {
            result = node.val;
            return;
        }
        inorder(node.right);
    }


	/**
	 * 671. Second Minimum Node In a Binary Tree
	 * 每个node有0或2个孩子，并且 root == min(left, right). 找出第二小的数..
	 *
	 * 有可能root == left == right，要一直往下找
	 */

	// 不用额外的 min 参数
	public int findSecondMinimumValue1(TreeNode root) {
		if (root.left == null)
			return -1;

		int leftMin = root.left.val == root.val ? findSecondMinimumValue(root.left) : root.left.val;
		int rightMin = root.right.val == root.val ? findSecondMinimumValue(root.right) : root.right.val;

		if (leftMin != -1 && rightMin != -1)    return Math.min(leftMin, rightMin);
		else                                    return Math.max(leftMin, rightMin);
	}

	public int findSecondMinimumValue(TreeNode root) {
		return dfsSecondMin(root, root.val);
	}

	// 用额外的参数min来track
	private int dfsSecondMin(TreeNode root, int min) {
		if (root.left == null)
			return root.val == min ? -1 : root.val;

		if (root.val > min)
			return root.val;

		// else , root == left == right
		int  leftMin = dfsSecondMin(root.left, min);
		int  rightMin = dfsSecondMin(root.right, min);

		if (leftMin != -1 && rightMin != -1)    return Math.min(leftMin, rightMin);
		else if (leftMin != -1)                 return leftMin;
		else                                    return rightMin;
	}

    

    
	/** 108. Convert Sorted Array to Binary Search Tree 
	 * are sorted in ascending order, convert it to a height balanced BST
	 * 找中点，这顺序跟In order一样。 中点就保证是balanced 然后前一半再找中点，recursion
	 * 
	 * @param num
	 * @return
	 */
	public TreeNode sortedArrayToBST(int[] num) {
		if (num.length == 0 || num == null) {
			return null;
		}
		return sortArrBSTRec(num, 0, num.length - 1); // length - 1
	}

	private TreeNode sortArrBSTRec(int[] arr, int low, int high) {
		if (low > high) {
			return null;
		}
		int mid = low + (high - low) / 2;
		TreeNode cur = new TreeNode(arr[mid]);
		cur.left = sortArrBSTRec(arr, low, mid - 1);
		cur.right = sortArrBSTRec(arr, mid + 1, high);

		return cur;
	}
	

	// 很慢，而且用3个stack，先建node，存左右index，再算mid来替换val
	public TreeNode sortedArrayToBSTIterative(int[] nums) {
		TreeNode root = new TreeNode(-1); // need to replace val
		Stack<TreeNode> stack = new Stack<TreeNode>();
		stack.push(root);

		Stack<Integer> leftIdx = new Stack<Integer>();
		leftIdx.push(0);

		Stack<Integer> rightIdx = new Stack<Integer>();
		rightIdx.push(nums.length - 1);

		while (!stack.isEmpty()) {
			TreeNode cur = stack.pop();
			int left = leftIdx.pop();
			int right = rightIdx.pop();
			int mid = left + (right - left) / 2;
			System.out.println(mid);
			cur.val = nums[mid];				//因为push是node为-1，所以pop出来要还成n[mid]

			if (left <= mid - 1) {
				cur.left = new TreeNode(-1);
				stack.push(cur.left);
				leftIdx.push(left);
				rightIdx.push(mid - 1);
			}
			if (right >= mid + 1) {
				cur.right = new TreeNode(-1);
				stack.push(cur.right);
				leftIdx.push(mid + 1);
				rightIdx.push(right);
			}
		}
		return root;
	}
	

	/** 109. Convert Sorted List to Binary Search Tree
	 * @param head
	 * @return
	 */

	ListNode cur; // 定义为全局变量较方便。若当参数传进去会麻烦。
					// 因为左子树完head=head.next. 然后右子树传进head时变成后面一个。

	public TreeNode sortedListToBST(ListNode head) {
		if (head == null) {
			return null;
		}
		cur = head;
		ListNode n = head;
		int size = 0;
		
		while (n != null) {		// calculate the length of list
			size++;
			n = n.next;
		}
		return sortListBSTRec(0, size - 1);

		// return buildTree(size);
	}

	private TreeNode sortListBSTRec(int low, int high) {
		if (low > high) {
			return null;
		}
		int mid = low + (high - low) / 2;
		// inorder traversal。先建左子树，然后new根，再建右子树。然后连起来
		TreeNode left = sortListBSTRec(low, mid - 1); // build left sub-tree
		
		TreeNode parent = new TreeNode(cur.val); // build parent
		parent.left = left;
		cur = cur.next; // move to next so that it can be added
		
		TreeNode right = sortListBSTRec(mid + 1, high);
		parent.right = right;
		return parent;
	}

	/** 或者这种 1s
	 * 顺序也是不能乱，因为cur跟着一步步走，所以要先left，再建根，再right
	 * @param n
	 * @return
	 */
	private TreeNode buildTree(int n) {
		if (n == 0) {
			return null;
		}
		
		TreeNode node = new TreeNode(-1);
		node.left = buildTree(n / 2);
		node.val = cur.val;				 // after building left sub tree, cur移到mid，这时才赋值
		cur = cur.next;
		node.right = buildTree(n - n/2 - 1); // the right half (minus the
												// root)
		return node;
	}
	

	/** 方法2，好理解。 不过每次要重新找中点，比较慢
	 *  跟 sorted Array 一样
	 * @param head
	 * @return
	 */
	public TreeNode sortedListToBST2(ListNode head) {
		if (head == null)
			return null;
		return listBSTrec(head, null);
	}

	// 在区间[start, end]里递归，后面的end是包括在内的，这样可以避免要多用一个指针来记录mid前的节点
	private TreeNode listBSTrec(ListNode start, ListNode end) {
		if (start == end) {
			return null;
		}
		ListNode mid = findMiddle(start, end);
		TreeNode parent = new TreeNode(mid.val);
		parent.left = listBSTrec(start, mid);
		parent.right = listBSTrec(mid.next, end);

		return parent;
	}

	private ListNode findMiddle(ListNode start, ListNode end) {
		ListNode slow = start, fast = start;
		while (fast != end && fast.next != end) { // 跟end比
			fast = fast.next.next;
			slow = slow.next;
		}
		return slow;
	}

	
	/**
	 * Recover Binary Search Tree
	 *
	 * 2 elements of a binary search tree (BST) are
	 * swapped by mistake.Recover the tree without changing its structure. A
	 * solution using O(n) space is pretty straight forward. 
	 * 普通用法是用queue放输出结果
	 * Could you devise a constant space solution? 用TreeNode pre, fir,
	 * sec来存放。只用了O(3)的space复杂度
	 * 把他想成inorder序列 1,2,6,4,5,3,7 当pre > root,就找到first = pre;
	 * 之后的pre往后走，直到pre > root，遇到7就停下
	 * @param root
	 */
	TreeNode pre = null;
	TreeNode first = null;
	TreeNode second = null;
	//没法作为rec的参数传进去。因为每个rec里虽然会变，但是跳出去以后，这些局部变量就弹出栈清空了。
	// 所以如果只是void的话，参数里面是不会改变什么的

	public void recoverTree(TreeNode root) {
		if (root == null) {
			return;
		}
		inorderHelper(root); // inorder traverse the bst.
		// 或者用inorderMorrisHelper(root);
		
		int tmp = first.val;
		first.val = second.val;
		second.val = tmp;
	}
	
	private void inorderHelper(TreeNode root) {
		if (root == null) {
			return;
		}

		inorderHelper(root.left); // go to left first

		if (pre != null && pre.val > root.val) {
			if (first == null) { // the 1st to find first, never found before
				first = pre;
			}
			second = root; // 已找到1st.找2nd得情况. prev > second(root) 所以是=root
							// 别放在else{}, 如果刚好这2个元素挨着的话就直接赋值。
		}
		pre = root; // 继续搜.

		inorderHelper(root.right); // then go to right
	}
	
	
	// inorder traversal用stack模板
	public void recoverTreeIterative(TreeNode root) {
	     //   inorder(root);
	        TreeNode first = null;
	        TreeNode second = null;
	        TreeNode pre = null;
	        TreeNode cur = root;
	        
	        Stack<TreeNode> stack = new Stack<>();
	        while (cur != null || !stack.isEmpty()) {
	            if (cur != null) {
	                stack.push(cur);
	                cur = cur.left;
	            } else {
	                cur = stack.pop();
	                // 操作
	                if (pre != null && pre.val > cur.val) {
	                    if (first == null) {
	                        first = pre;
	                    } 
	                    second = cur;
	                }
	                pre = cur;
	                
	                cur = cur.right;	//模板
	            }
	        }
	        
	        int tmp = first.val;    //only swap val, not node
	        first.val = second.val;
	        second.val = tmp;
	    }
	
	
	/** Recover BST 用MORRIS TRAVERSAL来做
	 * 如果left为空，print root, root=root.right
	 * 如果left非空，就找inorder里root的前驱节点tmp（left里的最右）
	 * 				如果tmp.right为空，那么让它指向root, tmp.right = root
	 * 				如果.........非空，证明visit过了，这时就可以去掉thread线，print root，并往右走
	 * https://discuss.leetcode.com/topic/29161/share-my-solutions-and-detailed-explanation-with-recursive-iterative-in-order-traversal-and-morris-traversal/2
	 * @param root
	 */
	public void recoverTreeMorris(TreeNode root) {
	  //  inorder(root);
		TreeNode first = null;
		TreeNode second = null;
		TreeNode pre = null;
		TreeNode inorderPre = null;  //rightmost node in left subtree

		//morris traversal
		while (root != null) {
			if (root.left == null) {
				// originally print root(traversal) - replace for this question
				if (pre != null && pre.val > root.val) {
					if (first == null) {
						first = pre;
					}
					second = root;
				}
				pre = root;

				root = root.right;
			} else {
				inorderPre = root.left;
				while (inorderPre.right != null && inorderPre.right != root) {
					inorderPre = inorderPre.right;  //find the previous node of root inorder
				}

				if (inorderPre.right != null) {     //visited
					// originally print root(traversal) - replace for this question
					if (pre != null && pre.val > root.val) {
						if (first == null) {
							first = pre;
						}
						second = root;
					}
					pre = root;

					inorderPre.right = null;
					root = root.right;
				} else {
					inorderPre.right = root;
					root = root.left;
				}
			}
		}

		int tmp = first.val;    //only swap val, not node
		first.val = second.val;
		second.val = tmp;
	}


	/**
	 * 450. Delete Node in a BST
	 * 先找key node，再删...  记得用nextMin替换
	 * @param root
	 * @param key
	 * @return
	 */
	public TreeNode deleteNode(TreeNode root, int key) {
		if (root == null)
			return null;

		if (root.val < key) {
			root.right = deleteNode(root.right, key);
		} else if (root.val > key) {
			root.left = deleteNode(root.left, key);
		} else {                    // == found

			if (root.left == null) {                    // left空
				return root.right;
			} else if (root.right == null) {            // right空
				return root.left;
			}
			// has 2 children..                         // 有2孩
			TreeNode rightMin = findMin(root.right);
			root.val = rightMin.val;
			root.right = deleteNode(root.right, rightMin.val);

		}
		return root;
	}


	/**
	 * 450. Delete Node in a BST
	 *
	 * 用iterative方法
	 */
	public TreeNode deleteNodeIterative(TreeNode root, int key) {
		if (root == null)
			return null;

		TreeNode node = root;
		TreeNode parent = null;

		// find the node
		while (node != null && node.val != key) {
			parent = node;

			if (node.val < key) {
				node = node.right;
			} else {
				node = node.left;
			}
		}

		if (node == null)       // not found key
			return root;

		if (parent == null) {
			return deleteRootNode(node);
		} else if (parent.left == node) {
			parent.left = deleteRootNode(node);
		} else {
			parent.right = deleteRootNode(node);
		}

		return root;
	}

	private TreeNode deleteRootNode(TreeNode node) {
		if (node == null)           return null;
		if (node.left == null)      return node.right;
		if (node.right == null)     return node.left;

		// has 2 children
		TreeNode parent = node;
		TreeNode nextMin = node.right;
		while (nextMin.left != null) {
			parent = nextMin;
			nextMin = nextMin.left;
		}

		node.val = nextMin.val;     // swap val

		if (parent.left == nextMin) {
			parent.left = nextMin.right;    // move nextMin's right subtree up
		} else {
			parent.right = nextMin.right;   // 没有更小的了，刚开始的node.right, 直接跳过nextMin用它的right
		}
		return node;
	}


	private TreeNode findMin(TreeNode node) {
		while (node.left != null) {
			node = node.left;
		}
		return node;
	}
	
	
	
	/** 114. Flatten Binary Tree to Linked List 
	 *  直观 Divide & Conquer
	 * 先分别flatten left和right, 再连起来
	 * @param root
	 */
	public void flatten(TreeNode root) {
        if (root == null)   return ;
        
        flatten(root.left);
        flatten(root.right);
        
        TreeNode right = root.right;
        root.right = root.left;
        root.left = null;
        
        while (root.right != null) {        //一直找从左子树过来的最后点
            root = root.right;
        }
            
        root.right = right;
     }
	
	
	/** Flatten - morris
	 * 先找到左子树的rightmost child, 这样能连右子树
	 * 左子树变成右子树，之后root再往右（往前移）
	 * @param root
	 */
	public void flattenIterative(TreeNode root) {
		while (root != null) {
			if (root.left != null) {
				TreeNode pre = root;
				while (pre != null) {
					pre = pre.right;	// find the right most child in left
				}
				pre.right = root.right;	//left的最右子->root右连起来
				root.right = root.left;	  //左子树变成右
				root.left = null;
			}
			root = root.right;
		}
	}
	
	

	/** 114. Flatten Binary Tree to Linked List 
	 * 但是有全局变量不大好
	 * @param root
	 */
	TreeNode last = null;   //上一个node
	public void flatten2(TreeNode root) {
		if (root == null)	return;

		if (last != null) {
			last.left = null;
			last.right = root;
		}

		last = root;
		TreeNode right = root.right; // 先存临时变量，否则后面会变
		flatten(root.left);
		flatten(right);
	}
	

	// 可以把last变成参数传，刚开始初始为null
	public TreeNode flatten(TreeNode root, TreeNode last) {
		if (root == null) {
			return last; // return last而不是null
		}

		if (last != null) {
			last.left = null;
			last.right = root;
		}

		last = root;
		TreeNode right = root.right;
		last = flatten(root.left, last);
		return flatten(right, last);
	}
	
	
	// 用preorder+stack做
	public void flatten3(TreeNode root) {
        if (root == null)
            return;
            
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        
        while (!stack.isEmpty()) {
            TreeNode cur = stack.pop();
            if (cur.right != null) {
                stack.push(cur.right);
            }
            if (cur.left != null) {
                stack.push(cur.left);
            }
            if (!stack.isEmpty()) {     //if only 1 in stack which popped
                cur.right = stack.peek();
            }
            cur.left = null;
        }
	}

	
	
	/** 199. Binary Tree Right Side View
	 * 一棵树，从右边看过去，返回list。右边的树叶会遮住同一层的，如果左边更长，就加左边的
	 * @param root
	 * @return
	 *
	 * 这根level order很像
	 */
	public List<Integer> rightSideViewBFS(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null)   return result;
        
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = q.poll();
                if (i == 0) {
                    result.add(node.val);
                }
                if (node.right != null)   q.offer(node.right);
                if (node.left != null)    q.offer(node.left);		//这里不能else..有可能左子树后面更深
            }
        }
        return result;
    }
	
	// 跟level order的DFS 递归 很像
	public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        rightView(result, root, 0);
        return result;
    }
    
    public void rightView(List<Integer> result, TreeNode node, int depth) {
        if (node == null)   return;
        
        if (depth == result.size())				//在新的一层level直接加node
            result.add(node.val);
        
        rightView(result, node.right, depth + 1);
        rightView(result, node.left, depth + 1);
    }
	
    
	/**116. Populating Next Right Pointers in Each Node - Perfect二叉树
	 * 假定是perfect binary tree  1 -> NULL
	 * 							/ \ 
	 * 						   2 -> 3 -> NULL 
	 *     					  / \  / \ 
	 *     					 4->5->6->7 -> NULL
	 * 
	 * @param root
	 * 每次走parent的时候，顺便把下一层的孩子连上，还要tmpLink也一起解决 
	 * 
	 * 因为兄弟的孩子们不知道怎么连，所以用tmpLink记录parent.right，使5->6 每层开始再把tmpLink设空。
	 * 如果不为空，就可以连兄弟left
	 */
    	// Recursion 递归 好理解 
 	public void connectdfs(TreeLinkNode root) {
 		if (root == null) {
 			return;
 		}

 		// left child -> right
 		if (root.left != null) {
 			root.left.next = root.right;
 		}

 		// right -> left . root.next is the brother of root
 		if (root.right != null && root.next != null) {
 			root.right.next = root.next.left;
 		}

 		connectdfs(root.left);
 		connectdfs(root.right);
 	}
 	
 	
 	/** 116. Populating Next Right Pointers in Each Node - Perfect二叉树
 	 * @param root
 	 * 跟上面的递归一样，只是用while写.. 
 	 * 最外层是 while (root != null) ... 然后  root往left向下走.. 中间那层都是node向next
 	 */
	public TreeLinkNode connect(TreeLinkNode root) {
		if (root == null)
			return null;

		TreeLinkNode cur = root;
		TreeLinkNode nextLeftMost = null;

		while (cur.left != null) {
			nextLeftMost = cur.left;

			while (cur != null) {
				if (cur.left != null) {
					cur.left.next = cur.right;

					cur.right.next = cur.next == null ? null : cur.next.left;

					cur = cur.next;         // 往next走
				}
			}

			cur = nextLeftMost;
		}

		return root;
	}
	
	
	/**
	 * 117. Populating Next Right Pointers in Each Node II
	 * 任意二叉树
	 *
	 * @param root
	 * 这时要用几个变量
	 * head为下一层开始的点，这样iterate这层以后 root = head 往下移
	 * pre代表这一层的previous node，这样可以连起来
	 */
	public void connectII(TreeLinkNode root) {
        if (root == null)   return;
        
		TreeLinkNode cur = root;  			//current node of current level
		TreeLinkNode pre = null;  			//previous node of 当前level
		TreeLinkNode nextLeftMost = null; 	//head of the next level

        while (cur != null) {			// 需要是node，不能是root
			nextLeftMost = null;
			pre = null;

            while (cur != null) {          //iterate current level
                if (cur.left != null) {
                    if (pre != null) {
                        pre.next = cur.left;
                    } else {
                        nextLeftMost = cur.left;		//前面没pre就是头了
                    }
                    pre = cur.left;		 // 记得更新pre
                }
                
                if (cur.right != null) {
                    if (pre != null) {
                        pre.next = cur.right;
                    } else {
                        nextLeftMost = cur.right;
                    }
                    pre = cur.right;			// 记得更新pre
                }
                
                // move to next node，同一层
                cur = cur.next;
            }
            
            // move to next level
            cur = nextLeftMost;

        }
    }
	

	/** Recursion
	 * !!!! 先处理右子树right
	 * 再处理 左子树 left
	 * 
	 * @param root
	 */
	private void connectNodesII(TreeLinkNode root) {
		if (root == null)
			return;

		if (root.left != null) {
			root.left.next = root.right != null ? root.right : findNext(root.next);
		}

		if (root.right != null) {
			root.right.next = findNext(root.next);
		}

		connectNodesII(root.right);       // 先把right 连起来
		connectNodesII(root.left);
	}

	private TreeLinkNode findNext(TreeLinkNode node) {
		if (node == null)
			return node;

		if (node.left != null)      return node.left;
		if (node.right != null)     return node.right;

		return findNext(node.next);
	}
	
	

	/** 105. Construct Binary Tree from Preorder and Inorder Traversal 建树
	 * @param preorder
	 * @param inorder
	 * @return
	 * preorder的start建root，在inorder里找这个root所在的pos, 这样能分成左右两边，分治处理
	 * 分的时候inorder容易，毕竟pos在inorder里。但preorder的范围要注意
	 */
	public TreeNode buildTree(int[] preorder, int[] inorder) {
		if (inorder.length != preorder.length) {
			return null;
		}
		return buildTreeHelper(inorder, 0, inorder.length - 1, preorder, 0,
				preorder.length - 1);
		// len - 1. 因为是包含的
	}

	private TreeNode buildTreeHelper(int[] inorder, int instart, int inend,
			int[] preorder, int prestart, int preend) {
		if (instart > inend) { // 也可以判断prestart, preend
			return null;
		}
				// 根据 preOder的 prestart来建根。|| post根据postorder[postend]建根
		TreeNode root = new TreeNode(preorder[prestart]);
		
				// 找inorder里面的根，这样就能分left子树和right子树
		int pos = findRootPos(inorder, instart, inend, preorder[prestart]);
				// 也可以用hashmap存(inorder[i], i)来得到index，这快点
			/* 或者直接就
			 * int pos = ins;
		        while (inorder[pos] != root.val) {
		            pos++;              // find root in inorder
		        }
			 */
		
		root.left = buildTreeHelper(inorder, instart, pos - 1, preorder, prestart + 1, (pos - instart) + prestart);
				// preEnd = (pos - instart) + prestart
				// 因为pos是root，pos-instart算inorder里左边个数（范围），再加上prestart算是preorder的范围
		root.right = buildTreeHelper(inorder, pos + 1, inend, preorder,	prestart + pos - instart + 1, preend);
				// prestart = preend + (pos - inend + 1)因为最前面删除一个root所以要往后一位+1
				// prestart还可以从-start开始算，跟上面left的end一样只是+1. preS = pos - instart + prestart + 1

			/*
			 后序post order & inorder 
			 root.left = buildTreeHelper(inorder, instart, pos - 1, postorder, poststart, (pos - instart) + poststart - 1); 
			 //postend = (pos - instart) + poststart - 1 因为最后面删除一个root，所以-1
			 root.right = buildTreeHelper(inorder, pos + 1, inend, postorder, (pos - inend) + postend, postend - 1);
			 */
		return root;
	}

	// find root in inorder 找根
	private int findRootPos(int[] in, int start, int end, int key) {
		for (int i = start; i <= end; i++) {
			if (in[i] == key) {
				return i;
			}
		}
		return -1;
	}
	
	
	/** Construct tree
	 * @param preorder
	 * @param inorder
	 * @return
	 * Keep pushing the nodes from the preorder into a stack (and keep making the tree by adding nodes to the left of the previous node) 
	 * until the top of the stack matches the inorder.
	 * At this point, pop the top of the stack until the top does not equal inorder (keep a flag to note that you have made a pop).
	 * Repeat 1 and 2 until preorder is empty. 
	 */
	public TreeNode buildTreeIter(int[] preorder, int[] inorder) {
	    if (preorder.length == 0) return null;
	    Stack<TreeNode> s = new Stack<>();
	    TreeNode root = new TreeNode(preorder[0]), cur = root;
	    for (int i = 1, j = 0; i < preorder.length; i++) {
	        if (cur.val != inorder[j]) {
	            cur.left = new TreeNode(preorder[i]);
	            s.push(cur);
	            cur = cur.left;
	        } else {
	            j++;
	            while (!s.empty() && s.peek().val == inorder[j]) {
	                cur = s.pop();
	                j++;
	            }
	            cur = cur.right = new TreeNode(preorder[i]);
	        }
	    }
	    return root;
	}
	
	
	
	/**
	 * 270. Closest Binary Search Tree Value   - O(logN)
	 * given double target, 找最接近的值。
	 * @param root
	 * @param target
	 * @return
	 * 需要一直while找，因为最接近的可能在left子树的最右 leaf那
	 */
	public int closestValue(TreeNode root, double target) {
        double min = Math.abs(root.val - target);
        int res = root.val;
         
        while (root != null) {
            if (root.val == target)
                return root.val;
            
            if (Math.abs(root.val - target) < min) {
                min = Math.abs(root.val - target);
                res = root.val;
            }

            root = root.val > target ? root.left : root.right;
        }
        return res;
    }
	
	
	// 用recursion做，当前值curVal VS 孩子最小childVal
	public int closestValueRec(TreeNode root, double target) {
		int curVal = root.val;
        TreeNode child = curVal > target ? root.left : root.right;
        
        if (child == null)		//之后没孩子，那就只能这个node是答案
            return curVal;
        
        int childVal = closestValue(child, target);
        
        return Math.abs(curVal - target) < Math.abs(childVal - target) ? curVal : childVal;
	}



	/**
	 * 272. Closest Binary Search Tree Value II - 慢... 看后面的优化
	 * 找前k个最接近target的数
	 * given一个BST..中序排完后是 1，2，5，6，7，18，20. 找离5最近的就是5，6，7
	 *
	 * 最简单的就是traverse一下，放到maxHeap里面比较
	 * O(nlogk + n). 慢
	 */
	public List<Integer> closestKValuesSlow(TreeNode root, double target, int k) {
		List<Integer> result = new ArrayList<>();
		PriorityQueue<Integer> maxHeap = new PriorityQueue<>(k, (a, b) -> Double.compare(Math.abs(b - target), Math.abs(a - target)));

		inOrderToHeap(maxHeap, root, k);

		return new ArrayList<>(maxHeap);
	}

	private void inOrderToHeap(PriorityQueue<Integer> maxHeap, TreeNode node, int k) {
		if (node == null)
			return;

		inOrderToHeap(maxHeap, node.left, k);

		maxHeap.offer(node.val);
		if (maxHeap.size() > k) {
			maxHeap.poll();
		}

		inOrderToHeap(maxHeap, node.right, k);
	}


	/**
	 * 272. Closest Binary Search Tree Value II - O(n)
	 * 找前k个最接近target的数
	 * given一个BST..中序排完后是 1，2，5，6，7，18，20. 找离5最近的就是5，6，7
	 * @param root
	 * @return
	 *
	 * 其实既然上面也是inorder, 并不需要用heap来check，用简单的 k 窗口维护就行
	 * 只用维护一个k 大小的窗口，左右移动。
	 *
	 * 大小 < k时继续加，== k 时就要比较1st和cur node跟target的差距
	 * a. 如果1st 的diff 大于当前cur，那就removeFirst, 因为隔得最远
	 * b. 否则就直接return了.. 提前 结束，因为node后面更大
	 */
	public List<Integer> closestKValuesII(TreeNode root, double target, int k) {
		LinkedList<Integer> res = new LinkedList<>();
		inorderCollect(root, target, k, res);
		return res;
	}

	public void inorderCollect(TreeNode root, double target, int k, LinkedList<Integer> res) {
		if (root == null)
			return;

		inorderCollect(root.left, target, k, res);

		if (res.size() == k) {
			//if size k, add curent and remove head if it's optimal, otherwise return
			if (Math.abs(target - root.val) < Math.abs(target - res.peekFirst())) {
				res.removeFirst();            //因为最前面的差距最大
			} else {
				return;            //找完了..后面的差距更大，不会是结果
			}
		}

		res.add(root.val);			//记得加进去

		inorderCollect(root.right, target, k, res);
	}

	
	/**
	 * 272. Closest Binary Search Tree Value II
	 *
	 * 这个复杂度是  O(k + logN)
	 * @param root
	 * @param target
	 * @param k
	 * @return
	 *
	 * 以target为目标点，找最接近的smaller和bigger值，
	 * 那么最接近的肯定是 left的max或者right的min..接下来是第二max或第二min..
	 * 那么可以用smaller和bigger分别存candidate, 每次pop完一个，就继续push candidate..直到result为k为止
	 *
	 * 这样复杂度就变成tree的高度 logN
	 *
	 * 用stack是因为 越往后 越接近target值..
	 * target分别跟smaller和bigger比，哪个接近就放进result，同时再更新stack加入更多candidate
	 *
	 * 然后像merge sort一样来对比这两个stack .. merge 2 stack min的头, 同时不停地更新 predecessor和successor组成的stack
	 * 注意node.val == target的情况，也要加进stack里，但是只加其中一个就行，不能两个都加否则重复
	 */
	public List<Integer> closestKValues(TreeNode root, double target, int k) {
		List<Integer> result = new ArrayList<>();
		Stack<TreeNode> smaller = new Stack<>();
		Stack<TreeNode> bigger = new Stack<>();

		pushPredecessor(root, target, smaller);
		pushSuccessor(root, target, bigger);

		TreeNode cur = null;

		while (result.size() < k) {
			if (smaller.isEmpty() || (!bigger.isEmpty() && target - smaller.peek().val > bigger.peek().val - target)) {
				cur = bigger.pop();
				result.add(cur.val);
				pushSuccessor(cur.right, target, bigger);		// 继续更新bigger，因为可能后面稍微大点的是candidate
			} else {								// 找right是因为之前bigger已经找完left子树了，接下来要找right子树的最小
				cur = smaller.pop();
				result.add(cur.val);
				pushPredecessor(cur.left, target, smaller);
			}
		}

		return result;
	}

	private void pushPredecessor(TreeNode node, double target, Stack<TreeNode> stack) {
		while (node != null) {
			if (node.val < target) {
				stack.push(node);
				node = node.right;
			} else {
				node = node.left;
			}
		}
	}

	private void pushSuccessor(TreeNode node, double target, Stack<TreeNode> stack) {
		while (node != null) {
			if (node.val >= target) {
				stack.push(node);
				node = node.left;
			} else {
				node = node.right;
			}
		}
	}


	// 跟上面很像... 只在最开始 inorder.. while里面不会再push predecessor或successor
	public List<Integer> closestKValues1(TreeNode root, double target, int k) {
        List<Integer> result = new ArrayList<>();
        
        // 用stack来存 root之前的pre和之后的succ..因为是从最远的(最左或右)开始加，所以越近的越先pop
        Stack<Integer> pre = new Stack<>();     //predecessors
        Stack<Integer> succ = new Stack<>();     //successors
        
        inorder(root, target, false, pre);		// 以root为中点找	
        inorder(root, target, true, succ);
        
        // compare 2 stacks and pick the smaller one (merge sort)
        while (k-- > 0) {
            if (pre.isEmpty()) {
                result.add(succ.pop());
            } else if (succ.isEmpty()) {
                result.add(pre.pop());
            } else if (Math.abs(target - pre.peek()) < Math.abs(target - succ.peek())) {
                result.add(pre.pop());
            } else {
                result.add(succ.pop());
            }
        }
        return result;
    }
    
    public void inorder(TreeNode node, double target, boolean reverse, Stack<Integer> stack) {
        if (node == null)   return;
        
        inorder(reverse ? node.right : node.left, target, reverse, stack);
        
        // predecessor一直找到 中间的node为止，successor也是必须 > target, 否则就停下. 这样只用差不多一半
        if ((!reverse && node.val > target) || (reverse && node.val <= target))
            return;                         // 可以pre或succ任意一个包含==的情况，不能漏掉
        stack.push(node.val);
        
        inorder(reverse ? node.left : node.right, target, reverse, stack);
    }




	/** 285. Inorder Successor in BST
	 * 有可能p的parent就是它的successor
	 * @param root
	 * @param p
	 * @return
	 * 只有go left时，succ才有可能是root。
	 * 如果找到==或right的话，都不用更新succ
	 *
	 * 要么一直在右边，那succ就是null..
	 * 要么是个折线，在左子树的右边。只要一左，那succ就很可能是当前node，因为是左子树的最小的larger than最右
	 */
	public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
		TreeNode succ = null;
		while (root != null) {
			if (root.val > p.val) {
				succ = root;   				 // only when go left, the succ change to cur
				root = root.left;
			} else {    					// when == or < , just right
				root = root.right;
			}
		}
		return succ;
	}

	// recursion方法
	public TreeNode inorderSuccessorRec(TreeNode root, TreeNode p) {
		if (root == null) {
			return null;
		}

		if (root.val > p.val) {
			TreeNode succ = inorderSuccessor(root.left, p);
			return (succ != null) ? succ : root;
		} else {
			return inorderSuccessorRec(root.right, p);
		}
	}


	// 如果是求predecessor
	public TreeNode inorderPredecessorBST (TreeNode root, TreeNode p) {
		TreeNode pre = null;
		while(root != null) {
			if(root.val < p.val) {
				pre = root;
				root = root.right;
			}
			else root = root.left;
		}
		return pre;
	}

	// 如果是求predecessor  - recursion
	public TreeNode predecessor(TreeNode root, TreeNode p) {
		if (root == null)
			return null;

		if (root.val >= p.val) {
			return predecessor(root.left, p);
		} else {
			TreeNode right = predecessor(root.right, p);
			return (right != null) ? right : root;
		}
	}


	/**
	 * 510. Inorder Successor in BST II
	 * 只给要找的node，以及有parent node.. 但是不给root
	 * @param x
	 * @return
	 *
	 * 那就看有没right孩子，分2种情况找
	 */
	public ParentTreeNode inorderSuccessor(ParentTreeNode x) {
		ParentTreeNode succ = null;

		if (x.right != null) {      // find left most
			succ = x.right;
			while (succ.left != null) {
				succ = succ.left;
			}
			return succ;

		} else {                    // 只能往上找
			succ = x.parent;
			while (succ != null && succ.val < x.val) {
				succ = succ.parent;
			}
			return succ;
		}
	}

	// followup - 不lookup value
	// 还是跟上面一样的思路。   判断 往上找parent时，当前是不是parent的left.. 是的话就是parent了
	public ParentTreeNode inorderSuccessor2(ParentTreeNode x) {

		if (x.right != null) {              // find left most 一样
			x = x.right;
			while (x.left != null) {
				x = x.left;
			}
			return x;

		} else {							// 只能往上找
			while (x != null) {
				if (x.parent == null) {
					return null;
				} else if (x.parent.left == x) {		// 查是不是parent的left即可
					return x.parent;
				} else {
					x = x.parent;
				}
			}
			return x;
		}
	}


	/**
	 * 701. Insert into a Binary Search Tree
	 * 把val 插入到BST里..
	 *
	 * 简单...  只要加到叶子就行..
	 */
	public TreeNode insertIntoBST(TreeNode root, int val) {
		if (root == null)
			return new TreeNode(val);

		TreeNode node = root;
		TreeNode parent = null;

		while (node != null) {
			parent = node;
			if (node.val > val) {
				node = node.left;
			} else {
				node = node.right;
			}
		}

		if (parent.val < val) {
			parent.right = new TreeNode(val);
		} else {
			parent.left = new TreeNode(val);
		}

		return root;
	}

	// Recursion 做法
	public TreeNode insertIntoBST1(TreeNode root, int val) {
		if (root == null)
			return new TreeNode(val);

		if (val > root.val) {
			root.right = insertIntoBST1(root.right, val);
		} else {
			root.left = insertIntoBST1(root.left, val);
		}

		return root;
	}

    
	
	/**
	 * 96. Unique Binary Search Trees
	 * Given n, how many structurally unique BST's
	 * (binary search trees) that store values 1...n?
	 * @param n
	 * @return 
	 * Let a[n] = number of unique BST's given values 1..n, then
     * a[n] = a[0] * a[n-1]     // put 1 at root, 2...n right
     *      + a[1] * a[n-2]     // put 2 at root, 1 left, 3...n right
     *      + ...
     *      + a[n-1] * a[0]     // put n at root, 1...n-1 left
	 * Therefore: Count[i] = ∑ Count[0...k] * [ k+1....i]         0<=k<i-1
	 */
	public int numTrees(int n) {
		int[] count = new int[n + 1];
		count[0] = 1; // empty tree, just 1 type

		for (int i = 1; i <= n; i++) {	//i ~ n
			for (int j = 0; j < i; j++) {			// j做root，左右两边的可能性相乘
				count[i] += count[j] * count[i - j - 1];
			}
		}
		
		//或者这个比较好理解
		int[] dp = new int[n + 1];
		dp[0] = dp[1] = 1;      
        
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {			//j作为root
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
		return count[n];
	}

	
	/**
	 * 95. Unique Binary Search Trees II 输出所有结果
	 * @param n
	 * @return 
	 * 1. 这道题的思路就是对1..n中的每一个数都依次让它做root
	 * 2. 分出左右区间递归求左子树和右子树的所有可能性
	 * 3. 最后把左右区间求得的子结果依次分别作为root的左右孩子
	 * Divide & Conquer
	 * 
	 * 注意一点.. start > end时可以终止，但是要先加 result.add(null), 再返回result
	 * 因为可以避免判断只有左子树 或 只有右子树的情况，这样依然能继续生成树
	 */
	public ArrayList<TreeNode> generateTrees(int n) {
		return dfsG(1, n);
	}

	private ArrayList<TreeNode> dfsG(int start, int end) {
		// 其实AL里放root，下面还连着子树
		ArrayList<TreeNode> result = new ArrayList<TreeNode>();

			// AL里面添加null，使得每个AL里面至少有一个元素（null）。这样可以避免判断只有左区间或只有右区间的情况。
		if (start > end) {
			result.add(null); 		// 终止条件，到底部加null结束树
			return result;
		}

		for (int i = start; i <= end; i++) { // 1..n中的每一个数都依次让它做root
							// 一次次分，最后start>end 才不分，因为是sorted array, 自动符合BST条件
			ArrayList<TreeNode> left = dfsG(start, i - 1); // left和right至少会有一个元素null！
			ArrayList<TreeNode> right = dfsG(i + 1, end);
			for (TreeNode l: left) { 				// 因为size至少为1(null)，所以左右都会被访问到
				for (TreeNode r: right) {
					TreeNode root = new TreeNode(i); 	//每次都要new root
					root.left = l;
					root.right = r; 	// 这些存顶点，但下面连着子树
					result.add(root);
				}
			}
		}
		return result;
	}

	

	static int result, num;

	public static int kNodeBST(TreeNode root, int k) {
		num = k;
		inorder(root);
		return root.val;
	}

	private static void inorder(TreeNode node) {
		if (node == null)
			return;
		inorder(node.left);
		if (num == 0)
			result = node.val;
		inorder(node.right);
	}
	
	
	/**
	 * 298. Binary Tree Longest Consecutive Sequence
	 * 找从parent到child最长的连续sequence的 max length. 但不能反过来孩子到parent.. 连续的3456这种
	 * @param root
	 * @return
	 * 很简单的dfs . 跟上一个值target比较就好
	 */
	public int longestConsecutive(TreeNode root) {
        if (root == null)   return 0;
        
        return dfsLength(root, 1, 1, root.val + 1);
    }
    
	// 返回的是 max val
    public int dfsLength(TreeNode node, int max, int curPath, int target) {
        if (node == null)
            return max;
        
        if (node.val == target) {
            curPath++;
            max = Math.max(max, curPath);
        } else {
            curPath = 1;
        }
        
        int left = dfsLength(node.left, max, curPath, node.val + 1);
        int right = dfsLength(node.right, max, curPath, node.val + 1);
        
        return Math.max(left, right);
    }


	/**
	 * 549. Binary Tree Longest Consecutive Sequence II
	 * 跟上题一样，但是可以increasing 或 decreasing
	 */
	int maxLen = 0;

	public int longestConsecutiveII(TreeNode root) {
		if (root == null)
			return 0;

		maxHelper(root);

		return maxLen;
	}

	private int[] maxHelper(TreeNode node) {
		if (node == null)
			return new int[]{0, 0};

		int inc = 1, dec = 1;

		int[] left = maxHelper(node.left);
		int[] right = maxHelper(node.right);

		if (node.left != null) {
			if (node.left.val == node.val + 1)  inc += left[0];
			if (node.left.val == node.val - 1)  dec += left[1];
		}

		if (node.right != null) {
			if (node.right.val == node.val + 1)  inc = Math.max(inc, 1 + right[0]);
			if (node.right.val == node.val - 1)  dec = Math.max(dec, 1 + right[1]);
		}

		maxLen = Math.max(maxLen, inc + dec - 1);

		return new int[]{inc, dec};
	}
    
    
    /** 366. Find Leaves of Binary Tree  --- Linkedin 面筋
     * 每次吹掉所有叶子，一层层剥开.. 最后return叶子list
     * @param root
     * @return
     * 跟level order很像
     * bottom up来算max depth.. leaf的深度是0，往上就+1. 
     * 这样depth一样相当于同一层，在每层level把node放进去
     * 
     * 因为不知道leaf在哪里，而且是0层，所以要返回int level
     */
    public List<List<Integer>> findLeaves(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        helper(result, root);
        return result;
    }
    
    public int helper(List<List<Integer>> result, TreeNode node) {
        if (node == null)   return -1;				//记得是-1. 
        			//这里不需要判断是否叶子，因为即使是叶子，也需要后面的new list, add(node), 所以要合在一起，这里就不另外判断
        
        int left = helper(result, node.left);
        int right = helper(result, node.right);
        int depth = Math.max(left, right) + 1;      //跟求max depth一样
        
        if (depth == result.size()) {
            result.add(new ArrayList<Integer>());
        }
        result.get(depth).add(node.val);            // 叶子的depth是0
        
        node.left = node.right = null;          // 真正remove leaf，可有可无
        
        return depth;
    }
    
    
    
    /** 366. Find Leaves of Binary Tree 
     * @param root
     * @return
     * 没算深度。。一层层剥洋葱..每层找到leaf后设成null来去掉叶子.. recursion  这个是remove leaf
     */
    public List<List<Integer>> findLeavesBFS(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        
        while (root != null) {
            List<Integer> leaves = new ArrayList<>();
            root = removeLeaf(root, leaves);	// 记得要设root.最后一次等root在removeLeaf()之后变为null，这样while循环才会停
            result.add(leaves);
        }
        return result;
    }
    
    public TreeNode removeLeaf(TreeNode node, List<Integer> leaves) {
        if (node == null)
        	return null;
        
        if (node.left == null && node.right == null) {
            leaves.add(node.val);
            return null;
        }
        
        node.left = removeLeaf(node.left, leaves);
        node.right = removeLeaf(node.right, leaves);

        return node;
    }

    // 跟上面很像，只是这里 return Boolean看是否为leaf
	public List<List<Integer>> findLeaves1(TreeNode root) {
		List<List<Integer>> result = new ArrayList<>();

		while (root != null) {
			List<Integer> leaves = new ArrayList<>();
			// root = removeLeaves(root, leaves);
			if(isLeaf(root, leaves)) {
				root = null;                // 记得，最后一次等root是leaf时，要设为null，这样while循环才会停
			}
			result.add(leaves);
		}
		return result;
	}

	private boolean isLeaf(TreeNode node, List<Integer> list) {
		if (node.left == null && node.right == null) {
			list.add(node.val);
			return true;
		}

		if (node.left != null) {
			if (isLeaf(node.left, list)) {
				node.left = null;
			}
		}
		if (node.right != null) {
			if(isLeaf(node.right, list)) {
				node.right = null;
			}
		}

		return false;
	}




	/** 404. Sum of Left Leaves
     * 所有left叶子的sum
     * @param root
     * @return
     * recursion.. 在parent角度，看下面是不是叶子.. 不用pre来存
     * 先看left孩子，再看right孩子
     */
    public int sumOfLeftLeavesRec(TreeNode root) {
        if (root == null)   return 0;
        int sum = 0;
        
        if (root.left != null) {		//看left来自
            if (root.left.left == null && root.left.right == null) {
                sum += root.left.val;
            } else {
                sum += sumOfLeftLeavesRec(root.left);
            }
        }
        
        sum += sumOfLeftLeavesRec(root.right);         // 再看right
        
        return sum;
    }
    
    // 或者刚开始isLeft=false, 把isLeft当参数传进来
    public int sumLeftLeaves(TreeNode root, boolean isLeft) {
        if (root == null)   return 0;
        if (root.left == null && root.right == null)
            return isLeft ? root.val : 0;
        
        return sumLeftLeaves(root.left, true) + sumLeftLeaves(root.right, false);
    }
    
    
    
    /** 404. Sum of Left Leaves
     * interatIve方法。也可以用q,都一样因为没order只分
     */
    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null)   return 0;
        
        int sum = 0;
        Stack<TreeNode> s = new Stack<>();
        s.push(root);
        
        while (!s.isEmpty()) {
            TreeNode parent = s.pop();
            if (parent.left != null) {		//left孩子
                if (parent.left.left == null && parent.left.right == null) {    //left是叶子
                    sum += parent.left.val;
                } else {
                    s.push(parent.left);
                }
            }
            if (parent.right != null) {   		//right孩子
                s.push(parent.right);
            }
        }
        return sum;
    }
    
    
    int min = Integer.MAX_VALUE;
    TreeNode prevv = null;
    
    /** 530. Minimum Absolute Difference in BST
     * 找任意2个node的最小差..
     * @param root
     * @return
     * 最小差的话，那么就把BST根据inorder来，这样是sorted，跟上一个或下一个比就行
     * O(n)
     */
    public int getMinimumDifferenceBST(TreeNode root) {
        if (root == null)   return 0;
        
        getMinimumDifference(root.left);
        
        if (prevv != null) {
            min = Math.min(min, root.val - prevv.val);
        }
        prevv = root;
        
        getMinimumDifference(root.right);
        
        return min;
    }
    
    
    TreeSet<Integer> treeSet = new TreeSet<>();
    
    /** Minimum Absolute Difference - 普通树
     * @param root
     * @return
     * 普通树的话，不像BST.. 
     * 但也是尽量找nearest最近的值，这样能知道最小的差是多少
     * 所以用TreeSet<Integer>, (注意是int不是node, 因为要找int最近的).
     * 那么nearset就是floor或者ceiling
     * 时间复杂度 O(nlogn), 空间是O(n)
     */
    public int getMinimumDifference(TreeNode root) {
        if (root == null)   return 0;
        
        // 找离最近的floor和ceiling，跟node的差 和min比
        if (!treeSet.isEmpty()) {
            if (treeSet.floor(root.val) != null) {
                min = Math.min(min, Math.abs(root.val - treeSet.floor(root.val)));
            }
            if (treeSet.ceiling(root.val) != null) {
                min = Math.min(min, Math.abs(root.val - treeSet.ceiling(root.val)));
            }
        }
        
        // preoder. 
        treeSet.add(root.val);
        getMinimumDifference(root.left);
        getMinimumDifference(root.right);
        
        return min;
    }
    
    
    /** 501. Find Mode in Binary Search Tree
     * mode表示出现最多的元素
     * @param root
     * @return
     * 最简单的就是用HashMap.. 然后用全局变量max来keep出现最多的。
     * 
     * 如果不用额外space，就是下面这种
     * 需要inorder 2次
     * 1. 第一次 找到max count
     * 2. 	然后Reset中间值
     * 3. 第二次 把modeCount个出现最多的放进int[]里
     */
    public int[] findMode(TreeNode root) {
        inorder2(root);          // 1st to find max mode count
        
        modes = new int[modeCount];     // reset for 2nd round
        modeCount = 0;
        curCount = 0;			//这样 2nd轮不会进去 curCount > maxCount的情况
        
        inorder2(root);
        return modes;           // 2nd to put all modes
    }
    
    int preVal = 0;
    int curCount = 0;
    int maxCount = 0;
    int modeCount = 0;
    int[] modes;
    
    public void handleVal(int val) {
        if (val != preVal) {
        	preVal = val;
            curCount = 0;       //现在这个val出现过几次
        }
        curCount++;
        
        if (curCount > maxCount) {
            maxCount = curCount;
            modeCount = 1;
        } else if (curCount == maxCount) {
            if (modes != null) {                //只是2nd round
                modes[modeCount] = val;
            }
            modeCount++;
        }
    }
    
    public void inorder2(TreeNode node) {
        if (node == null)   return;
        inorder2(node.left);
        handleVal(node.val);        //handle stuff
        inorder2(node.right);
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
	 * 然后pop一个node以后，把right子树的left也push进去. 这样才能保证next会是right子树里left most最小的
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
	
	
	
	/** 297. Serialize and Deserialize Binary Tree 
	 * 用string表示tree，反过来用node表示成tree
	 * @param root
	 * @return
	 *
	 * 用Preorder recursion 会比较简单
	 */
	public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        serialize(root, sb);
        return sb.toString();
    }
    
    private void serialize(TreeNode node, StringBuilder sb) {
        if (node == null) {
            sb.append("X").append(",");     //记得加"," 才能连到后面
			return;
        }

		sb.append(node.val).append(",");
		serialize(node.left, sb);
		serialize(node.right, sb);
    }

    // 用Queue来放nodes，这样poll了以后就空了，不占用过多空间
    public TreeNode deserialize(String data) {
     //   String[] arr = data.split(",");
        Queue<String> nodesQ = new LinkedList<>();
        nodesQ.addAll(Arrays.asList(data.split(",")));
        return deserialize(nodesQ);
    }
    
    private TreeNode deserialize(Queue<String> nodesQ) {
        String val = nodesQ.poll();
        if (val.equals("X"))    
            return null;
            
        TreeNode node = new TreeNode(Integer.parseInt(val));
        node.left = deserialize(nodesQ);
        node.right = deserialize(nodesQ);
        return node;
    }
    
    
    
    /** 297. Serialize and Deserialize Binary Tree 
     * @param root
     * @return
     * BFS方法 level order
     */
    public String serializeBFS(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        if (root == null)   return "";
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            if (node == null) {
                sb.append("X,");
                continue;
            }
            sb.append(node.val).append(",");
            q.add(node.left);
            q.add(node.right);
        }
        return sb.toString();
    }
    
    // 这的Queue<TreeNode>放的是node，这样后面才能poll出来比较好连。不能跟DFS一样string
    public TreeNode deserializeBFS(String data) {
        if (data.equals(""))    return null;
        String[] arr = data.split(",");
        
        Queue<TreeNode> q = new LinkedList<>();     //存刚创建好的TreeNode
        TreeNode root = new TreeNode(Integer.parseInt(arr[0]));
        q.add(root);
        
        // 每次都创建 左右2个点.. 如果null就不用
        for (int i = 1; i < arr.length; i++) {
            TreeNode node = q.poll();
            if (!arr[i].equals("X")) {
                TreeNode left = new TreeNode(Integer.parseInt(arr[i]));
                node.left = left;
                q.add(left);
            }
            i++;			//i 记得往后走一步
            if (!arr[i].equals("X")) {
                TreeNode right = new TreeNode(Integer.parseInt(arr[i]));
                node.right = right;
                q.add(right);
            }
        }
        return root;
    }
    
    
    
    /**
	 * 428. Serialize and Deserialize N-ary Tree  多叉树
     * @return
     * 用DFS
     *   	1 
	      / | \
	     2  3  4
	       / \  \
	      5   6  7
	 * serialize结果是 (1(2)(3(5)(6))(4(7)))  用()代表孩子
     */
	public String serialize(NTreeNode root) {
		if (root == null)
			return "";

		StringBuilder sb = new StringBuilder();
		serialize(root, sb);
		return sb.toString();
	}

	private void serialize(NTreeNode root, StringBuilder sb) {
		if (root == null) {
			return;
		}

		sb.append(root.val).append(" ");

		if (root.children != null && root.children.size() != 0){
			sb.append("[ ");
			for (NTreeNode node : root.children){
				serialize(node, sb);
			}
			sb.append("] ");
		}
	}

	// Decodes your encoded data to tree.
	public NTreeNode deserialize2(String data) {
		if (data == "" || data.trim() == "")
			return null;

		String[] arr = data.trim().split(" ");
		Stack<NTreeNode> stack = new Stack<>();

		NTreeNode root = new NTreeNode(Integer.parseInt(arr[0]));
		NTreeNode cur = root;

		for (int i = 1; i < arr.length; i++) {
			if (arr[i].equals("[")) {				// 上一轮的cur，到这有[ 那么就是这个 i 的parent了，push进去
				stack.push(cur);
			} else if (arr[i].equals("]")) {		// 所有children弄完，pop出parent
				stack.pop();
			} else {
				cur = new NTreeNode(Integer.parseInt(arr[i]));		// 记得也要 更新cur, 而不是直接add就行
				stack.peek().children.add(cur);
			}
		}
		return root;
	}


	/**
	 * 428. Serialize and Deserialize N-ary Tree  多叉树
	 *
	 * 用 parent, 孩子size，children xxxx.. 直接 , 隔开
	 */
	public String serialize1(NTreeNode root) {
		if (root == null)
			return "";

		List<String> list = new ArrayList<>();
		serialize1(root, list);
		return String.join(",", list);
	}

	private void serialize1(NTreeNode root, List<String> list) {
		if (root == null) {
			return;
		}

		list.add(String.valueOf(root.val));
		list.add(String.valueOf(root.children.size()));     // 加入children个数来区分

		for (NTreeNode node : root.children){
			serialize1(node, list);
		}
	}


	public NTreeNode deserialize1(String data) {
		if (data == "")
			return null;

		String[] arr = data.split(",");
		Queue<String> q = new LinkedList<>(Arrays.asList(arr));
		return deserialize1(q);
	}

	private NTreeNode deserialize1(Queue<String> q) {
		NTreeNode root = new NTreeNode(Integer.parseInt(q.poll()));
		int size = Integer.parseInt(q.poll());

		for (int i = 0; i < size; i++) {
			root.children.add(deserialize1(q));
		}

		return root;
	}
	
    
    class NTreeNode {
    	List<NTreeNode> children;
    	int val;
    	
    	public NTreeNode (int v) {
    		val = v;
    		children = new ArrayList<>();
    	}
    	
    	public void addChildren (NTreeNode n) {
    		children.add(n);
    	}
    	
    	public List<NTreeNode> getChildren() {
    		return children;
    	}
    }


	/**
	 * 429. N-ary Tree Level Order Traversal - easy
	 * @param root
	 * @return
	 */
	public List<List<Integer>> levelOrder(NTreeNode root) {
		List<List<Integer>> result = new ArrayList<>();
		if (root == null)
			return result;

		Queue<NTreeNode> q = new LinkedList<>();
		q.add(root);

		while (!q.isEmpty()) {
			int size = q.size();
			List<Integer> level = new ArrayList<>();
			for (int i = 0; i < size; i++) {
				NTreeNode node = q.poll();
				level.add(node.val);

				for (NTreeNode child : node.children) {
					q.add(child);
				}
			}
			result.add(level);
		}
		return result;
	}


	/**
	 * 449. Serialize and Deserialize BST
	 *
	 * 这题不同的是 要as compact as possible, 因为BST特性，可以判断上下边界.. 不需要存null作为 # 到preorder string里
	 * encode的跟普通树一样..
	 *
	 */

	// pre-order
	private void serializeBST(TreeNode root, StringBuilder sb) {
		if (root == null) {
			return;
		}
		sb.append(root.val).append(" ");

		serializeBST(root.left, sb);
		serializeBST(root.right, sb);
	}

	public TreeNode deserializeBST(String data) {
		if (data.isEmpty())
			return null;

		Queue<String> q = new LinkedList<>();
		q.addAll(Arrays.asList(data.split(",")));
		return deserialize(q, Integer.MIN_VALUE, Integer.MAX_VALUE);
	}

	private TreeNode deserialize(Queue<String> q, int min, int max) {
		if (q.isEmpty())
			return null;

		String s = q.peek();
		int val = Integer.parseInt(s);

		if (val < min || val > max)			// BST 特性判断上下边界.. 所以serialize时无需建X null
			return null;

		q.poll();
		TreeNode root = new TreeNode(val);

		root.left = deserialize(q, min, val);
		root.right = deserialize(q, val, max);
		return root;
	}

	// 法2
	public TreeNode deserializeBST1(String data) {
		if (data == null || data.length() == 0) {
			return null;
		}
		String[] nodes = data.split(" ");
		int[] index = new int[] {0};
		return buildBST(nodes, index, Integer.MAX_VALUE);
	}

	private TreeNode buildBST(String[] preorder, int[] idx, int max) {
		int i = idx[0];
		if (i == preorder.length || Integer.parseInt(preorder[i]) > max)
			return null;

		TreeNode root = new TreeNode(Integer.parseInt(preorder[i]));
		idx[0]++;

		root.left = buildBST(preorder, idx, root.val);
		root.right = buildBST(preorder, idx, max);

		return root;
	}

    
    
    /** 536. Construct Binary Tree from String
     * Input: "4(2(3)(1))(6(5))"
		Output: return the tree root node representing the following tree:
		
		       4
		     /   \
		    2     6
		   / \   / 
		  3   1 5   
     * @param s
     * @return
     * DFS判断左右子树.. 用 leftPar来++或--。当它为0时，证明这个subtree结束，可以dfs
     */
    public TreeNode str2tree(String s) {
        if (s == null || s.length() == 0)
            return null;
        
        int firstPar = s.indexOf("(");
        int val = firstPar == -1 ? Integer.parseInt(s) : Integer.parseInt(s.substring(0, firstPar));
        
        TreeNode root = new TreeNode(val);
        if (firstPar == -1)
            return root;
        
        int count = 0, start = firstPar;
        for (int i = start; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
            	count++;
            } else if (s.charAt(i) == ')') {
            	count--;
            }
            
            // 处理subtree
            if (count == 0) {
                if (start == firstPar) {		// 记得用firstPar区分
                    root.left = str2tree(s.substring(start + 1, i));
                    start = i + 1;          // start挪到后面的right子树
                } else {
                    root.right = str2tree(s.substring(start + 1, i));
                }
            }
        }
        return root;
    }
    
    
    
    /** 536. Construct Binary Tree from String
     * 用stack，只存node..
     * 每次遇到')'，就说明孩子end了，那就pop一个孩子出来..基本在stack里的都是parent
     * 最后也是只剩root在stack里
     */
    public TreeNode str2tree1(String s) {
        Stack<TreeNode> stack = new Stack<>();
        int len = s.length();
        for (int i = 0; i < len; i++) {
            char c = s.charAt(i);
            if (c == ')') {
                stack.pop();        //每次pop掉孩子(peek都是parent)
            } else if (Character.isDigit(c) || c == '-') {
                int j = i;
                while (i + 1 < len && Character.isDigit(s.charAt(i+1))) {
                    i++;
                }
                int num = Integer.parseInt(s.substring(j, i + 1));
                TreeNode node = new TreeNode(num);
                if (!stack.isEmpty()) {
                    TreeNode parent = stack.peek();
                    if (parent.left == null) {      //先加left孩子
                        parent.left = node;
                    } else {                        //如果已经有left，就加right
                        parent.right = node;
                    }
                }
                stack.push(node);
            }
        }
        return stack.isEmpty() ? null : stack.peek();
    }
    
    
    

    
    /** 545. Boundary of Binary Tree
     * 返回逆时针的boundary，从root开始左到下，然后右边从下往上
     * @param root
     * @return
     * 这个方法比较naive，但清晰.. 
     * 先找左边界，然后加leaf，再加右边界
     *    preorder   preorder    postorder
     * 找3次.. 比较麻烦
     */
    public List<Integer> boundaryOfBinaryTree(TreeNode root) {
        if (root == null)   return list;
        
        list.add(root.val);
        if (root.left != null)  getLeftBoundary(root.left);
        if (root.left != null || root.right != null)    getLeaves(root);
        if (root.right != null)     getRightBoundary(root.right);
        
        return list;
    }
    
    List<Integer> list = new ArrayList<>();
    
    private void getLeftBoundary(TreeNode root) {
        if (root == null)   return;
        if (root.left == null && root.right == null)    return;
        
        list.add(root.val);
        if (root.left != null)      getLeftBoundary(root.left);
        else if (root.right != null)    getLeftBoundary(root.right);
    }   
    
    private void getLeaves(TreeNode root) {
        if (root == null)   return;
        
        if (root.left == null && root.right == null) {
        	list.add(root.val);
            return;
        }
        
        getLeaves(root.left);   
        getLeaves(root.right);
    }
    
    private void getRightBoundary(TreeNode root) {
        if (root == null)   return;
        if (root.left == null && root.right == null)    return;
        
        if (root.right != null)      getRightBoundary(root.right);
        else if (root.left != null)    getRightBoundary(root.left);
        list.add(root.val);       //之后再加root
    }  
    
    
    
    /** 545. Boundary of Binary Tree
     * 这个只是走一次O(n)
     * 0 - root, 1 - left boundary node, 2 - right boundary node, 3 - middle node.
     * 左list的话，遇到root或left boundary或leaf都加到leftList里
     * right boundary加到rightList
     * 
     * middle node不考虑
     * 每次往左走时，可能left孩子为空，如果right非空，可能是left boundary
     * 反之亦然
     */
    public List<Integer> boundaryOfBinaryTree1(TreeNode root) {
        List<Integer> leftList = new LinkedList<>();
        List<Integer> rightList = new LinkedList<>();
        
        preorder(root, leftList, rightList, 0);
        
        leftList.addAll(rightList);
        return leftList;
    }
    
    private void preorder(TreeNode cur, List<Integer> leftList, List<Integer> rightList, int flag) {
        if (cur == null)    return;
        if (flag == 2)      rightList.add(0, cur.val);
        else if (flag <= 1 || (cur.left == null && cur.right == null)) {
            leftList.add(cur.val);
        }
        
        preorder(cur.left, leftList, rightList, leftChildFlag(cur, flag));
        preorder(cur.right, leftList, rightList, rightChildFlag(cur, flag));
    }
    
    private int leftChildFlag(TreeNode cur, int flag) {
        if (flag <= 1)      return 1;
        // 现在找right boundary, 但right为空，只能让left当右边界
        else if (flag == 2 && cur.right == null)    return 2;  
        
        return 3;
    }
    
    private int rightChildFlag(TreeNode cur, int flag) {
        if (flag == 0 || flag == 2)      return 2;
        // 现在找left boundary, 但left为空，只能让right当左边界
        else if (flag == 1 && cur.left == null)    return 1;  
        
        return 3;
    }
    
    
    
    /** 156. Binary Tree Upside Down
     * Given a binary tree where all the right nodes are 
     * either leaf nodes with a sibling (a left node that shares the same parent node) 
     * or empty, flip it upside down and turn it into a tree where the original right nodes turned into left leaf nodes. 
     * Return the new root.
     *    Root                   L
	     /  \        --->      /  \
	    L    R                R   Root
     * @param root
     * @return
     */
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null || root.left == null)
            return root;
        
        TreeNode newRoot = upsideDownBinaryTree(root.left);
        root.left.left = root.right;
        root.left.right = root;

        root.left = null;		// 因为最后 翻转完 root变成leaf，所以需要left, right = null
        root.right = null;
        
        return newRoot;
    }
    
    
    public TreeNode upsideDownBinaryTree1(TreeNode root) {
        TreeNode cur = root;
        TreeNode next = null;
        TreeNode tmpRight = null;
        TreeNode pre = null;
        
        while (cur != null) {
            next = cur.left;
            
            // swap
            cur.left = tmpRight;
            tmpRight = cur.right;
            cur.right = pre;
            
            //move down
            pre = cur;
            cur = next;
        }
        return pre;
    }


	/**
	 * 958. Check Completeness of a Binary Tree
	 *
	 * 只要看如果某个node是null的话，后面需要都是null.. 否则就不是complete tree
	 *
	 * @param root
	 * @return
	 */
	public boolean isCompleteTree(TreeNode root) {
		Queue<TreeNode> q = new LinkedList<>();
		q.add(root);

		boolean nullChild = false;

		while (!q.isEmpty()) {
			TreeNode node = q.poll();
			if (node == null) {
				nullChild = true;
			} else {
				if (nullChild) {
					return false;
				}
				q.offer(node.left);
				q.offer(node.right);
			}
		}
		return true;
	}


	/**
	 * 427. Construct Quad Tree - 建四叉树
	 * 把一个 N * N的boolean矩阵 建成 四叉树.. 主要就是一层层切成4块..
	 * 如果每块都是一样的true或false, 那就是leaf node不用继续分.. 否则里面不同的话还要继续recursive分四等份 直到里面的range都一样的value为止
	 *
	 * 有的parent node因为是true & false相混合，所以val 不确定，随便..
	 *
	 * @param grid
	 * @return
	 *
	 * 基本就是先 for循环整个矩阵，一旦发现不一样的value，那就开始建node.. dfsBuild(xx)四个node..  一样的话就直接return leaf
	 */
	public Node construct(int[][] grid) {
		return buildQuadTree(grid, 0, 0, grid.length);
	}

	private Node buildQuadTree(int[][] grid, int x, int y, int len) {
		if (len <= 0)
			return null;

		boolean isLeaf = true;

		for (int i = 0; i < len; i++) {
			for (int j = 0; j < len; j++) {
				if (grid[x + i][y + j] != grid[x][y]) {     // 一不相等，就要break来recursive建node
					isLeaf = false;
					break;
				}
			}
		}

		if (isLeaf)
			return new Node(grid[x][y] == 1, true, null, null, null, null);

		return new Node(false, false,
			buildQuadTree(grid, x, y, len / 2),                 // 记住顺序要对
			buildQuadTree(grid, x, y + len / 2, len / 2),
			buildQuadTree(grid, x + len / 2, y, len / 2),
			buildQuadTree(grid, x + len / 2, y + len / 2, len / 2));
	}


	class Node {
		public boolean val;
		public boolean isLeaf;
		public Node topLeft;
		public Node topRight;
		public Node bottomLeft;
		public Node bottomRight;

		public Node() {}

		public Node(boolean _val,boolean _isLeaf,Node _topLeft,Node _topRight,Node _bottomLeft,Node _bottomRight) {
			val = _val;
			isLeaf = _isLeaf;
			topLeft = _topLeft;
			topRight = _topRight;
			bottomLeft = _bottomLeft;
			bottomRight = _bottomRight;
		}
	}
    
    

	public static void main(String[] args) {
		TreeSolution sol = new TreeSolution();
		TreeNode n1 = new TreeNode(1);
		TreeNode n2 = new TreeNode(2);
		TreeNode n3 = new TreeNode(3);
		TreeNode n4 = new TreeNode(4);
		TreeNode n5 = new TreeNode(5);
		n1.left = n2;
		n1.right = n3;
		n2.left = n4;
		n2.right = n5;

	//	sol.findLeavesBFS(n1);

		
	}
}
