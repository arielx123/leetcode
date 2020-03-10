

import java.util.*;

public class MatrixSolution {
	
	/** Set Matrix Zeroes 
	 * m*n的矩阵，有0的话就设该行或该列为0
	 * @param matrix
	 * ！！！！注意！！
	 * 如果判断到0，就直接row/col 设0的话，后面继续扫元素就都是0，这样整个matrix都清空了
	 * 所以要先标记在1st行/列，最后再删
	 */
	public void setZeroes(int[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        boolean[] row = new boolean[rows];
        boolean[] col = new boolean[cols];
        
        // store row, col index with value 0
        // Boolean数组存放 哪个i,j 是0。灰常巧妙
        for (int i = 0; i < rows; i++) {  
            for (int j = 0; j < cols; j++) { 
                if (matrix[i][j] == 0) {
                    row[i] = true;
                    col[j] = true;
                }
            }
        }
        // set 0
        for (int i = 0; i < rows; i++) {  
            for (int j = 0; j < cols; j++) { 
                if (row[i] || col[j]) {			//某row[i]或col[j]是0就置零
                    matrix[i][j] = 0;
                }
            }
        }
    }

	/** 这个只用O(1) space.. 
	 * 如果m[i][j]=0, 那就设第一行，第一列为0.. 
	 * 但是这样区分不了究竟第一行或第一列有没0，所以需要一个 col0来看第一列（或者第一行，一样的）是否有0
	 */
	public void setZeroesBetter(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        
        boolean col0 = false;			//看第一列是否有0.。 所以下面的列j都是从1开始看，而非0
        
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0)  
                col0 = true;
            for (int j = 1; j < n; j++) {       //j从1开始，不是0
                if (matrix[i][j] == 0) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 1; j--) {      // j到1，不是0
                if (matrix[i][0] == 0 || matrix[0][j] == 0)
                    matrix[i][j] = 0;
            }
            if (col0)   matrix[i][0] = 0;
        }
    }
	
	
	
	/**
     * 54. Spiral Matrix
	 * Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.
	 * 螺旋输出, 洋葱一层层剥开
	 * @param matrix
	 * @return
	 */
	public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        if (matrix.length == 0)     return result;
        
        int rowStart = 0;
        int rowEnd = matrix.length - 1;
        int colStart = 0;
        int colEnd = matrix[0].length - 1;
        
        while (rowStart <= rowEnd && colStart <= colEnd) {
            for (int j = colStart; j <= colEnd; j++) {      //to right
                result.add(matrix[rowStart][j]);
            }
            rowStart++;
            
            for (int i = rowStart; i <= rowEnd; i++) {      // to down
                result.add(matrix[i][colEnd]);
            }
            colEnd--;
            
            if (rowStart <= rowEnd) {                   // 确保start <= end 
                for (int j = colEnd; j >= colStart; j--) {		//go left
                    result.add(matrix[rowEnd][j]);
                }
            }
            rowEnd--;
            
            if (colStart <= colEnd) {                       // check
                for (int i = rowEnd; i >= rowStart; i--) {		//go up
                    result.add(matrix[i][colStart]);
                }
            }
            colStart++;
        }
        
        return result;
    }
	
	
	// 跟上题差不多，输出结果
	public int[][] generateMatrix(int n) {
        int[][] m = new int[n][n];
        int num = 1;
        int row = 0;
        int col = 0;
        
        while (n > 0) {		// n就是remainRow之类的
            if (n == 1) {
                m[row][col] = num;
                break;
            }
            
            for (int i = 0; i < n - 1; i++) {
                m[row][col++] = num++;
            }
            for (int i = 0; i < n - 1; i++) {
                m[row++][col] = num++;
            }
            for (int i = 0; i < n - 1; i++) {
                m[row][col--] = num++;
            }
            for (int i = 0; i < n - 1; i++) {
                m[row--][col] = num++;
            }
            
            row++;          //记得 ++，才能到内层
            col++;
            n -= 2;         // like level, should -2. n跟上面的remainRow/remainCol一样的作用
        }
        return m;
    }
	
	
	/** Rotate Image
	 * 顺时针转90°
	 * @param matrix
	 */
	public void rotate(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return;
        }
        int len = matrix.length;
        
        for (int layer=0; layer < len / 2; layer++) {
            int first = layer;
            int last = len - 1 - first;
            for (int i = first; i < last; i++) {
                int offset = i - first;
                int temp;
                //save top
                temp = matrix[first][i];
                //left to top
                matrix[first][i] = matrix[last - offset][first];
                //bottom to left
                matrix[last - offset][first] = matrix[last][last - offset];
                //right to bottom
                matrix[last][last - offset] = matrix[i][last];
                //top to right
                matrix[i][last] = temp;
            }
        }
    // 2nd method    
        for (int i = 0; i < len / 2; i++) {
            for (int j = 0 ; j < (len + 1) / 2; j++) {		//记得是(len+1)/2
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[len - j - 1][i];
                matrix[len - j - 1][i] = matrix[len - i - 1][len - j - 1];
                matrix[len - i - 1][len - j - 1] = matrix[j][len - i - 1];
                matrix[j][len - i - 1] = tmp;
            }
        }
    }
	
	
	/** 74. Search a 2D Matrix
	 * Integers in each row are sorted from left to right.
		The first integer of each row is greater than the last integer of the previous row.
	 	[
		  [1,   3,  5,  7],
		  [10, 11, 16, 20],
		  [23, 30, 34, 50]
		]
	 * @param matrix
	 * @param target
	 * @return
	 */
	public boolean searchMatrix(int[][] matrix, int target) {
        int cols = matrix[0].length;
        int rows = matrix.length;
        
        int s = 0, e = cols * rows - 1;
        while (s <= e) {
            int mid = s + (e - s) / 2;
            int ele = matrix[mid / cols][mid % cols];
            if (ele == target) {
                return true;
            } else if (ele < target) {
                s = mid + 1;
            } else {
                e = e - 1;
            }
        }
        
        return false;
    }
	
	
	/** 240. Search a 2D Matrix II
	 * Integers in each row are sorted in ascending from left to right.
		Integers in each column are sorted in ascending from top to bottom.
		[
		  [1,   4,  7, 11, 15],
		  [2,   5,  8, 12, 19],
		  [3,   6,  9, 16, 22],
		  [10, 13, 14, 17, 24],
		  [18, 21, 23, 26, 30]
		]
	 * @param matrix
	 * @param target
	 * @return
     * 从矩阵第一行最右侧开始查找，当前值比target大往左走，比target小的话往下走。
	 */
	public boolean searchMatrixII(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
           return false;
       int rows = matrix.length;
       int cols = matrix[0].length;
       
       // look from right-top corner, a 90 angle 右上角的直角 --> sorted list
       int i = 0, j = cols - 1;
       while (i < rows && j >= 0) {
           if (matrix[i][j] == target) {
               return true;
           } else if (matrix[i][j] > target) {
               j--;
           } else {
               i++;
           }
       }
       
       return false;
   }
	
	
	/** 378. Kth Smallest Element in a Sorted Matrix
	 * [ 1,  5,  9],
	   [10, 11, 13],
	   [12, 13, 15]   行列分别sorted，但是 不一定是整个蛇形有序的
	 * 跟373. Find K Pairs with Smallest Sums很类似
     *
     * @param matrix
     * @param k
     * @return
     * 把第一行放heap里，之后x+1就行。记得是min heap
     * 这样的好处是：不需要另外有个set<Pair>来记录已经用过的值.. 因为都是往下x+1 找，不会重复用
     */
    public int kthSmallestShorter(int[][] matrix, int k) {
    	Queue<NumType> heap = new PriorityQueue<NumType>();
    	int m = matrix.length;
        int n = matrix[0].length;
        // add 1st row in heap
        for (int j = 0; j < n; j++) {
            heap.offer(new NumType(0, j, matrix[0][j]));
        }
        
        for (int i = 0; i < k - 1; i++) {		// k - 1
        	NumType num = heap.poll();
            if (num.x == m - 1)  continue;
            heap.add(new NumType(num.x + 1, num.y, matrix[num.x + 1][num.y]));
            
        }
        return heap.peek().val;
    }
    
    // 把number的type和Comparable合起来。
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
	
	
	/** 378. Kth Smallest Element in a Sorted Matrix  - 忽略
	 * @param matrix
	 * @param k
	 * @return
	 * 每次加[x+1, y], [x, y+1]到heap里，之后弹出最小的
	 * 用minHeap最多存K个。add和Poll分别logK, 所以总的复杂度O(klogk)。
	 * 最坏情况是k是m*n(这题里假定是正方形), 那worst就是O(n^2 logn^2)
     *
     * 这个需要考虑 used情况  麻烦。 也占更多空间
	 */
	public int kthSmallest(int[][] matrix, int k) {
        if (matrix == null || matrix.length == 0) {
            return -1;
        }
        if (matrix.length * matrix[0].length < k) {
            return -1;
        }
        
        Queue<Number> heap = new PriorityQueue<Number>(k, new NumComparator());
        heap.add(new Number(0, 0, matrix[0][0]));
        int m = matrix.length, n = matrix[0].length;
        boolean[][] added = new boolean[m][n];  // 需要用added来存，防止之前heap里有过的数又再加进去，这样就错le 
        added[0][0] = true;       //use add to avoid add duplicate
        
        int[] nextX = {1, 0};  // next x's index
        int[] nextY = {0, 1};
        
        // pop at last, so < k-1
        for (int i = 0; i < k - 1; i++) {
            Number min = heap.poll();
            for (int j = 0; j < 2; j++) {   //only consider 2 elements
                int nx = min.x + nextX[j];  // j = 0, (x+1, y)
                int ny = min.y + nextY[j];  // j = 1, (x, y+1)
                if (nx < m && ny < n && !added[nx][ny]) {
                    added[nx][ny] = true;
                    heap.add(new Number(nx, ny, matrix[nx][ny]));
                }
            }
        }
        return heap.peek().val;
    }
    
    class Number {
        int x, y, val;
        public Number (int x, int y, int val) {
            this.x = x;
            this.y = y;
            this.val = val;
        }
    }
    
    class NumComparator implements Comparator<Number> {
        public int compare(Number n1, Number n2) {
            return n1.val - n2.val;
        }
    }
    
    
    /** 378. Kth Smallest Element in a Sorted Matrix
     * 用二分法，找第K个元素<=mid
     * @param matrix
     * @param k
     * @return
     * 因为每行每列都有序，那么for(每行), 看有多少count个数是 <= mid的.  那么就while(j列的matrix[i][j] > mid) j--来找
     * 算完所有行所有列以后，这次二分以mid来找的 count就是 前count个smallest数
     * 然后count 和 k 比较，如果不到所需的k个smallest，那就low=mid+1 （这样mid变大），增加查找的范围
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

    
    
    
    /** Submatrix Sum to 0
     * 找出和为0 的submatrix
     * @param matrix
     * @return
     * 刚开始 先预处理，求出 左上角到i,j的 sum
     * 中间跟subarray sum一样。
     * 
     * 因为submatrix形状大小不固定，所以要3层循环..
     * 找出top和bottom，for两次row，然后再找right，需要for cols. 
     */
    public int[][] submatrixSum(int[][] matrix) {
        int[][] result = new int[2][2];
        int m = matrix.length;
        int n = matrix[0].length;
        
        int[][] sum = new int[m+1][n+1];
        
        // 预处理  求以 左上角0,0开始，到i,j结束的 submatrix sum
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sum[i+1][j+1] = matrix[i][j] + sum[i+1][j] + sum[i][j+1] - sum[i][j];
            }
        }
        
        for (int lx = 0; lx < m; lx++) {
            for (int rx = lx + 1; rx <=m; rx++) {
                Map<Integer, Integer> map = new HashMap<Integer, Integer>();
                for (int j = 0; j <= n; j++) {			//不一定是正方形，有可能是矩形，所以要循环j<=n的列数来算
                    int diff = sum[rx][j] - sum[lx][j];		// lx+1 ~ rx的和
                    if (map.containsKey(diff)) {
                        int k = map.get(diff);
                        result[0][0] = lx;
                        result[0][1] = k;
                        result[1][0] = rx - 1;
                        result[1][1] = j - 1;
                        return result;
                    } else {
                        map.put(diff, j);
                    }
                }
            }
        }
        return result;
    }
    
    
    /** Sliding Window Matrix Maximum
     * 求出边长为k的sliding window matrix的最大和
     * @param matrix
     * @param k
     * @return
     */
    public int maxSlidingWindowMatrix(int[][] matrix, int k) {
    	int m = matrix.length;
    	int n = matrix[0].length;
    	int sum[][] = new int[m + 1][n + 1];
    	    	
    	// 求sum数组
    	for (int i = 1; i <= m; i++) {
    		for (int j = 1; j <= n; j++) {
    			sum[i][j] = matrix[i - 1][j - 1] + sum[i][j - 1] + sum[i - 1][j] - sum[i - 1][j - 1];
    		}		// 其实是当前matrix[i][j]值，只是因为长度多了1所以要 - 1
    	}
    
    	int max = Integer.MIN_VALUE;
    	for (int i = k; i <= m; i++) {
    		for (int j = k; j <= n; j++) {
    			int val = sum[i][j] - sum[i - k][j] - sum[i][j - k] + sum[i - k][j - k];
    			max = Math.max(max, val);
    		}
    	}
    	
    	return max;
    }
    
}
