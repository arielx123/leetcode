import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeSet;

public class Google {


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


    class WorkerBikeDistance implements Comparable<Solution.WorkerBikeDistance> {
        int worker;
        int bike;
        int distance;

        public WorkerBikeDistance(int worker, int bike, int distance) {
            this.worker = worker;
            this.bike = bike;
            this.distance = distance;
        }

        @Override
        public int compareTo(Solution.WorkerBikeDistance other) {
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
     * 1066. Campus Bikes II
     * 如果保证 所有 sum距离最小
     */
    public int assignBikesII(int[][] workers, int[][] bikes) {
        dfs(workers, 0, bikes, new boolean[bikes.length], 0);
        return min;
    }

    int min = Integer.MAX_VALUE;

    private void dfs(int[][] workers, int worker, int[][] bikes, boolean[] used, int sum) {
        if (worker == workers.length) {
            min = Math.min(min, sum);
            return;
        }

        if (sum > min)
            return;  // early termination

        for (int bike = 0; bike < bikes.length; bike++) {
            if (used[bike])
                continue;

            used[bike] = true;
            dfs(workers, worker + 1, bikes, used, sum + getDistance(workers[worker], bikes[bike]));
            used[bike] = false;
        }
    }

    private int getDistance(int[] worker, int[] bike) {
        return Math.abs(worker[0] - bike[0]) + Math.abs(worker[1] - bike[1]);
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

    // 简洁，指向grandfather，循环次数少
    public int find(int[] roots, int id) {
        while(id != roots[id]) {
            roots[id] = roots[roots[id]];  // 根roots[id]指向grandfather
            id = roots[id];
        }
        return id;
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
                }
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
     * 1087. Brace Expansion
     * "{a1,b}c7{d,e}f" 变成  ["a1c7df","a1c7ef","bc7df","bc7ef"]
     */
    public List<String> expand(String S) {
        // 按照{}split "{a1,b}c7{d,e}f" -> a1,b / c7 / d,e / f
        String[] temp = S.split("[{}]");
        List<List<String>> parts = new ArrayList<>();

        for (String s : temp) {
            if (!s.contains(",")) {
                parts.add(Arrays.asList(s));
            } else {
                parts.add(Arrays.asList(s.split(",")));
            }
        }

        List<String> result = new ArrayList<>();

        generatePermutations(parts, result, 0, "");

        return result;
    }

    private void generatePermutations(List<List<String>> parts, List<String> result, int pos, String str) {
        if (pos == parts.size()) {
            result.add(str);
            return;
        }

        for (String s : parts.get(pos)) {
            generatePermutations(parts, result, pos + 1, str + s);
        }
    }


    // 法2
    public List<String> expand2(String S) {

        List<String> res = new ArrayList<>();

        dfsHelper(res, S, 0, "");

        return res;
    }

    private void dfsHelper(List<String> result, String input, int pos, String str) {
        if (pos == input.length()) {
            result.add(str);
            return;
        }

        int left = input.indexOf('{', pos);
        int right = input.indexOf('}', pos);

        // 没有 {} 正常的string，直接加进result返回
        if (left < 0) {
            result.add(str + input.substring(pos));
            return;
        }

        String beforeBracket = input.substring(pos, left);
        String sub = input.substring(left + 1, right);
        String[] parts = sub.split(",");

        for (String part : parts) {
            dfsHelper(result, input, right + 1, str + beforeBracket + part);
        }
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


    /**
     * 947. Most Stones Removed with Same Row or Column
     * 给一堆stone的坐标.. 只有某个stone在同一行或列有别的stone时才能remove..
     * 问最多能remove多少个stone
     *
     * 其实 stone with same row/col的话，那就是联通component..
     * 尽量remove后保持更多的connected component，这样能让后面的remove掉..
     * 那么这个问题其实是 stone总个数 - connectedComponents..
     *
     * return stones.length - components;
     *
     * 那么就可以 DFS 找islands了
     *
     * 后面有union find的解法
     */
    public int removeStones(int[][] stones) {
        Set<int[]> visited = new HashSet();
        int components = 0;

        for (int[] stone : stones) {
            if (!visited.contains(stone)) {
                dfsCheckConnect(stones, visited, stone);
                components++;
            }
        }

        return stones.length - components;
    }

    private void dfsCheckConnect(int[][] stones, Set<int[]> visited, int[] stone){
        visited.add(stone);

        for (int[] next: stones){
            if (!visited.contains(next)){
                // stone with same row or column. group them into island
                if (next[0] == stone[0] || next[1] == stone[1])
                    dfsCheckConnect(stones, visited, next);
            }
        }
    }

    // 用union find
    public int removeStonesUF(int[][] stones) {
        if (stones == null || stones.length <= 1);
        int n = stones.length;

        int[] roots = new int[n];
        for (int i = 0; i < n; i++) {
            roots[i] = i;
        }

        int components = n;

        // 石头两两比较
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (stones[i][0] != stones[j][0] && stones[i][1] != stones[j][1])
                    continue;
                // 在同一行 或 列
                int root1 = find(roots, i);
                int root2 = find(roots, j);
                if (root1 != root2) {
                    roots[root1] = root2;
                    components--;
                }
            }
        }
        return n - components;
    }


    /**
     * Minimize the distance to the farthest point  最方便寓所
     * 找个apartment, 能离requires地点的max距离最小
     */
    public int findBestLocation(List<Set<String>> blocks, List<String> requires) {
        // key is facility, val is list of block ids, facility所在的block IDs
        Map<String, List<Integer>> facilityToBlocks = createMap(blocks);

        int result = 0;
        int minResult = Integer.MAX_VALUE;

        for (int i = 0; i < blocks.size(); i++) {
            int maxDist = 0;
            for (String facility : requires) {
                int minDist = getMinDist(facilityToBlocks, facility, i);   // 找离这个facility最近的block
                maxDist = Math.max(maxDist, minDist);           // 在最近的里面挑最远的
            }

            if (minResult > maxDist) {
                minResult = maxDist;
                result = i;             // 改成这个block
            }
        }
        return result;
    }

    // 每个facility, 对应 出现的 block ids
    private Map<String, List<Integer>> createMap(List<Set<String>> blocks) {
        Map<String, List<Integer>> map = new HashMap<>();

        for (int i = 0; i < blocks.size(); i++) {
            for (String facility : blocks.get(i)) {
                map.putIfAbsent(facility, new ArrayList<>());
                map.get(facility).add(i);
            }
        }
        return map;
    }

    // 找离这个facility最近的block
    private int getMinDist(Map<String, List<Integer>> map, String facility, int pos) {
        int min = Integer.MAX_VALUE;

        for (int block : map.get(facility)) {
            min = Math.min(min, Math.abs(pos - block));
        }
        return min;
    }


    /**
     * 727. Minimum Window Subsequence
     * subsequence的顺序要一样.. 不像 minimum window substring 可以不同顺序
     * S = "abcdebdde", T = "bde"
     * Output: "bcde"
     *
     * dp[i][j] 表示范围S中前i个字符包含范围T中前j个字符的子串的起始位置 ** start idx **
     *
     * 相等时 dp[i][j] = dp[i - 1][j - 1];
     * 不同时 dp[i][j] = dp[i - 1][j];
     *
     * 最后扫完T 后，看有没match上的，算min window
     *
     * 后面有更快的two pointer
     */
    public String minWindowSubsequence(String S, String T) {
        int sLen = S.length();
        int tLen = T.length();
        int startPos = -1;
        int minLen = Integer.MAX_VALUE;
        int[][] dp = new int[sLen + 1][tLen + 1];       // dp[i][j] 表示范围S中前i个字符包含范围T中前j个字符的子串的起始位置start idx

        // 初始化 最开始都是 -1
        for (int[] arr : dp) {
            Arrays.fill(arr, -1);
        }
        // 初始化 i的第一列起始是 i
        for (int i = 0; i <= sLen; i++) {
            dp[i][0] = i;
        }

        // 开始算
        for (int i = 1; i <= sLen; i++) {
            for (int j = 1; j <= tLen; j++) {
                if (S.charAt(i - 1) == T.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }

            if (dp[i][tLen] != -1) {         // match完T 了
                int len = i - dp[i][tLen];      // i - startIdx
                if (minLen > len) {
                    minLen = len;
                    startPos = dp[i][tLen];
                }
            }
        }
        return startPos == -1 ? "" : S.substring(startPos, startPos + minLen);
    }


    /**
     * 727. Minimum Window Subsequence - two pointer sliding window
     *
     * 一旦match了，i, j开始往前缩短window，看是否能找到最小的minLen
     *
     * 之后因为 i++ 会继续往前，这时j=0, 但新的i可能就不match了，所以OK，继续往前走
     */
    public String minWindow(String S, String T) {
        int sLen = S.length();
        int tLen = T.length();
        int start = -1;
        int minLen = Integer.MAX_VALUE;

        for (int i = 0, j = 0; i < sLen; i++) {
            if (S.charAt(i) == T.charAt(j)) {
                j++;
            }

            // match了
            if (j == tLen) {
                int end = i + 1;

                // 开始往前缩短window
                while (--j >= 0) {
                    while (S.charAt(i) != T.charAt(j)) {    // 不同就i--
                        i--;
                    }
                    i--;        // 正常是 i & j 都 -- 往前
                }

                i++;        // 减过头了加回来
                j++;

                if (end - i < minLen) {
                    minLen = end - i;
                    start = i;
                }
            }
        }
        return start == -1 ? "" : S.substring(start, start + minLen);
    }


    /**
     * Random generate maze 随机生成迷宫
     */


    /**
     * 222. Count Complete Tree Nodes
     * count个数
     */
    public int countNodes(TreeNode root) {
        int leftHeight = leftHeight(root);
        int rightHeight = rightHeight(root);

        if (leftHeight == rightHeight) {        // full 有所有node
            return (1 << leftHeight) - 1;       // 2 ^ height - 1
        }

        return 1 + countNodes(root.left) + countNodes(root.right);
    }

    private int leftHeight(TreeNode node) {
        int height = 0;
        while (node != null) {
            node = node.left;
            height++;
        }
        return height;
    }

    private int rightHeight(TreeNode node) {
        int height = 0;
        while (node != null) {
            node = node.right;
            height++;
        }
        return height;
    }


    /**
     * 1024. Video Stitching
     * Return the minimum number of clips needed so that we can cut the clips into segments
     * that cover the entire sporting event ([0, T]).
     * If the task is impossible, return -1.
     *
     * 先sort下start时间.. 之后尽量找end 大的
     */
    public int videoStitching(int[][] clips, int T) {
        int count = 0;
        Arrays.sort(clips, (a,b) ->  a[0] - b[0]);      // sort by start time

        int end = 0, newEnd = 0;

        for (int i = 0; i < clips.length;) {
            if (clips[i][0] > end)     // 需要有重叠
                return -1;

            while (i < clips.length && clips[i][0] <= end) {
                newEnd = Math.max(newEnd, clips[i][1]);
                i++;
            }
            count++;
            end = newEnd;

            if (end >= T)           // 找到
                return count;
        }
        return -1;
    }

    // DP 方法
    public int videoStitchingDP(int[][] clips, int T) {
        int[] dp = new int[T + 1];      // 在范围 i 内最少需要几个clips
        Arrays.fill(dp, T + 1);
        dp[0] = 0;

        for (int i = 0; i <= T; i++) {
            for (int[] c : clips) {
                if (c[0] <= i && i <= c[1]) {         // 这个i 被 clip包含了.. 能cover
                    dp[i] = Math.min(dp[i], dp[c[0]] + 1);
                }
            }
            if (dp[i] == T + 1)
                return -1;
        }
        return dp[T];
    }


    /**
     * Skip Iterator
     * 实现一个iterator class, input是正常的iterator, 这个class可以实现hasNext(), next(), skip(element),
     * skip(element)会跳过正常iterator里next occurence of the given element。
     * 如果skip call n times, 就跳过下面 n个given element，iterator里的elements可以有重复。
     *
     * 跟341. Flatten Nested List Iterator 有点像
     */
    public class SkipIterator {

        Iterator<Integer> iterator;
        Map<Integer, Integer> counter = new HashMap();
        Integer nextNum;

        public SkipIterator(Iterator<Integer> iterator) {
            this.iterator = iterator;
        }

        boolean hasNext() {
            if (counter.containsKey(nextNum))        // 如果skip的是最后数字，需要把它变null才不会打印出来
                nextNum = null;

            while (iterator.hasNext()) {
                nextNum = iterator.next();
                if (counter.containsKey(nextNum)) {
                    counter.put(nextNum, counter.get(nextNum) - 1);
                    if (counter.get(nextNum) == 0) {
                        counter.remove(nextNum);
                    }
                } else {
                    break;
                }
            }
            return nextNum != null;
        }

        int next() {
            int ans = nextNum;
            nextNum = null;     // 记得设null，连续call几次haxNext()的话会跳过Integer
            return ans;
        }

        void skip(int num) {
            counter.put(num, counter.getOrDefault(num, 0) + 1);
        }
    }

}
