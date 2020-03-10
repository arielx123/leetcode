import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
import java.util.Random;
import java.util.Set;

public class Design {

    /** 359. Logger Rate Limiter
     * 如果某个string在last 10s内没被print，那就return TRUE
     * Returns true if the message should be printed in the given timestamp, otherwise returns false.
     If this method returns false, the message will not be printed.
     The timestamp is in seconds granularity.

     * timestamp - map.get(msg) >= 10或者没出现过的，都更新map且TRUE
     *
     * 法1：直接用个hashMap存 <message, timestamp>
     */
    public boolean loggerShouldPrintMessage(int timestamp, String message) {
        Map<String, Integer> map = new HashMap<>();

        if (map.containsKey(message) && timestamp - map.get(message) < 10) {
            return false;
        }
        map.put(message, timestamp);
        return true;
    }


    /** 法2 (慢)： 把超过10秒的都删掉 */
    public void Logger() {
        time = new int[10];
        sets = new HashSet[10];
        for(int i = 0; i < 10; i++) {
            sets[i] = new HashSet<String>();
        }
    }

    Set<String>[] sets;
    int[] time;


    public boolean shouldPrintMessage(int timestamp, String message) {
        int idx = timestamp % 10;
        if (time[idx] != timestamp) {
            time[idx] = timestamp;
            sets[idx].clear();
        }

        // 看看10秒内有没出现过message
        for(int i = 0; i < 10; i++) {
            if (timestamp - time[i] < 10 && sets[i].contains(message)) {
                return false;
            }
        }

        sets[idx].add(message);

        return true;
    }



    /** 362. Design Hit Counter
     * 用queue表示.. get时超过300s 那就poll，最后q里还有多少就是总共的hits数
     *
     * 缺点：如果有很多重复时间的话，比如第3秒有很多次hit，那q的size会非常大
     * 空间会占用很多
     *
     * 那就看法11
     */
    public void HitCounter1() {
        q = new LinkedList<>();
        totalHits = 0;
    }

    LinkedList<Integer> q;

    public void hit1(int timestamp) {
        q.add(timestamp);
    }

    /** Return the number of hits in the past 5 minutes.
     @param timestamp - The current timestamp (in seconds granularity).
      * 返回过去 5min内 的hits总数
     */
    public int getHits1(int timestamp) {
        while (!q.isEmpty() && timestamp - q.peek() >= 300) {
            q.poll();
        }
        return q.size();
    }

    /**
     * improvement：新建一个类，存timestamp和count，那么q的size都是300
     *
     * 然后要用Deque<TimeWithCount>，因为也需要知道最后那个time跟当前time是否一样..
     *
     * 同时维护totalHits.. 每次超过5min要poll, 就 totalHits -= poll().count
     */
    int totalHits;
    Deque<TimeWithCount> dq;

    public void hit11(int timestamp) {
        if (!dq.isEmpty() && dq.getLast().timestamp == timestamp) {
            dq.getLast().count++;
        } else {
            dq.add(new TimeWithCount(timestamp, 1));
        }

        totalHits++;
    }

    /** Return the number of hits in the past 5 minutes.
     @param timestamp - The current timestamp (in seconds granularity). */
    public int getHits11(int timestamp) {
        while (!dq.isEmpty() && timestamp - dq.peek().timestamp >= 300) {
            totalHits -= dq.pollFirst().count;
        }
        return totalHits;
    }

    class TimeWithCount {
        int timestamp;
        int count;

        public TimeWithCount(int timestamp, int count) {
            this.timestamp = timestamp;
            this.count = count;
        }
    }



    /** 362. Design Hit Counter - 如果count次数很多的话
     * 用2个数组表示，比较容易写
     * counts主要表示 同一时间hit的次数
     */
    public void HitCounter2() {
        times = new int[300];           // 某个time出现的
        counts = new int[300];
    }

    int[] times;
    int[] counts;

    int lastPos = 0;

    /** Record a hit.
     * 将时间戳对300取余，然后看times此位置中之前保存的时间戳和当前的时间戳是否一样
     * 一样说明是同一个时间戳，那么对应的点击数counts[i]++
     * 否则，就超过5min，那么就reset成1
     * O(1) 复杂度
     */
    public void hit2(int timestamp) {

        int idx = timestamp % 300;
        if (times[idx] != timestamp) {     //已经过了5min
            times[idx] = timestamp;
            counts[idx] = 1;
        } else {
            counts[idx]++;              //相同time，那就count++
        }
    }

    /** 复杂度是O(n), 每次都要for 300次，只算5 min内的
     */
    public int getHits(int timestamp) {
        int result = 0;
        for (int i = 0; i < 300; i++) {
            if (timestamp - times[i] < 300) {       //time[i]在5min内
                result += counts[i];
            }
        }
        return result;
    }



    /** 只用一个数组counts..  还有一个lastPos算最后的timestamp。
     * 只要counts[i]非0，就证明这个时间出现过..
     * 而且counts存的都是 5min内的，所以每次都要更新太久之前的
     * O(n)的复杂度
     */
    public void hit3(int timestamp) {
        if (timestamp - lastPos > 300) {
            Arrays.fill(counts, 0);
        } else {
            // 如果之前有count，那么都是5min之前的，需要置0
            for (int i = lastPos + 1; i <= timestamp - 1; i++) {
                counts[i % 300] = 0;
            }
        }
        counts[--timestamp % 300]++;
        lastPos = timestamp;
    }

    /** 因为counts都是 5 min内的，所以for循环的次数比较少
     */
    public int getHits3(int timestamp) {
        if (timestamp - lastPos > 300)
            return 0;

        int result = 0;
        // 因为lastPos已经是最后的，之后没有更新，所以只用算lastPos以及5min内的之前那段就行
        int preN = 300 - (timestamp - lastPos) + 1;
        for (int i = 0; i < preN && lastPos - i >= 0; i++) {
            result += counts[(lastPos - i) % 300];
        }
        return result;
    }



    /** 535. Tiny Url
     * 随机生成6位 数字或字母
     * @param longUrl
     * @return
     */
    public String encodeURL(String longUrl) {
        if (longMap.containsKey(longUrl))
            return BASE_URL + longMap.get(longUrl);

        String shortKey = getRandomShortKey();
        while (shortMap.containsKey(shortKey)) {
            shortKey = getRandomShortKey();
        }
        shortMap.put(shortKey, longUrl);
        return BASE_URL + shortKey;
    }

    private String getRandomShortKey() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 6; i++) {
            sb.append(CHAR_SET.charAt(rand.nextInt(62)));
        }
        return sb.toString();
    }

    static String BASE_URL = "http://tinyurl.com/";
    static String CHAR_SET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    Map<String, String> shortMap = new HashMap<>();
    Map<String, String> longMap = new HashMap<>();
    Random rand = new Random();

    // Decodes a shortened URL to its original URL.
    public String decodeURL(String shortUrl) {
        return shortMap.get(shortUrl.replace(BASE_URL, ""));
    }


}
