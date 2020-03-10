import java.util.*;



/** LFU Cache - least frequently used
 * set时如果达到capacity，先去掉freq最小的。如果有多个freq最小的，那就去掉least recently.
 * @author Yan
 * 这题用 HashMap 和 double linked list, 和 linkedHashSet
 * 1. hashmap 来存 key/value, 和 key/freqNode
 * 2. 这个 FreqNode 里面有 代表这个次数的freq..是Final不会变。把出现了freq次的node，根据freqMap的freq次数，加到对应的list里
 *    prev, next 还有  LinkedHashSet<Integer> keys，来存 所有出现次数为freq的数(key)
 * 		LinkedHashSet 能有序地从最先 到 最近加入的顺序，并且能O(1)时间拿到对应的key.
 * 3. 所以用 hashmap 来存 key，和对应的 出现次数的node.
 * 4. 每次 set时，记得要increase freq, 并且把他更新到 head..  head指的是 最小freq的最早(least recently). 
 * 			所以要remove oldest时，去掉head. 并更新 2个map
 * 5. !!!!!!!!!注意，在new freqNode时 （有新的 次数），如果是插入到2个node中间，记得一定要把prev 和next设好!!!!!!!
 * 		FreqNode newNode = new FreqNode(cntNode.freq + 1, cntNode, cntNode.next, key);
		cntNode.next.prev = newNode;        // 记得把 后面next.prev连到newNode
		cntNode.next = newNode;
 *
 * 后面还有只用3个hashmap, 不用 double linkedlist 的做法，简单
 */
public class LFUCache {
    FreqNode head = null;		// head是最小freq的node. 从小往大排，要remove的话也是先remove它
	int capacity;
	Map<Integer, Integer> valueMap;		//store normal node's key & value, not FreqNode
	Map<Integer, FreqNode> freqMap;		// value is the FreqNode
	
	public LFUCache (int capacity) {
		this.capacity = capacity;
		valueMap = new HashMap<Integer, Integer>();
		freqMap = new HashMap<Integer, FreqNode>();
	}
	
	public int get(int key) {
		if (!valueMap.containsKey(key)) {
			return -1;
		}
		increaseFreq(key);
		return valueMap.get(key);
	}
	
	
	private void increaseFreq(int key) {
		FreqNode cntNode = freqMap.get(key);
		cntNode.keys.remove(key);		// 旧的freq要从keys里删掉，因为freq已经要+1了
		
		if (cntNode.next == null) {		// currently no freq that is freq+1
			cntNode.next = new FreqNode(cntNode.freq + 1, cntNode, null, key);
		} else if (cntNode.next.freq == cntNode.freq + 1) {
			cntNode.next.keys.add(key);
		} else {							// next node might be freq + 3, 要在 2个node 中间插入新的node
			FreqNode newNode = new FreqNode(cntNode.freq + 1, cntNode, cntNode.next, key);
			cntNode.next.prev = newNode;        // 记得把 后面next.prev连到newNode
			cntNode.next = newNode;
		}
		
		freqMap.put(key, cntNode.next);		//add this freq+1 node
		
		if (cntNode.keys.isEmpty()) {
			removeFreqNode(cntNode);
		}
	}
	

	public void set(int key, int value) {
		if (capacity == 0)	return;
		if (valueMap.containsKey(key)) {
		    valueMap.put(key, value);
		    increaseFreq(key);				//记得update freq
		} else {
			if (valueMap.size() == capacity) {
				removeOldest();
			}
			valueMap.put(key, value);
			addToHead(key);
		}
		
	}


	private void removeFreqNode(FreqNode node) {
		if (head == node) {			// this is the 1st freq node (smallest , 1)
			head = node.next;
		} else {
			node.prev.next = node.next;
		}

		if (node.next != null) {		//记得后面的也要连前面
			node.next.prev = node.prev;
		}
	}
    
	private void removeOldest() {
		if (head == null)	return;		//no element at all
		int oldestKey = head.keys.iterator().next();		// 最早放进去的值。因为linkedHashset的顺序就是从老到新
		head.keys.remove(oldestKey);
		if (head.keys.isEmpty()) {
			removeFreqNode(head);
		}
		valueMap.remove(oldestKey);
		freqMap.remove(oldestKey);
	}
	
	private void addToHead(int key) {
		if (head == null) {
			head = new FreqNode(1, null, null, key);
		} else if (head.freq == 1) {
			head.keys.add(key);
		} else {		// head freq >= 2
			FreqNode n = new FreqNode(1, null, head, key);
			head.prev = n;      //记得 head.prev = n
			head = n;
		}
		freqMap.put(key, head);		//update head
	}
	
	class FreqNode {
		public FreqNode prev, next;
		public final int freq;		//freq不变。
		public LinkedHashSet<Integer> keys = new LinkedHashSet<>();		// list that has same frequency. store keys.
		
		public FreqNode(int freq, FreqNode prev, FreqNode next, int key) {
			this.prev = prev;
			this.next = next;
			this.freq = freq;
			keys.add(key);
		}
	}


	/**
	 * 460. LFU Cache - least frequently used
	 * 这里不用FreqNode, 而是用Map<Integer, LinkedHashSet<Integer>> countToNodes来代替
	 * 而且用min 来track，方便remove
	 * @param capacity
	 */
	public void LFUCache2(int capacity) {
		min = -1;
		this.capacity = capacity;
//		keyToVal = new HashMap<>();
//		keyToCount = new HashMap<>();
//		countToNodes = new HashMap<>();
		countToNodes.put(1, new LinkedHashSet<>());     // 记得要put 1先
	}

	private int min = -1;
	private final Map<Integer, Integer> keyToVal= new HashMap<>();
	private final Map<Integer, Integer> keyToCount = new HashMap<>();
	private final Map<Integer, LinkedHashSet<Integer>> countToNodes = new HashMap<>();

	public int get2(int key) {
		if (!keyToVal.containsKey(key)) {
			return -1;
		}
		int count = keyToCount.get(key);

		// add count
		if (!countToNodes.containsKey(count + 1)) {
			countToNodes.put(count + 1, new LinkedHashSet<>());
		}
		countToNodes.get(count + 1).add(key);
		keyToCount.put(key, count + 1);

		// remove old count
		countToNodes.get(count).remove(key);
		if (count == min && countToNodes.get(count).size() == 0) {
			min++;          // 现在的count没了，最小的freq min变成后面的 min+1
		}

		return keyToVal.get(key);
	}

	public void put2(int key, int value) {
		if (capacity <= 0)
			return;

		if (keyToVal.containsKey(key)) {
			get(key);           // call get(key)这样 +- count
			keyToVal.put(key, value);
			return;
		}

		if (keyToVal.size() == capacity) {
			int oldestKey = countToNodes.get(min).iterator().next();
			keyToVal.remove(oldestKey);
			keyToCount.remove(oldestKey);
			countToNodes.get(min).remove(oldestKey);
		}

		min = 1;
		keyToVal.put(key, value);
		keyToCount.put(key, 1);
		countToNodes.get(1).add(key);
	}


	/**
	 * 432. All O`one Data Structure
	 * 所有操作都是 O(1), 增(没有时create), 减(1时remove), getMax(), getMin()
	 * 没有删除..
	 *
	 * 这个跟LFU很像...
	 * 也是需要有个CountNode 里面有 Set<String> keys 之后增减freq时remove掉
	 *
	 * !!!! 注意要在 加完新node以后再删除old的!!!! 否则origNode.prev啥的都删掉没了
	 */
	public void AllOne() {
		smallHead = new CountNode(null, -1, null, null);
		tail = new CountNode(null, -1, smallHead, null);
		smallHead.next = tail;
		map = new HashMap<>();
		countMap = new HashMap<>();
	}

	CountNode smallHead;         // small
	CountNode tail;         // big
	Map<String, Integer> map;
	Map<Integer, CountNode> countMap;


	/** Inserts a new key <Key> with value 1. Or increments an existing key by 1. */
	public void inc(String key) {
		if (!map.containsKey(key)) {
			map.put(key, 1);
			if (!countMap.containsKey(1)) {
				CountNode newNode = new CountNode(key, 1, smallHead, smallHead.next);
				smallHead.next.prev = newNode;
				smallHead.next = newNode;
				countMap.put(1, newNode);
			} else {
				countMap.get(1).keys.add(key);
			}
		} else {
			int count = map.get(key);
			map.put(key, count + 1);

			// update countMap
			updateCountMap(count, count + 1, false, key);
		}
	}

	/** Decrements an existing key by 1. If Key's value is 1, remove it from the data structure. */
	public void dec(String key) {
		if (!map.containsKey(key)) {
			return;
		}

		int count = map.get(key);

		if (count == 1) {
			countMap.get(1).keys.remove(key);
			if (countMap.get(1).keys.isEmpty()) {
				removeFromListNode(countMap.get(1));
				countMap.remove(1);
			}
			map.remove(key);
		} else {
			map.put(key, count - 1);
			// update countMap
			updateCountMap(count, count - 1, true, key);
		}
	}

	/** Returns one of the keys with maximal value. */
	public String getMaxKey() {
		if (tail.prev == smallHead)
			return "";

		return tail.prev.keys.iterator().next();
	}

	/** Returns one of the keys with Minimal value. */
	public String getMinKey() {
		if (smallHead.next == tail)
			return "";

		return smallHead.next.keys.iterator().next();
	}

	private void updateCountMap(int orig, int target, boolean smaller, String key) {

		CountNode origNode = countMap.get(orig);

		// update next count
		CountNode targetNode = countMap.get(target);
		if (targetNode == null) {
			if (smaller) {
				targetNode = new CountNode(key, target, origNode.prev, origNode);
				origNode.prev.next = targetNode;
				origNode.prev = targetNode;
			}
			else {
				targetNode = new CountNode(key, target, origNode, origNode.next);
				origNode.next.prev = targetNode;
				origNode.next = targetNode;
			}

			countMap.put(target, targetNode);
		} else {
			targetNode.keys.add(key);
		}

		// remove from original  注意要在 加完新node以后再删除!!!! 否则origNode.prev啥的都删掉没了
		origNode.keys.remove(key);
		if (origNode.keys.isEmpty()) {
			removeFromListNode(origNode);
			countMap.remove(orig);
		}
	}

	private void removeFromListNode(CountNode node) {
		node.prev.next = node.next;
		node.next.prev = node.prev;
	}

	class CountNode {
		CountNode prev;
		CountNode next;
		Set<String> keys;
		int count;

		public CountNode(String key, int count, CountNode prev, CountNode next) {
			this.prev = prev;
			this.next = next;
			this.count = count;
			keys = new HashSet<>();
			keys.add(key);
		}
	}
	
	public static void main(String[] args) {
		LFUCache lfu = new LFUCache(2);
		lfu.set(1,1);
		lfu.set(2,2);
		lfu.get(1);
		lfu.set(3, 3);
		lfu.get(2);
		lfu.get(3);
		lfu.set(4,4);
		lfu.get(1);
		lfu.get(3);
		lfu.get(4);
	}
}
