import java.util.HashMap;

/** LRU Cache  - HashMap + double linked list
 * @author Yan
 * Design and implement a data structure for Least Recently Used (LRU) nextNum. It should support the following operations: get and set.
 * get(key) - Get the value (will always be positive) of the key if the key exists in the nextNum, otherwise return -1
 * set(key, value) - Set or insert the value if the key is not already present. 
 * When the nextNum reached its capacity, it should invalidate the least recently used item before inserting a new item.
 * 
 * 需要O(1)的时间，那就要HashMap..
 * 其他操作也是O(1)，那么要double Linkedlist 才能保持head 和 tail，最近的和least frequent
 * 
 * 注意在 set时，可能key相同，那么要更新val。并且要挪到head之前，要删掉original位置
 */
public class LRUCache {
    private int capacity;
    private HashMap<Integer, Node> map = new HashMap<Integer, Node>();		// <key, Node>
    private Node head = new Node(-1, -1);	//头尾都是dummy node
    private Node tail = new Node(-1, -1);
    
    public LRUCache(int capacity) {
        this.capacity = capacity;
        tail.prev = head;			//记得连起来
        head.next = tail;
    }
    
    //get the value & delete original place & update更新 this node in recently used(head.next)
    public int get(int key) {
        if (!map.containsKey(key)) {
            return -1;
        }
        Node curr = map.get(key);    

        // 记得删掉原先的位置!!!!
        curr.prev.next = curr.next;
        curr.next.prev = curr.prev;
        
        //move to head
        moveToHead(curr);
        
        return curr.val;
    }
    
	 // if key exists but value different (1,2) then insert(1,5)
	 // 且set要更新放到tail那
    public void set(int key, int value) {
    	// 记得如果存在，要更新...这里包括 删原先位置 & moveToHead.
        if (get(key) != -1) {		//这里要用get(k) 因为更新了head. 不能仅仅用map.contains
            map.get(key).val = value;
            return;
        }
        if (map.size() == capacity) {
            map.remove(tail.prev.key);          // first delete in hashmap 先删
            tail.prev = tail.prev.prev;         // delete last node
            tail.prev.next = tail;
        }
        
        Node newN = new Node(key, value);   // insert new
        moveToHead(newN);                   //update to head
        map.put(key, newN);
    }

    // 这个只是move node to head.. node从原先位置断开是在别的地方写..
    // 因为new Node时没有定义prev, next, 不能断开..  所以这里只是move to head而没有断开的逻辑
    private void moveToHead(Node curr) {
        curr.next = head.next;                  // ensure the sequence is right
        head.next.prev = curr;					
        head.next = curr;
        curr.prev = head;
    }
    
    private class Node {
        Node prev;
        Node next;
        int val;
        int key;
        
        public Node(int key, int value) {
            this.key = key;
            val = value;
            prev = null;
            next = null;
        }
    }
    
}