
import java.util.*;

class ListNode {		//去掉public
    int val;
    ListNode next;
    ListNode(int x) {
        val = x;
        next = null;
    }
}

public class MathSolution {
	
	/** 172. Factorial Trailing Zeroes
	 * 算出 n! 后面有多少个0
	 * @param n
	 * @return
	 * 要看有多少个5, 因为5 前面肯定有2..所以能变10.
	 * 记得要 调用剩下的
	 */
	public int trailingZeroes(int n) {
        if (n == 0)     return 0;
        return n / 5 + trailingZeroes(n / 5);
    }
	
	
	/** Integer to Roman
	 * Given an integer, convert it to a roman numeral.
	 * Input is guaranteed to be within the range from 1 to 3999.
	 * @param num
	 * @return
	 */
	public String intToRoman(int num) {
        String[] symbol = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        int[] nums = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        
        StringBuilder sb = new StringBuilder();
	    for (int i = 0; i < nums.length; i++) {
	        while (num >= nums[i]) {
	            num -= nums[i];
	            sb.append(symbol[i]);
	        }
	    }
	    return sb.toString();
    }
	
	
	public static String intToRoman1(int num) {
	    String M[] = {"", "M", "MM", "MMM"};
	    String C[] = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
	    String X[] = {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
	    String I[] = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
	    return M[num / 1000] + C[(num % 1000) / 100] + X[(num % 100) / 10] + I[num % 10];
	}
	
	
	public String intToRoman2(int num) {
        int digit = 0;                                      //从个位开始，然后再十位，百位
        char[][] map = {{'I', 'V', 'X'}, {'X', 'L', 'C'}, {'C', 'D', 'M'}, {'M', 'M', 'M'}};
        String result = "";
        while (num != 0) {									//位数除完了，num就为0
            int cur = num % 10;
            if (cur <= 3) {
                for (int i = 1; i <= cur; i++) {
                    result = map[digit][0] + result;            
                }
            } else if (cur == 4) {
            // //不能 re += xxx + xx 否则只是数字。记得先与res要加(). 但re+=xx 还是可以是字母
                result = map[digit][0] + (map[digit][1] + result);        // IV.
            } else if (cur == 5) {
                result = map[digit][1] + result;                      // V.
            } else if (cur <= 8) {
                for (int i = 1; i <= cur - 5; i++) {        
                    result = map[digit][0] + result;                    // II
                }
                result = map[digit][1] + result;                        // II+V
            } else {                                           			 // 9
                result = map[digit][0] + (map[digit][2] + result);       // IX.
            }
            
            num /= 10;										//得到位数，十位/千位
            digit ++;                                       // which digit
        }
        return result;
    }
	
	
	
	/**Roman to Integer   1~3999
	 * @param s
	 * @return
	 */
	public int romanToInt(String s) {
        if (s == null || s.length() == 0)
            return -1;
        int result = 0;
        //也可以用while. 这样 if{i -= 2}. else{i--}
        for (int i = s.length() - 1; i >= 0; i--) {     //从个位开始算
            if (i > 0 && trans(s.charAt(i)) > trans(s.charAt(i-1)) ) {
                result += trans(s.charAt(i)) - trans(s.charAt(i-1));
                i--;                            // IX=4, 这样要减2位。因为for有i--，so这里再减1就行
            } else {
                result += trans(s.charAt(i));
            }
        }
        return result;
    }
    private int trans(char c){
        switch(c){
            case 'I': return 1;
            case 'V': return 5;
            case 'X': return 10;
            case 'L': return 50;
            case 'C': return 100;
            case 'D': return 500;
            case 'M': return 1000;
        }
        return 0;
    }
    
    
    
    /** 273. Integer to English Words
     * 1234567 -> "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"
     */
    private final String[] belowTen = new String[] {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
    private final String[] belowTwenty = new String[] {"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
    private final String[] belowHundred = new String[] {"", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
        
        
    public String numberToWords(int num) {
        if (num == 0) return "Zero";
        
        return helper(num);
    }
    
    public String helper(int num) {
        String result = "";
        if (num < 10) result = belowTen[num];
        else if (num < 20) result = belowTwenty[num-10];
        else if (num < 100) result = belowHundred[num/10] + " " + helper(num%10); 
        else if (num < 1000) result = helper(num/100) + " Hundred " + helper(num%100); 
        else if (num < 1000000) result = helper(num/1000) + " Thousand " + helper(num%1000); 
        else if (num < 1000000000) result = helper(num/1000000) + " Million " + helper(num%1000000); 
        else  result = helper(num/1000000000) + " Billion " + helper(num%1000000000); 
        
        return result.trim();
    }
    
    
    /** 9. Palindrome Number
     * 回文。 676 TRUE   要算出base看是几位数，然后最高位和最低位比
     * 或者用下面的reverse integer来反转，再跟原来的比较
     * @param x
     * @return
     */
    public boolean isPalindrome(int x) {
        if (x < 0) return false;
        if (x == 0) return true;

        int base = 1, tmp = x;
        while (tmp >= 10) {
            tmp /= 10;
            base *= 10;             // 10^n. 676-- base 100
        }
        
        while (x != 0) {
            if (x % 10 != x / base)      // x / base, get the highest digit
               return false;            //  x % 10,   get the lowest digit
            
            x = (x % base) / 10;        // x % base -> remove highest digit
                                        // x / 10 -> remove lowest digit
            base /= 100;               //上面去掉最高&最低2个数，所以/100
        }

        long n = 10L;

        return true;
    }
    
    
    /** 7. Reverse Integer 
     * @param x
     * @return
     * x % 10 得到最低位，循环这些低位*10，这样就能变成高位. 
     * 再x /= 10 去掉 最低位 得到 次低位
     * 无需判断正负，因为都是-.
     */
    public int reverse(int x) {
        int result = 0;
        while (x != 0) {						// result * 10 把之前低位 *10 这样成新的高位
            int tmp = result * 10 + x % 10;    // x % 10, get the lowest digit            
            
         // check if overflow 如果 overflow后，tmp跟之前没*10的result是不同的
            if (tmp / 10 != result) 	
            	return 0;
            
            result = tmp;
            x /= 10;                         // x / 10 -> remove lowest digit
        }
        return result;   //或者看是否 > integer.max || < int.min
    }
    
    
    /** 8. String to Integer (atoi) 
     * 考虑各种情况
     * for(i:0 ~ len-1) { num = num * 10 + (str.charAt(i) - '0');}
     * @param str
     * @return
     */
    public int atoi(String str) {
        if(str == null){  
            return 0;  
        }  
        str = str.trim();  
        if (str.length() == 0) {
            return 0;
        }
        int sign = 1;
        int i = 0;
        long num = 0;
        if (str.charAt(0) == '+') {
            i++;
        } else if (str.charAt(0) == '-') {
            sign = -1;
            i++;
        }
        
        for ( ; i < str.length(); i++) {
            if (!Character.isDigit(str.charAt(i))) {
                break;
            }
            num = num * 10 + (str.charAt(i) - '0');
            
            if (sign == 1 && num > Integer.MAX_VALUE)    return Integer.MAX_VALUE;
            if (sign == -1 && (-1) * num < Integer.MIN_VALUE)    return Integer.MIN_VALUE;
        }
        return (int) num * sign;
        
        // 如果result是int的话
        /*
        if(ch <'0' || ch>'9') break;
        
        if(Integer.MAX_VALUE/10 < ret || Integer.MAX_VALUE/10 == ret && Integer.MAX_VALUE%10 < (ch-'0') )
            return sign==-1 ? Integer.MIN_VALUE : Integer.MAX_VALUE;
        */
    }
    
    
    /** 66. Plus One 
     * 数字存在数组里。加1. 有overflow的话新建数组
     * 在for里面 加上 carry>0 . 这样可以提前break
     * @param digits
     * @return
     */
    public int[] plusOne(int[] digits) {
        int carry = 1;
        for (int i = digits.length - 1; i >= 0 && carry > 0; i--) {
            int sum = digits[i] + carry;		
            digits[i] = sum % 10;
            carry = sum / 10;
            if (carry == 0)
            	return digits;
        }
        
        if (carry == 0) {               // not overflow
            return digits;
        }
          // if overflow
        int[] result = new int[digits.length + 1];
        result[0] = 1;
        return result;
    }
    
    public int[] plusOneEasy(int[] digits) {
    	int n = digits.length;
    	for(int i=n-1; i>=0; i--) {
            if(digits[i] < 9) {
                digits[i]++;
                return digits;
            }   
            digits[i] = 0;
        }
    
        int[] newNumber = new int [n+1];
        newNumber[0] = 1;
        return newNumber;
    }
    
    
    
    /** 369. Plus One Linked List
     * 1->2->3 加1 变成 1->2->4  return head
     * @param head
     * @return
     * 找到 最后一个非9 的node. 然后n.val++, 后面全变0.
     * n在dummy，也cover了9999..+1 后有一个carry的情况
     */
    public ListNode plusOne(ListNode head) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode n = dummy;
        ListNode last = dummy;  //last non-9
        
        while (n != null) {
            if (n.val != 9) {
                last = n;
            }
            n = n.next;
        }
        
        last.val++;				//最后一个非9的 +1
        last = last.next;
        while (last != null) {		//把9及后面的都变0
            last.val = 0;
            last = last.next;
        }
        
        if (dummy.val == 1) {		//说明last at dummy在第一个while里没动过，overflow成1
            return dummy;
        }
        
        return dummy.next;   //正常情况
    }
    
    
    // 复杂。 先reverse，再+1，再reverse
    public ListNode plusOneWorse(ListNode head) {
        ListNode newHead = reverse(head);
        ListNode tmp = newHead;
        int carry = 1;
        while (tmp != null) {
            tmp.val += carry;
            if (tmp.val < 10) {
                carry = 0;
                break;
            }
            tmp.val %= 10;
            tmp = tmp.next;
        }
        
        head = reverse(newHead);
        if (carry == 1) {
            newHead = new ListNode(1);
            newHead.next = head;
            return newHead;
        }
        return head;
    }
    
    public ListNode reverse(ListNode head) {
        ListNode pre = null;
        ListNode next;
        while (head != null) {
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }
    
    
    /** Add Binary
     * a = "11"  b = "1"  Return "100".
     * @param a
     * @param b
     * @return
     */
    public String addBinary(String a, String b) {
        StringBuilder sb = new StringBuilder();
        int ai = a.length() - 1;
        int bi = b.length() - 1;
        int carry = 0;
        while (ai >= 0 || bi >= 0) {
            int aa = ai < 0 ? 0 : a.charAt(ai) - '0';
            int bb = bi < 0 ? 0 : b.charAt(bi) -'0';
            int sum = aa + bb + carry;
            carry = sum / 2;
            sb.insert(0, sum % 2);
            ai--;	//不用if > 0 否则死循环
            bi--;	//如果a,b长度一样，那最后是ai=0 && bi=0，这样就跳出循环
        }
        if (carry == 1)
            sb.insert(0, 1);			//也可以直接append, 最后reverse sb就行
        return sb.toString();	
    }
    
    
    
    /** 415. Add Strings
     * 给两个数，是string。相加
     * @param num1
     * @param num2
     * @return
     */
    public String addStrings(String num1, String num2) {
        int carry = 0;
        int i = num1.length() - 1, j = num2.length() - 1;
        StringBuilder sb = new StringBuilder();
        for (; i >= 0 || j >= 0 || carry == 1; i--, j--) {
            int x = i >= 0 ? num1.charAt(i) - '0' : 0;
            int y = j >= 0 ? num2.charAt(j) - '0' : 0;
            sb.append((x + y + carry) % 10);
            carry = (x + y + carry) / 10;
        }
        
        return sb.reverse().toString();
    }
    
    
    
    /** Add Float String
     * 如果两个数是float的话
     * 那就把开始的j加成跟i一样的小数点数（长点）。直到j < len2就可以获值
     * j = num1.length() - 1 - point1 + point2
     * @param num1
     * @param num2
     * @return
     */
    public String addFloatStrings(String num1, String num2) {
        int carry = 0;
        int point1 = num1.indexOf('.');
        int point2 = num2.indexOf('.');
        int offset = (num1.length() - 1 - point1) - (num2.length() - 1 - point2);
        if (offset < 0)
        	return addFloatStrings(num2, num1);
        
        int i = num1.length() - 1;
// 让j变成后面的小数位跟num1一样.. 所以是n1的小数位len1-1-point1, 加上自己的小数位p2
        int j = num1.length() - 1 - point1 + point2;	
        
        StringBuilder sb = new StringBuilder();
        for (; i >= 0 || (j >= 0 && j < num2.length()) || carry == 1; i--, j--) {
        	if (i == point1 && j == point2) {
        		sb.append('.');
        	} else {
        		int x = i >= 0 ? num1.charAt(i) - '0' : 0;
                int y = j >= 0 && j < num2.length() ? num2.charAt(j) - '0' : 0;
                sb.append((x + y + carry) % 10);
                carry = (x + y + carry) / 10;
        	}
        }
        
        return sb.reverse().toString();
    }
    
    
    /** 2. Add Two Numbers 
     * Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
	   Output: 7 -> 0 -> 8
     * @param n1
     * @param n2
     * @return
     * 注意while里是 l1 或 l2非空就行。 否则&&的话，之后还要再循环多次判断 很麻烦
     * 每次新建一个node来存sum. 而且用pre表明l1/l2前一个，这样当其中之一null或者最后有carry才能连起来
     */
    public ListNode addTwoNumbers(ListNode n1, ListNode n2) {
        ListNode dummy = new ListNode(-1);
        ListNode tmp = dummy;		// !!!!!!!! tmp=dummy后，往后走加的值也算是dummy加的
        int carry = 0;
        while (n1 != null || n2 != null || carry != 0) {
            int a = n1 == null ? 0 : n1.val;
            int b = n2 == null ? 0 : n2.val;
            int sum = a + b + carry;
            carry = sum / 10;
            tmp.next = new ListNode(sum % 10);
            tmp = tmp.next;
            if (n1 != null)		n1 = n1.next;
            if (n2 != null)		n2 = n2.next;
        }
        
        // if (carry == 1)				//可以直接在while条件中加 carry!=0
        //     tmp.next = new ListNode(1);
        return dummy.next;
    }
    
    
    
    /** 445. Add Two Numbers II
     * 是正常顺序，需要先算后面才能算前面
     * @param l1
     * @param l2
     * @return
     * 可以先各自l1, l2 reverse, 然后按照I的来算，最后再reverse
     * 
     * 这个用stack，这样就能从后往前。建node也是从后往前
     */
    public ListNode addTwoNumbersIIStack(ListNode l1, ListNode l2) {
        Stack<Integer> s1 = new Stack<Integer>();
        Stack<Integer> s2 = new Stack<Integer>();
        
        while(l1 != null) {
            s1.push(l1.val);
            l1 = l1.next;
        }
        while(l2 != null) {
            s2.push(l2.val);
            l2 = l2.next;
        }
        
        int sum = 0;
        ListNode dummy = new ListNode(0);       //dummy从后往前
        while (!s1.empty() || !s2.empty()) {
            if (!s1.empty()) sum += s1.pop();
            if (!s2.empty()) sum += s2.pop();
            dummy.val = sum % 10;
            ListNode node = new ListNode(sum / 10);
            node.next = dummy;          //先建前面的carry，然后再往前挪
            dummy = node;
            sum /= 10;
        }
        
        return dummy.val == 0 ? dummy.next : dummy;
    }
    
    
    /** 445. Add Two Numbers II
     * 是正常顺序，需要先算后面才能算前面
     * @param l1
     * @param l2
     * @return
     * 如果不能用reverse，那就跟Plus One差不多
     * 找到最后一个非9的，然后把last后面的每次都置0..
     * 复杂度挺高
     */
    public ListNode addTwoNumbersII(ListNode l1, ListNode l2) {
        int len1 = getLength(l1);
        int len2 = getLength(l2);
        if (len2 > len1) {          //swap, 保证l1 longer
            ListNode tmp = l1;
            l1 = l2;
            l2 = tmp;
        }
        
        int diff = Math.abs(len1 - len2);
        ListNode dummy = new ListNode(0);
        ListNode node = dummy;
        ListNode last = dummy;          // 最后一个非9的
        
        while (diff-- > 0) {
            node.next = new ListNode(l1.val);
            if (l1.val != 9) {
                last = node.next;
            }
            node = node.next;
            l1 = l1.next;
        }
        
        int carry = 0;
        
        // 正常的l1跟l2相加
        while (l1 != null) {
            int sum = l1.val + l2.val;
            if (sum >= 10) {
                sum -= 10;
                last.val++;
                last = last.next;
            
                // 把last后面的置0
                while (last != null) {
                    last.val = 0;
                    last = last.next;
                }
                last = node;
            }
            // 正常往后走
            node.next = new ListNode(sum);
            if (sum != 9) {
                last = node.next;
            }
            
            node = node.next;
            l1 = l1.next;
            l2 = l2.next;
        }
        
        if (dummy.val == 1)     return dummy;
        
        return dummy.next;
    }
    
    
    private int getLength(ListNode node){
        int len = 0;
        while (node != null){
            len++;
            node = node.next;
        }
        return len;
    }
    
    
    
    /** 43. Multiply Strings
     * @param num1
     * @param num2
     * @return
     * 相乘 需要两层for循环。
     * 开一个 pos[m+n]
     * 当前值为 p[i+j+1]后面, 前面的carry为[i+j]在前面
     * 需要 sum / 10 + 上一次的p[i+j]
     */
    public String multiply(String num1, String num2) {
        int m = num1.length(), n = num2.length();
        int[] pos = new int[m + n];
        
        for (int i = m - 1; i >= 0; i--) { 
            for (int j = n - 1; j >= 0; j--) {
                int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                int sum = mul + pos[i + j + 1];         //加上之前的
                pos[i + j] += sum / 10;      // carry, the one before n[i+j+1] 记得加上之前的p[i+j]值
                pos[i + j + 1] = sum % 10;
            }
        }
        
        StringBuilder sb = new StringBuilder();
        // 有可能是 0*0, 最后结果都是0
        for (int p : pos) {
            if (sb.length() == 0 && p == 0)     //开头为0的话，跳过
                continue;
            sb.append(p);
        }
        
        return sb.length() == 0 ? "0" : sb.toString();
    }
    
    
    
    /** 29. Divide Two Integers
     * @param dividend
     * @param divisor
     * @return
     * 比如 15 / 3, 基本思想是看15减3 能减几次..，直到为0为止。但是这样会太慢。
     * 那么可以除数3每次 *2 这么减，能加快速度，直到找到 <=15的最大数，就能得出所需要的multiple数
     * 
     * 需要a >= b, 因为从3开始，最后算到12时，余数为3，所以需要再while一次 ，multiple再+1
     */
    public int divide(int dividend, int divisor) {
    	if (divisor == 0 || (dividend == Integer.MIN_VALUE && divisor == -1)) return Integer.MAX_VALUE;
    	// 考虑负数情况
    	boolean negative = (dividend > 0 && divisor < 0) || (dividend < 0 && divisor > 0);
        
        long a = Math.abs((long) dividend);			//转long和绝对值，左移可能越界
        long b = Math.abs((long) divisor);
        int result = 0;
        
        while (a >= b) {
            long tmp = b;
            long multiple = 1;
            while (a >= (tmp << 1)) {       //找到小于a的最大b倍数
                tmp <<= 1;
                multiple <<= 1;
            }
            a -= tmp;
            result += multiple;     //最大的倍数
        }
        return negative ? -result : result;
    }


    /**
     * 166. Fraction to Recurring Decimal 分数 转 小数
     *  1/2 = "0.5".   但如果除不尽有循环小数，就用()包住  2 / 3 = "0.(6)"
     * @param numerator
     * @param denominator
     * @return
     *
     * 用hashmap存余数.. 只要remainder存在了，就会一直重复循环下去，这时就用 () 包着并break
     */
    public String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0)
            return "0";

        StringBuilder result = new StringBuilder();

        if (numerator < 0 ^ denominator < 0) {
            result.append("-");
        }

        long dividend = Math.abs(Long.valueOf(numerator));
        long divisor = Math.abs(Long.valueOf(denominator));

        result.append(dividend / divisor);

        long remainder = dividend % divisor;

        if (remainder == 0)
            return result.toString();

        result.append(".");

        Map<Long, Integer> map = new HashMap<>();       // 放重复的余数, 对应的index
        map.put(remainder, result.length());

        while (remainder != 0) {
            remainder *= 10;
            result.append(remainder / divisor);
            remainder %= divisor;

            // once remainder in map, it will repeat.. so wrap it with () and break
            if (map.containsKey(remainder)) {
                result.insert(map.get(remainder), "(");
                result.append(")");
                break;
            } else {
                map.put(remainder, result.length());
            }
        }
        return result.toString();
    }
    
    
    /**  50. Pow(x, n)
     * 每次n/2, 然后下面想底数增大
     * 如果边界条件 n为int.min. 那么long absN = Math.Abs((long)n);最后return要看是否n<0
     */
    public double myPow(double x, int n) {
        if (n == 0)     return 1;
        
        if (n < 0) {
            x = 1/x;
            n = -n;
        }
            
        double result = 1.0;
        while (n > 0) {
            if (n % 2 == 1) {
                result *= x;
            }
            x *= x;
            n /= 2.0;
        }
        return result;
    }

    // recursion
    public double myPow1(double x, int n) {
        if (n == 1)     return x;
        if (n == 0)     return 1;

        if (n < 0) {

            if (n == Integer.MIN_VALUE)
                return 1.0 / (myPow(x, Integer.MAX_VALUE) * x);

            x = 1 / x;
            n = -n;
        }

        return n % 2 == 0 ? myPow(x * x, n / 2) : myPow(x * x, n / 2) * x;
    }
    
    
    
    /** Sqrt(x) 求root
     * 找不到就找最近似的
     * @param x
     * @return
     */
    public static int sqrt(int x) {
        int lo = 1;		//也能为1
        int hi = x;				
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (mid == x / mid) {		//mid相乘会越界，所以用除/
                return mid;
            } else if (mid > x / mid) {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        
        // ===============也可以long mid
        while (lo <= hi) {
            long mid = lo + (hi - lo) / 2;
            if (mid * mid == x) {
                return (int)mid;
            } else if (mid * mid > x) {
                hi = (int)mid - 1;
            } else {
                lo = (int)mid + 1;
            }
        }
        
        return hi;			// 取小于的最大值. hi比较小
    }
    
    
    public float mySqrt(int x) {
    	float lo = 1, hi = x;
    	float e = 0.000001f;
        
        while (hi - lo > e) {
        	float mid = lo + (hi - lo) / 2;
            if (mid * mid == x) {
                return mid;
            } else if (mid * mid < x) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return hi;
    }
    
    
    /** Sqrt(x) 求root
     * 找不到就找最近似的, 用数学方法
     * @param x
     * @return
     */
    public static int sqrt2(int x) {
    	long r = x;				// 用long, 因为下面r*r会越界。且包括了0的情况
        while (r * r > x) {		// 如果r > x / r，r == 0的话就会出错
            r = (r + x / r) / 2;
        }
        return (int) r;	
    }
    
    
    /** 367. Valid Perfect Square
     * 看是否为 平方数  16是的，14false
     * @param num
     * @return
     */
    public boolean isPerfectSquare(int num) {
        int lo = 0, hi = num;       //lo为0或1都行
        while (lo <= hi) {
            long mid = lo + (hi - lo) / 2;
            if (mid * mid == num) {
                return true;
            } else if (mid * mid < num) {
                lo = (int) mid + 1;
            } else {
                hi = (int) mid - 1;
            }
        }
        return false;
    }


    /**
     * 633. Sum of Square Numbers
     * whether there're two integers a and b such that a * a + b * b = c.
     * @param c
     * @return
     */
    public boolean judgeSquareSum(int c) {
        for (int i = 0; i <= Math.sqrt(c); i++) {
            int j = (int) Math.sqrt(c - i * i);
            if (j * j == c - i * i) {
                return true;
            }
        }
        return false;
    }

    public boolean judgeSquareSum1(int c) {
        int left = 0, right = (int) Math.sqrt(c);

        while (left <= right) {             // need <= .. not just < . because after sqrt may be the same
            int sum = left * left + right * right;
            if (sum == c) {
                return true;
            } else if (sum < c) {
                left++;
            } else {
                right--;
            }
        }
        return false;
    }
    
    
    
    /** 204. Count Primes
     * Count the number of prime numbers less than a non-negative number, n.
     * @param n
     * @return
     *
     * 主要是，先把是prime的数，及其倍数 都填满.. notPrime[i] = true. 这样 剩下的就是prime了
     * runtime complexity is O(n log log n)
     */
    public int countPrimes(int n) {
        boolean[] notPrime = new boolean[n];
        int count = 0;

        for (int i = 2; i < n; i++) {
            if (!notPrime[i]) {          // is prime的话，就把后面这个prime的倍数填满变true
                count++;
                for (int j = 2; i * j < n; j++) {       // 把prime的倍数填满
                    notPrime[i * j] = true;
                }
            }
        }
        return count;
    }
    
    
    /** 263. Ugly Number
     * Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. 
 		IE. 6, 8 are ugly while 14 is not ugly since it includes another prime factor 7.
     * Note that 1 is typically treated as an ugly number.
     * @param num
     * @return
     */
    public boolean isUgly(int num) {
        for (int i = 2; i < 6 && num > 0; i++) {
            while (num % i == 0) {
                num /= i;
            }
        }
        
        return num == 1;
    }
    
    

    // worse  但是比较好理解
   //用minHeap找每次的min，并Poll出来.. 然后再把min跟2，3，5相乘放进heap里。但是会有重复，所以用set来去重复
     public int nthUglyNumberHeap(int n) {
     	Queue<Long> minHeap = new PriorityQueue<>();
         Set<Long> set = new HashSet<>();
         Long[] p = new Long[3];
         p[0] = Long.valueOf(2);
         p[1] = Long.valueOf(3);
         p[2] = Long.valueOf(5);
         
         
         for (int i = 0; i < 3; i++) {
             minHeap.add(p[i]);
             set.add(p[i]);
         }
         Long num = Long.valueOf(1);
         for (int i = 1; i < n; i++) {
             num = minHeap.poll();
             for (int j = 0; j < 3; j++) {
                 if (!set.contains(p[j] * num)) {
                     set.add(p[j] * num);
                     minHeap.add(p[j] * num);
                 }
             }
         }
         return num.intValue();
     }
     
    
    /** 264. Ugly Number II
     *  find the n-th ugly number. prime factors only include 2, 3, 5. 
		For example, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 is the sequence of the first 10 ugly numbers.
     * @param n
     * @return
     * 也是跟min乘，但是要知道乘几个倍数。可以分成三组
     *  (1) 1×2, 2×2, 3×2, 4×2, 5×2, …
		(2) 1×3, 2×3, 3×3, 4×3, 5×3, …
		(3) 1×5, 2×5, 3×5, 4×5, 5×5
	   Merge sort，取最小的，再往后走
     */
    public int nthUglyNumber(int n) {
        if (n <= 1) {
            return n;
        }
        
        int p2 = 0, p3 = 0, p5 = 0;
        int[] ugly = new int[n];
        ugly[0] = 1;
        
        for (int i = 1; i < n; i++) {
            ugly[i] = Math.min(Math.min(2 * ugly[p2], 3 * ugly[p3]), 5 * ugly[p5]);
            if (ugly[i] == ugly[p2] * 2)    p2++;
            if (ugly[i] == ugly[p3] * 3)    p3++;   // not else, cause 2*3 == 3 * 2 这样避免后面的重复
            if (ugly[i] == ugly[p5] * 5)    p5++;
        }
        
        return ugly[n-1];
    }
    
    
    /** 313. Super Ugly Number
     * 跟II一样，只是prime是一个数组.. 做法跟上面一样
     * @param n
     * @param primes
     * @return
     * O(kN)  - k = primes长度
     */
    public int nthSuperUglyNumber(int n, int[] primes) {
        if (n <= 1) {
            return n;
        }
        
        int[] ugly = new int[n];
        int len = primes.length;
        int[] index = new int[len]; //index of primes
        ugly[0] = 1;
        
        for (int i = 1; i < n; i++) {
            int min = Integer.MAX_VALUE;
            for (int p = 0; p < len; p++) {
                min = Math.min(min, primes[p] * ugly[index[p]]); 	// prime * prime的倍数
            }
            
            ugly[i] = min;
            
            for (int p = 0; p < len; p++) {
            		// 下面这个同上
                if (primes[p] * ugly[index[p]] == min) {	//也可以ugly[i] % primes[p] == 0，但是更慢，因为要一直mod
                    index[p]++;
                }
            }
            
         // 上面2个for可以合并成一个
            ugly[i] = Integer.MAX_VALUE;	//min
            for (int p = 0; p < len; p++) {
                if (primes[p] * ugly[index[p]] == ugly[i-1]) {
                    index[p]++;			// index[p]先在这加， 下面那句就用+1以后的
                }
                ugly[i] = Math.min(ugly[i], primes[p] * ugly[index[p]]);
            }
        }
        return ugly[n-1];
    }
    
    
    /** 65. Valid Number
     * "abc" => false, "1 a" => false, "2e10" => true
     * @param s
     * @return
     */
    public boolean isNumber(String s) {
        boolean hasDot = false;
        boolean hasE = false;
        boolean hasNum = false;
        boolean numAfterE = true;       //记得是TRUE
        s = s.trim();
        
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if ('0' <= c && c <= '9') {
                hasNum = true;
                numAfterE = true;
            } else if (c == '.') {
                if (hasE || hasDot)     
                    return false;
                hasDot = true;
            } else if (c == 'e') {
                if (hasE || !hasNum)     
                    return false;
                hasE = true;
                numAfterE = false;
            } else if (c == '-' || c == '+') {
                if (i > 0 && s.charAt(i-1) != 'e') {
                    return false;
                }
            } else {
                return false;
            }
        }
        return hasNum && numAfterE;
    }
    
    
    
    /**150. Evaluate Reverse Polish Notation
     * ["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9
  		["4", "13", "5", "/", "+"] -> (4 + (13 / 5)) -> 6
     * @param tokens
     * @return
     * 简单。。 用stack就行。
     * 要注意 3/0 的情况。
     * 除了用switch，也可以用if-else
     */
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for (String s : tokens) {
            switch (s) {
                case "+":
                    stack.push(stack.pop() + stack.pop());
                    break;
                case "-":
                    stack.push(-stack.pop() + stack.pop());
                    break;
                case "*":
                    stack.push(stack.pop() * stack.pop());
                    break;
                case "/":
                    int n1 = stack.pop();
                    int n2 = stack.pop();
                    if (n1 == 0 || n2 == 0) {
                        stack.push(0);
                    } else {
                        stack.push(n2 / n1);
                    }
                    break;
                
                default:
                    stack.push(Integer.parseInt(s));
            }
        }
        return stack.pop();
    }
    
    
    
    /** 224. Basic Calculator
     * 有 ( ) + - . 没有乘除
     * @param s
     * @return
     * 用stack区分 ( )
     * sign表示 1 +, -1 - 并且sign也要push进stack里
     * 1. 用total随时记录当前的sum
     * 2. 遇到(, 就push total和sign，并且当前总和Reset为0
     * 3. 遇到), 就要pop，把 新的total * sign， 还要加上之前stack里的 
     */
    public int calculate(String s) {
        Stack<Integer> stack = new Stack<>();
        int total = 0;
        int num = 0;
        int sign = 1;
        for(int i = 0; i < s.length(); i++){
            char c = s.charAt(i);
            if (Character.isDigit(c)) {
                num = 0;
                while (i <  s.length() && Character.isDigit(s.charAt(i))) {
                    num = num * 10 + (s.charAt(i) - '0');
                    i++;
                }
                i--;        // will ++ later
                total += num * sign;
            } else if (c == '+') {
                sign = 1;
            } else if (c == '-') {
                sign = -1;
            } else if (c == '(') {
                stack.push(total);        // push total sum
                stack.push(sign);
                total = 0;              // total reset to 0, start over
                sign = 1;
            } else if (c == ')') {       
                total = stack.pop() * total + stack.pop();
            }
        }
        return total;
    }
    
    
    
    /** 227. Basic Calculator II
     * " 3+5 / 2 " = 5. 会有空格，没有(). 有 + - *, /
     * @param s
     * @return
     * 用stack放每个num, 也放负数。。最后都 +加起来 就行。
     * 每次遇到num以后，再根据之前的sign来判断要 负数 还是 * / 放stack里
     * 遇到* 或 / 就 s.peek * num，再push进去
     */
    public int calculateIIStack(String s) {
        // stack 放 num.. 也放负数。。最后都 +加起来 就行
        Stack<Integer> stack = new Stack<>();
        int len = s.length();
        char sign = '+';
        int preNum = 0;
        int total = 0;

        for (int i = 0; i < len; i++) {
            while (i < len && s.charAt(i) == ' ') {
                i++;
            }

            int num = 0;

            while (i < len && Character.isDigit(s.charAt(i))) {
                num = num * 10 + s.charAt(i) - '0';
                i++;
            }

            // sign是cur num之前的sign..  sign, num 这样的顺序
            if (sign == '+') {
                stack.push(num);
//                total += preNum;              // 这些是 不用stack，只用preNum记录的情况 O(1) space.. 具体看下面解法的解释
//                preNum = num;
            } else if (sign == '-') {
                stack.push(-num);
//                total += preNum;
//                preNum = -num;
            } else if (sign == '*') {
                stack.push(stack.pop() * num);
//                preNum *= num;
            } else if (sign == '/') {
                stack.push(stack.pop() / num);
//                preNum /= num;
            }

            if (i < len) {
                sign = s.charAt(i);     // 更新cur sign.
            }
        }

         while (!stack.isEmpty()) {
             total += stack.pop();
         }

        return total + preNum;
    }
    
    
    /** 227. Basic Calculator II
     * " 3+5 / 2 " = 5. 会有空格，没有(). 有 + - *, /
     * @param s
     * @return
     * 这题不需要stack, 直接用preVal存前面的值，和cur为现在的
     * 要注意几点
     * a. sign初始化为+， 而且每次更新
     * b. 只有+ - 才要 更新sum， 且 sum += pre... 是pre不是cur
     * c. *和/只用更新pre, 不要更新sum...因为优先级高，需要都算完*, /才能知道pre是什么
     */
    public int calculateII(String s) {
        int len = s.length();
        int cur = 0;
        int pre = 0;    //previous value
        int sum = 0;
        char sign = '+';    //store sign  初始为+
        int i = 0;
        
        while (i < len) {
            while ( i < len && s.charAt(i) == ' ') {
                i++;
            }
            cur = 0;
            while (i < len && Character.isDigit(s.charAt(i))) {		//也可以 c >= 0 && c <= 9
                cur = cur * 10 + (s.charAt(i) - '0');
                i++;
            }
            
            if (sign == '+') {      //the sign before num
                sum += pre;
                pre = cur;
            } else if (sign == '-') {
                sum += pre;			// 注意 + - 都是 更新sum， 且 sum += pre... 是pre不是cur
                pre = -cur;
            } else if (sign == '*') {
                pre = pre * cur;            //don't update sum, just pre，因为优先级高，还没算完
            } else if (sign == '/') {
                pre = pre / cur;
            }
            
            if (i == len) 
                break;
            
            sign = s.charAt(i);		// getting new sign
            i++;
        }
        
        return sum + pre;		//记得还要加上pre
    }


    /**
     * 679. 24 Game
     * 有4张牌，从1 - 9. 用*, /, +, -, (, ) 看能否组成24.. 只能用一个digit, 不能变成12 * 12这种
     * Input: [4, 1, 8, 7]
     * Output: True
     * Explanation: (8-4) * (7-1) = 24
     *
     * 只能recursion试不同结果了
     *
     * 注意这里要算上分数的.. 而不是整除。。 所以要用double 保留精度..
     *
     * 每次选2张牌.. 然后不同组合方式作为candidate来dfs.. 注意0的问题.. < 0.001即可
     *
     * 选了牌以后要删掉这些牌，然后加上next result, 再recursion算
     */
    public boolean judgePoint24(int[] nums) {
        List<Double> list = new ArrayList<>();
        for (int i : nums) {
            list.add((double) i);
        }
        return helper(list);
    }

    private boolean helper(List<Double> list) {
        if (list.size() == 1) {
            // 如果此时list只剩下了一张牌, 看结果是不是24
            return Math.abs(list.get(0) - 24.0) < 0.001;
        }

        // 选两张牌
        for(int i = 0; i < list.size(); i++) {
            for(int j = i + 1; j < list.size(); j++) {
                double n1 = list.get(i);
                double n2 = list.get(j);

                // 下一轮可能产生的结果组合
                List<Double> nextList = new ArrayList<>();
                nextList.addAll(Arrays.asList(n1 + n2, n1 - n2, n2 - n1, n1 * n2));
                // 除法处理0的情况
                if (Math.abs(n1) > 0.001)   nextList.add(n2 / n1);
                if (Math.abs(n2) > 0.001)   nextList.add(n1 / n2);

                list.remove(j);     // 因为这轮用了这2个数，要删掉, 先删后面的j
                list.remove(i);

                // 开始处理下一轮结果
                for (double next : nextList) {
                    list.add(next);

                    if (helper(list))
                        return true;

                    list.remove(list.size() - 1);
                }

                list.add(i, n1);    // 先加前面的i, idx比较小
                list.add(j, n2);
            }
        }
        return false;
    }

    
    
    /** 282. Expression Add Operators
     * 0-9 add binary operators (not unary) +, -, or *
     * "105", 5 -> ["1*0+5","10-5"]   可以组成不同长度的digit，比如10..
     * @param num
     * @param target
     * @return
     * 正常的dfs.. 要注意几点
     * 1. overflow，所以要long cur
     * 2. 0的出现，可以 10，101， 也可以0自己，但不能有0开头的 01。
     * 		一旦最后那样 num.charAt(pos) == '0' && i != pos，就要停。
     * 3. 算乘数时，记得另外一个变量存multiply表示 之前pre要乘的数。total - pre + pre * cur
     * 		if you have a sequence of 12345 and you have proceeded to 1 + 2 + 3, now your eval is 6 right? 
     * 		If you want to add a * between 3 and 4, you would take 3 as the digit to be multiplied, 
     * 			so you want to take it out from the existing eval. 
     * 		You have 1 + 2 + 3 * 4 and the eval now is (1 + 2 + 3) - 3 + (3 * 4)
     * 4. 用char[] 来代替str.substring, 用stringBuilder都更快 （看dfs_solution的）
     * 		long cur = Long.parseLong(new String(num, pos, i - pos + 1));  //最后是count个数，中间是从哪开始
     */
    public List<String> addOperators(String num, int target) {
        List<String> result = new ArrayList<>();
        helper(result, num, target, "", 0, 0, 0);
        return result;
    }
    
    public void helper (List<String> result, String num, int target, String path, int pos, long total, long pre) {
        if (pos == num.length()) {
            if (total == target) {
                result.add(path);
            }
            return;
        }
        
        for (int i = pos; i < num.length(); i++) {
            //  starts with 0 && i在pos后面，就是说 01的情况
        	if (num.charAt(pos) == '0' && i != pos)
                break;      // don't want to proceed if  "01", but ok if just "0", then i==pos
        	
            Long cur = Long.parseLong(num.substring(pos, i + 1));    // substring(pos, i+1)
            
            if (pos == 0) {		//只是第一个数.. 其他情况是后面else的
                helper(result, num, target, path + cur, i + 1, cur, cur);   
            } else {
                helper(result, num, target, path + "+" + cur, i + 1, total + cur, cur);
                helper(result, num, target, path + "-" + cur, i + 1, total - cur, -cur);    //pre = -cur
                helper(result, num, target, path + "*" + cur, i + 1, total - pre + pre * cur, cur * pre);
            }
        }
    }
   
    
    
    /** 246. Strobogrammatic Number
     * A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).
     * 判断num是否为那种
     * @param num
     * @return
     */
    public boolean isStrobogrammatic(String num) {
        int i = 0, j = num.length() - 1;
        while (i <= j) {
            if (num.charAt(i) == num.charAt(j) && (num.charAt(i) == '0' || num.charAt(i) == '1' || num.charAt(i) == '8')) {
                i++;
                j--;
            } else if ((num.charAt(i) == '6' && num.charAt(j) == '9') || (num.charAt(i) == '9' && num.charAt(j) == '6')) {
                i++;
                j--;
            } else {
                return false;
            }
        }
        
        // shorter 
        for (int m=0, n=num.length()-1; i <= j; i++, j--)
            if (!"00 11 88 696".contains(num.charAt(m) + "" + num.charAt(n)))
                return false;
        return true;
    }
    
    
    
    /** 247. Strobogrammatic Number II
     * Given n = 2, return ["11","69","88","96"]
     * @param n
     * @return
     * 记得判断 如果在最外面 i = n 时，不要用0
     * 
     * 1. 先根据n的奇偶来初始化result，从中间开始，往外扩散
     * 2. for循环result，并且建一个新的tmp, 每次在外面加2个数给中间的string
     * 3. 加完tmp后再赋值给result，这样可以下一次再for result加数
     */
    public List<String> findStrobogrammatic(int n) {
        List<String> odd = Arrays.asList("0", "1", "8");
        List<String> even = Arrays.asList("");
        List<String> result = n % 2 == 0 ? even : odd;		// 从中间开始往外扩散
        
        for (int i = n % 2 + 2; i <= n; i += 2) {		//i从2或3开始，因为下面要每次加2个数
            List<String> list = new ArrayList<>();
            for (String s : result) {
                if (i != n)     list.add("0" + s + "0");
                
                list.add("1" + s + "1");
                list.add("6" + s + "9");
                list.add("8" + s + "8");
                list.add("9" + s + "6");
            }
            result = list;
        }
        return result;
        
   //    return recHelper(n, n);		recursion
    }
    
    
    // recursion也比较好理解
    public List<String> recHelper(int n, int m) {
        if (n == 0)     return new ArrayList<String>(Arrays.asList(""));
        if (n == 1)     return new ArrayList<String>(Arrays.asList("0", "1", "8"));
        
        List<String> list = recHelper(n - 2, m);		//当是list已经拿到中间的string了
        
        List<String> result = new ArrayList<>();
        
        for (String s : list) {						//在里层string的外层加2个数
            if (n != m)     result.add("0" + s + "0");      // 0 cannot at the begining, so n != m can avoid
            
            result.add("1" + s + "1");
            result.add("6" + s + "9");
            result.add("8" + s + "8");
            result.add("9" + s + "6");
        }
        return result;
    }
    
    
    /** 248. Strobogrammatic Number III
     * 返回符合范围的个数  Given low = "50", high = "100", return 3. 
     * Because 69, 88, and 96 are three strobogrammatic numbers.
     * @param low
     * @param high
     * @return
     * 这方法naive
     */
    public int strobogrammaticInRangeNaive(String low, String high) {
        List<String> list = new ArrayList<>();
        for (int n = low.length(); n <= high.length(); n++) {
            list.addAll(recHelper(n, n));
        }
        
        int count = 0;
        for (String num : list) {
            if ((num.length() == low.length() && num.compareTo(low) < 0) || 
                (num.length() == high.length() && num.compareTo(high) > 0)) {
                    continue;
                }
            count++;
        }
        return count;
    }
    
    
    
    private static final char[][] pairs = {{'0', '0'}, {'1', '1'}, {'6', '9'}, {'8', '8'}, {'9', '6'}};

    public int strobogrammaticInRange(String low, String high) {
        int[] count = {0};
        for (int len = low.length(); len <= high.length(); len++) {
            char[] c = new char[len];
            dfs(low, high, c, 0, len - 1, count);
        }
        return count[0];
    }

    public void dfs(String low, String high , char[] c, int left, int right, int[] count) {
        if (left > right) {
            String s = new String(c);
            if ((s.length() == low.length() && s.compareTo(low) < 0) || 
                (s.length() == high.length() && s.compareTo(high) > 0)) {
                return;
            }
            count[0]++;
            return;
        }
        for (char[] p : pairs) {
            c[left] = p[0];
            c[right] = p[1];
            if (c.length != 1 && c[0] == '0') {		 //如果长度>1, 第一位不能为0
                continue;
            }
            if (left == right && p[0] != p[1]) {	//odd, 需要相同的数，00，11，88
                continue;
            }
            dfs(low, high, c, left + 1, right - 1, count);
        }
    }
    
    
    
    /** 360. Sort Transformed Array
     * 给个sorted Array，要求每个数用f(x) = ax2 + bx + c to each element x in the array.输出的要是sorted
     * @param nums
     * @param a
     * @param b
     * @param c
     * @return
     * 这里要分2种情况就行:
     * 1. a >= 0时，两边 > center
     * 2. a < 0时，ends < center
     * 至于后面 a == 0或 b>0都不用管，只需调用cal()算出到底是 nums[i]还是nums[j]算出来的大，即可。
     * 用two pointer来keep两端
     */
    public int[] sortTransformedArray(int[] nums, int a, int b, int c) {
        int n = nums.length;
        int[] arr = new int[n];
        int i = 0, j = n - 1;
        int k = a >= 0 ? n - 1 : 0;     
        // a >= 0, end > center;  a < 0时，2ends < center
        while (i <= j) {
            if (a >= 0) {
                arr[k--] = cal(nums[i], a, b, c) >= cal(nums[j], a, b, c) ? cal(nums[i++], a, b, c) : cal(nums[j--], a, b, c);
            } else {        //a < 0要从小的开始，所以是j--先
                arr[k++] = cal(nums[i], a, b, c) >= cal(nums[j], a, b, c) ? cal(nums[j--], a, b, c) : cal(nums[i++], a, b, c);
            }
        }
        return arr;
    }
    
    public int cal(int x, int a, int b, int c) {
        return a * x * x + b * x + c;
    }
    
    
    
    /** 400. Nth Digit
     * 在1,2,3...序列中找到第n个digit。比如 第11 是 0, 因为10拆成2个digits。第12也是1
     * @param n
     * @return
     * 1-------9 9*1 = 9 digits
	   10-----99 90 *2 = 180 digits
	   100---999 900 * 3 = 2700 digits
	   gave N = 1000, then 1000-9-180 = 811, it means the 811th digit local in [100, 999], and we know each number like 100 has three digit, so 811 / 3 = 270,
	   Then, we know the 270th number in [100, 999], is 270th + 100 (start from 100) = 370.
	   370 still has three digit, which one is the answer? 3, 7, 0
     */
    public int findNthDigit(int n) {
        int digits = 1;
        long base = 9;
        int start = 1;
        
        while (n > base * digits) {
            n -= base * digits;     
            base *= 10;     
            digits++;
            start *= 10;    // 加上10或100或1000
        }
        
        start += (n - 1) / digits;
        String s = Integer.toString(start);
        return s.charAt((n - 1) % digits) - '0';
    }
    
    
    
    /** 168. Excel Sheet Column Title
     * 1 -> A. .. 28 -> AB
     * @param n
     * @return
     */
    public String convertToTitle1(int n) {
        StringBuilder sb = new StringBuilder();
        while (n > 0) {
            n--;
            sb.insert(0, (char) (n % 26 + 'A'));
            n /= 26;
        }
        return sb.toString();
    }
    
    
    public String convertToTitle(int n) {
        return n > 0 ? convertToTitle(--n / 26) + (char)(n % 26 + 'A') : "";
    }
    
    
    /** 171. Excel Sheet Column Number
     * A -> 1,  AB -> 28 
     * @param s
     * @return
     */
    public int titleToNumber(String s) {
        int num = 0;
        
        for (char c : s.toCharArray()) {
            num = num * 26 + (c - 'A') + 1;
        }
        return num;
    }
    
    
    public String intToBinary(int n) {
    	String s = "";
    	/*
    	for (int i = 1 << 31; i > 0; i = i / 2) {
    		s += (n & i) == 1 ? "1" : "0"; 
    	}
    	*/
    	
    	while (n > 0)
        {
            s =  ( (n % 2 ) == 0 ? "0" : "1") +s;
            n = n / 2;
        }
    	return s;
    }


    /**
     * 593. Valid Square
     * 给4个点，看是否是正方形
     *
     * 看4条边长相等，并且2条对角线 相等， 长于边长..
     */
    public boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
        Set<Integer> set = new HashSet<>(Arrays.asList(dist(p1, p2), dist(p2, p3), dist(p3, p4), dist(p1, p4), dist(p1, p3), dist(p2, p4)));

        return !set.contains(0) && set.size() == 2;     // 只有对角线和边长.. set去重
    }

    private int dist(int[] p1, int[] p2) {
        return (p2[1] - p1[1]) * (p2[1] - p1[1]) + (p2[0] - p1[0]) * (p2[0] - p1[0]);
    }


    public boolean validSquare1(int[] p1, int[] p2, int[] p3, int[] p4) {
        int[][] p = {p1, p2, p3, p4};
        Arrays.sort(p, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);

        return dist(p[0], p[1]) != 0 &&
                   dist(p[0], p[1]) == dist(p[1], p[3]) &&
                   dist(p[1], p[3]) == dist(p[3], p[2]) &&
                   dist(p[3], p[2]) == dist(p[2], p[0]) &&
                   dist(p[0], p[3]) == dist(p[1],p[2]);

    }

    public boolean validSquare2(int[] p1, int[] p2, int[] p3, int[] p4) {
        long[] distances = {dist(p1, p2), dist(p2, p3), dist(p3, p4), dist(p1, p4), dist(p1, p3), dist(p2, p4)};

        long diagnal = 0;       // 只有2个对角线长
        long side = Integer.MAX_VALUE;          // 其他4边短点 same

        for (long d : distances) {
            diagnal = Math.max(diagnal, d);
            side = Math.min(side, d);
        }

        int count = 0;
        for (long d : distances) {
            if (d == diagnal) {
                count++;
            } else if (d != side) {
                return false;
            }
        }

        return count == 2;
    }


    /**
     * 780. Reaching Points - hard  不大懂
     * (x, y) and transforming it to either (x, x+y) or (x+y, y). 看(sx, sy)能否变成(tx, ty)
     *
     * https://leetcode.com/problems/reaching-points/discuss/230588/Easy-to-understand-diagram-and-recursive-solution
     * 这题关键是从 end (tx, ty) 找，这样能能快缩小范围不会TLE
     */
    public boolean reachingPoints(int sx, int sy, int tx, int ty) {
        // start from end
        while (tx >= sx && ty >= sy) {
            if (tx > ty) {
                if (ty > sy)    tx %= ty;
                else            return (tx - sx) % ty == 0;
            } else if (tx < ty) {
                if (tx > sx)    ty %= tx;
                else            return (ty - sy) % tx == 0;
            } else {
                break;
            }
        }
        return tx == sx && ty == sy;
    }


    
    public static void main(String[] args) {
    	MathSolution sol = new MathSolution();
    	System.out.println(sol.mySqrt(85));
    	System.out.println("add float " + sol.addFloatStrings("2.2678", "21.95"));
    	System.out.println(sol.addOperators("232",8));
    }
}
