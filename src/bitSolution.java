package bit;

import java.util.ArrayList;

public class bitSolution {
	
	/** Gray Code
	 * @param n
	 * @return
	 * Binary Code ：1011 要转换成Gray Code
	 * 1011 = 1（照写第一位）, 1(第一位与第二位异或 1^0 = 1), 1(第二位异或第三位， 0^1=1), 0 (1^1 =0) = 1110
	 * 其实就等于 (1011 >> 1) ^ 1011 = 1110
	 */
	public ArrayList<Integer> grayCode(int n) {
        int size = 1 << n;              // = 2^n
        ArrayList<Integer> result = new ArrayList<Integer>();
        for (int i = 0; i < size; i++) {
            result.add(i ^ (i >> 1));
        }
        return result;
    }
	
	
	/** 338. Counting Bits
	 * 给一个num，0 ≤ i ≤ num，算出i含有1的个数
	 * For num = 5 you should return [0,1,1,2,1,2].
	 * @param num
	 * @return
	 * 分出最低一位，和其他位
	 * 5-101可以分成"10" & "1"。"10"就是dp[i/2]的结果，最后奇偶就i%2
	 */
	public int[] countBits(int num) {
        int[] arr = new int[num + 1];
        for (int i = 1; i <= num; i++) {
       //     arr[i] = arr[i / 2] + i % 2;
            arr[i] = arr[i >> 1] + (i & 1);
        }
        return arr;
    }
	
	
	/** 136. Single Number
	 * 数组里所有元素都出现2次，除了一个。找到那个single number
	 * @param nums
	 * @return
	 * 跟missing number一样。
	 * 用XOR . b^b = 0  -->  a^b^b = a
	 */
	public int singleNumber(int[] nums) {
        int xor = 0;
        for (int num : nums) {
            xor ^= num;
        }
        return xor;
    }
	
	
	/** 393. UTF-8 Validation
	 * @param data
	 * @return
	 * 理解题意...
	 * 
	 * 注意else if ((n >> 7) == 1)  return false。 因为1000000或者1111111都是错的
	 */
	public boolean validUtf8(int[] data) {
        //count这个byte后面要跟多少个10开头的数..1110xxxx的话后面要有2个10开头的数
        int count = 0;   
        for (int n : data) {
            if (count == 0) {       //initial或者前面都符合，开始新的判断
                if ((n >> 5) == 0b110)  count = 1;
                else if ((n >> 4) == 0b1110)  count = 2;
                else if ((n >> 3) == 0b11110)  count = 3;
                else if ((n >> 7) == 1)  return false;   //开头判断的数不能10开头(因为后面需要算几个10开始的)
            } else {
                if ((n >> 6) != 0b10)   return false;   //后面要10开始
                else    count--;
            }
        }
        return count == 0;      //要刚好满足几个10的。
    }
	
	
	
	/** 191. Number of 1 Bits
	 * 给一个数，算出多少个1. 比如11的二进制是1011，有3个1
	 * @param n
	 * @return
	 */
	public int hammingWeight(int n) {
        int count = 0;
        for (int i = 0; i < 32; i++) {		//int 只有32位
            count += n & 1;
            n >>= 1;
        }
        return count;
    }
	

	public int hammingWeight1(int n) {
        int count = 0;
        while(n != 0){
            n = n & (n-1);		// 这个能移掉最右边的1..比如0100，那么n-1是0011.then &以后就是0，移掉唯一的1
            count++;
        }
        return count;
    }
	
	
	/** 461. Hamming Distance
	 * 跟上题很像，求出2个数之间不同的个数
	 *  1   (0 0 0 1)
		4   (0 1 0 0)
		       ↑   ↑
	 * @param x
	 * @param y
	 * @return
	 */
	public int hammingDistance(int x, int y) {
        int xor = x ^ y;
        int count = 0;
        
        while (xor != 0) {
            count += xor & 1;
            xor >>= 1;
        }
        return count;
    }
	
	
	
}
