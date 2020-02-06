//Leetcode 282 Expression Add Operators
/**
    * @param num: a string contains only digits 0-9
    * @param target: An integer
    * @return: return all possibilities
    * Input: num = "123", target = 6 ---> Output: ["1+2+3", "1*2*3"] 
    * using +, -, or *
*/
/* Avoid overflow, use long
## Time 
* Each digit have 3 situations(+,-,*)
* O(3^n)
T(n) = 2 * T(n-1) + 2 * T(n-2) + 2 * T(n-3) + ... + 2 *T(1);
T(n-1) = 2 * T(n-2) + 2 * T(n-3) + ... 2 * T(1);
Thus T(n) = 3T(n-1);
## Space 
* For nums = 00000....0 target 0 
* you get output of 3^(n-1) string
* O(3^n)
* O(n) for recursion stack*/
//枚举数字后再枚举数字前的符号 （第一个数字前不枚举符号
//sum 作为中间状态记录下前面的和
//增加一个状态 lastFactor (lastF) 记录下如果枚举乘号的话，当前的数应该和谁乘，也就是最后一个因子是谁
public class Solution {
	public List<String> addOperators(String num, int target) {
		List<String> ans = new ArrayList<String>();
		dfs();
		return ans;
	}


	//str is the current string of the num
	public void dfs(int start, String num, int target, List<String> ans, 
		long sum, String str, long lastF) {
		if (start == num.length()) {
			if (sum == target) {
				ans.add(str);
			}
			return;
		}

		for (int i = 0; i < num.length(); i++) {
			// get the total String to long, ex 23
			long x = long.parseLong(num.substring(start, i + 1));
			//first time
			if (start == 0) {
				dfs(i+1, num, target, ans, x, "" + x, x)
			} else {
				dfs(i+1, num, target, ans, x, str + "*" + x, x)
			}

		}
	}

}
//convert string to long --> Long.parseLong("String")