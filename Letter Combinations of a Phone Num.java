//(Leetcode 17) Letter Combinations of a Phone Number
//backtracking 解法 O(3^n),Space: O(1)
//add all these letters, String--> s.charAt to get the value
//String. length()
//Input: "5" Output: ["j", "k", "l"]
/**
  * @param digits: A digital string
  * @return: all posible letter combinations
*/

public class Solution {
    private String[] mapping = new String[] {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    
    public ArrayList<String> letterCombinations(String digits) {
        ArrayList<String> ans = new ArrayList<String>();
        
        if (digits == null || digits.equals("")) {
            return ans;
        }
        
        dfs(ans, digits, 0, "");

        return ans;
    }

   public void dfs(ArrayList<String> ans, String digits, int level, String str) {       
        //after condition, there is a space, coding style!
		if (level == digits.length()) {
            ans.add(str);
            return;
        }
        
    
        //change it to ASCII code, and then get the string
        String letters = mapping[digits.charAt(level) - '0'];
        
        for (int i = 0; i < letters.length(); i++) {
        	dfs(ans, digits, level+1, str+letters.charAt(i));
        }
    }
}