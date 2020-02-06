//294 Flip Game II
//翻转游戏：给定一个只包含两种字符的字符串：+和-，
//你和你的小伙伴轮流翻转"++"变成"--"。当一个人无法采取行动时游戏结束，另一个人将是赢家。
//编写一个函数判断是否能够保证先手胜利。
// s = "-++++-"
// 返回true，表示你一定会赢。只需翻转第二个和第三个加号使得
// s = "-+--+-"
// 此时对方无法继续操作（没有两个连续的加号）。
//此题可以用带回溯的搜索解决。每次搜索时尝试翻转两个连续加号，
//如果翻转后的局面是对手输，那么这个翻转是可行的；否则回溯到原来状态
//，继续搜索下一个可能的翻转操作。该算法的时间复杂度是指数级别的，

public class Solution {
    /**
     * @param s: the given string
     * @return: if the starting player can guarantee a win
     */
    //use the hashmap to store the information
    public class Solution {
	    public boolean canWin(String s) {
	        boolean[] state = new boolean[s.length()];
	        for (int i = 0; i < s.length(); i++) {
	            if (s.charAt(i) == '+') {
	                state[i] = true;
	            } else {
	                state[i] = false;
	            }
	        }
	        //create a boolean array, put all the '+' as teh true,
	        // and then is false
	        return search(state);
	    }
	    // s = "-++++-"
	    // fttttf
	    // f ff ttf
	    // f ff fff --> false
	    // f ff ttf -> true

	    public boolean search(boolean[] state) {
	        for (int i = 0; i < state.length - 1; i++) {
	        	//check if the current and the next is ++ 
	            if (state[i] && state[i + 1]) {
	            	//if so then change it to --
	                state[i] = false;
	                state[i + 1] = false;

	                // for the next one if the result would be ++
	                if (!search(state)) {
	                    state[i] = true;
	                    state[i + 1] = true;
	                    return true;
	                } else {
	                    state[i] = true;
	                    state[i + 1] = true;
	                }
	            }
	        }
	        return false;
	    }

}
