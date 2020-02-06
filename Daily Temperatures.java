//Daily Temperatures
public class Solution {
    /**
    *输入:  temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
	*输出:  [1, 1, 4, 2, 1, 1, 0, 0]
	*解释:
	*找到每个数字后面第一个大于自己的数字，输出两者的距离。
     * @param temperatures: a list of daily temperatures
     * @return: a list of how many days you would have to wait until a warmer temperature
     */
	//利用stack记录温度对应的index，如果当前温度值大于栈顶index对应的温度值，就pop
	//Time: O(n)
	//Space: O(n)
	//stack.peek()
	//stack.pop()
	//stack.push(i)
    public int[] dailyTemperatures(int[] temperatures) {
        // Write your code here
        if (temperatures == null || temperatures.length == 0) {
            return new int[0];
        }

        int[] result = new int[temperatures.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < temperatures.length; i++) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
            	//check if the current peek one on the stack is smaller than the index, then
            	//update the current stack peek
                result[stack.peek()] = i - stack.peek();//index distance which means is the distance
                stack.pop();
            }
            stack.push(i); // first time: add 73
        }
        return result;
    }
}