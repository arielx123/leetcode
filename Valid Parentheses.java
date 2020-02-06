//20 Valid Parentheses
public class Solution {
    /**
     * @param s: A string
     * @return: whether the string is a valid parentheses
     */
    //use hashmap and stack; stack: push, pop, peek,
    // LIFO (Last In First Out) principle
    public boolean isValidParentheses(String s) {
        // write your code here
        HashMap<Character, Character> pairs = new HashMap<Character, Character>();
        //HashMap, h, m capitalize
        pairs.put('{', '}');
        pairs.put('[', ']');
        pairs.put('(', ')');

        Stack<Character> stack = new Stack<Character>();
        for (int i = 0; i < s.length(); i++) {
        	Character current = s.charAt(i);
        	if (pairs.containsKey(current)) {
        		stack.push(current);
        	}

        	if(pairs.values().contains(current)){
        		if(!stack.empty() && pairs.get(stack.peek()) == current) {
        			stack.pop();
        		} else {
        			return false;
        		}
        	}
        }
        return stack.empty();


    }
}