//First Unique Character in a String
public class Solution {
    /**
     * @param str: str: the given string
     * @return: char: the first unique character in a given string
     */
    // int array, iterate through the string and count the char, 
    // then iterate it again to return the first 1
    // Time: O(n), space: O(1)
    public char firstUniqChar(String str) {
        // Write your code here
    	if (str == null) {
    		return '0';
    	}

    	int[] frequency = new int[256]; 

    	for (int i = 0; i < str.length(); i++) {
    		frequency[str.charAt(i)]++;
    	}

    	for (int i = 0; i< str.length(); i++) {
    		if (frequency[str.charAt(i)] == 1){
    			return str.charAt(i);
    		}
    	}

    	return '0';
    }
}