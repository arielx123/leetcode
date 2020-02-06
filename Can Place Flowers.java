//650 Can Place Flowers
//给定一个花圃（用一个包含0和1的数组来表示，其中0代表空，1代表非空），
//和一个数字n，返回n朵新的花在这个花圃上以能否在不违反“无相邻花”的规则种植
//O(n) , O(1)
public class Solution {
    /**
     * @param flowerbed: an array
     * @param n: an Integer
     * @return: if n new flowers can be planted in it without violating the no-adjacent-flowers rule
     */
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        // Write your code here
        int count = 0 ; 

        for (int i = 0; i < flowerbed.length; i++) {
        	if (flowerbed[i] == 0 &&
        		(i == 0 || flowerbed[i-1] == 0) &&
        		// order is pretty important, have to put the check number first
        		(i == flowerbed.length -1 || flowerbed[i+1] == 0)) {
        		flowerbed[i] = 1; // remember to change it to 1
        		count++;
        	}
        }

        if (count >= n) { // >= not only just >
        	return true;
        }
        return false;
    }
}