//paint house
/* Requirements: no adjacent two house can paint the same color
 * differnet than paint fence 
 * The cost of painting each house with a certain color is different.
 * Approach: Dynamic programming approach: dp[i][j]: minimum costs to paint i houses 
 		with last house of j color 
 * dp[i][j] = 1. when j = 0: dp[i][0] = costs[i][0] + Math.min(dp[i-1][1], dp[i-1][2])
            = 2. when j = 1: dp[i][1] = costs[i][1] + Math.min(dp[i-1][0], dp[i-1][2])
            = 3. when j = 2: dp[i][2] = costs[i][2] + Math.min(dp[i-1][0], dp[i-1][1])
 * Goal: Math.min(dp[n][0], Math.min(dp[n][1], dp[n][2]))
 * Trick: instead of creating a dp table, we can utilized the existing costs table */
 //Input: [[17,2,17],[16,16,5],[14,3,19]]

public class Solution {
    /**
     * @param costs: n x 3 cost matrix
     * @return: An integer, the minimum cost to paint all houses
     */
    public int minCost(int[][] costs) {
        // write your code here
        if (costs.length == 0 || costs[0].length == 0) {
        	return 0;
        }

        
        //dp[i][j]: minimum costs to paint i houses 
        //costs[i-1][1] 上一步的cost

        for (int i =1 ; i < costs.length; i++) {
        	costs[i][0] += Math.min(costs[i-1][1], costs[i-1][2]);
        	costs[i][1] += Math.min(costs[i-1][0], costs[i-1][2]);	
        	costs[i][2] += Math.min(costs[i-1][0], costs[i-1][1]);	
        }

        /* red, blue and green at the end represents the min sum for painting all the house
         * with the last house have different color */
        
        return Math.min(costs[costs.length-1][0], Math.min(costs[costs.length-1][1], costs[costs.length-1][2]));

    }
}