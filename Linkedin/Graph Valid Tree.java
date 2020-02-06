//Graph Valid Tree
/* Approach 1: Union Find, to see if all the edges are connected in a single group
* time complexity: O(n), space complexity: O(n)
*/
public class Solution {
     /**
     * @param n: An integer
     * @param edges: a list of undirected edges
     * @return: true if it's a valid tree, or false
     */
    
    public boolean validTree(int n, int[][] edges) {
        // write your code here
        if (n == 0 || edges.length != n - 1) return false;
        int[] parents = new int[n];
        for (int i = 0; i < n; ++i) {
            parents[i] = i;
        }
        int root_a, root_b;
        for (int[] edge : edges) {
            root_a = find(parents, edge[0]);
            root_b = find(parents, edge[1]);
            if (root_a == root_b) return false;
            parents[root_b] = root_a;
        }
        return true;
    }
    
    public int find(int[] parents, int target) {
        if (parents[target] == target) return target;
        return parents[target] = find(parents, parents[target]);
    }

}