// IMPORT LIBRARY PACKAGES NEEDED BY YOUR PROGRAM
// SOME CLASSES WITHIN A PACKAGE MAY BE RESTRICTED
// DEFINE ANY CLASS AND METHOD NEEDED
import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Collections;
import java.util.HashSet;
import java.util.*;
import java.util.Arrays;
// CLASS BEGINS, THIS CLASS IS REQUIRED
class Solution
{        
    // METHOD SIGNATURE BEGINS, THIS METHOD IS REQUIRED
    public ArrayList<String> popularNFeatures(int numFeatures, 
                                             int topFeatures, 
                                             List<String> possibleFeatures, 
                                             int numFeatureRequests, 
                                             List<String> featureRequests)
    {
        // WRITE YOUR CODE HERE
        ArrayList<String> res = new ArrayList<String>();
        HashMap<String, int[]> map = new HashMap<String, int[]>();
        
        if (numFeatures == 0 || numFeatureRequests == 0){
            return res;
        }
      
        for (String feature : possibleFeatures ) {
            int[] value = new int[]{0,0};
            map.put(feature, value);
        }
        
        for (String feature: featureRequests) {
            String lowerfeature = feature.toLowerCase();
            String[] words = lowerfeature.split("\\W+");
            HashSet<String> used = new HashSet<String>();
            for (String word : words) {
                if (!map.containsKey(word)){
                    continue;
                }
            
                int[] nums = map.get(word);
                nums[0]++;
                if(!used.contains(word)) {
                    nums[1]++;
            }
            used.add(word);
            }
        }
        
        PriorityQueue<String> pq = new PriorityQueue<String>((a,b) ->{
           if (map.get(a)[0] != map.get(b)[0]){
               return map.get(a)[0] - map.get(b)[0];
           }
           if (map.get(a)[1] != map.get(b)[1]){
               return map.get(a)[1] - map.get(b)[1];
           }
           return b.compareTo(a);
            
        });
        
        if (topFeatures > numFeatures) {
            for (String feature : map.keySet()) {
                if(map.get(feature)[0] > 0){
                    pq.add(feature);
                }
            }
        }else {
            for (String feature: possibleFeatures) {
                pq.add(feature);
                if(pq.size() > topFeatures) {
                    pq.poll();
                }
            }
        }
        
        while (!pq.isEmpty()) {
            res.add(pq.poll());
        }
        
        Collections.reverse(res);
        return res;
       
        
    }
    // METHOD SIGNATURE ENDS
}