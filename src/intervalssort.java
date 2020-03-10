56. Merge Intervals
class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals.length <= 1)
			return intervals;

		// Sort by ascending starting point
		Arrays.sort(intervals, (i1, i2) -> Integer.compare(i1[0], i2[0]));
        
        //create a array of list and add the first value inside
        List<int[]> result = new ArrayList<int[]>();
        int[] newi = intervals[0];
        result.add(newi);
        
        for (int[] i : intervals){
            //overlap
            if (i[0] <= newi[1]) {
                newi[1] = Math.max(newi[1], i[1]);
            } else {
                newi = i;
                result.add(newi);
            }
        }
        //toarray
        return result.toArray(new int[result.size()][]);

    }
}
