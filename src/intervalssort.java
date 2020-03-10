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


253. meeting rooms

    public int minMeetingRooms(int[][] intervals) {
        // Check for the base case. If there are no intervals, return 0
        if (intervals.length == 0) {
          return 0;
        }
        //min heap
        PriorityQueue<int[]> heap = new PriorityQueue<int[]>((o1,o2) -> o1[1]- o2[1]);
        //sort the array by its start time
        Arrays.sort(
            intervals,
            new Comparator<int[]>() {
                public int compare(int[]a, int[]b){
                   return a[0] - b[0];
                }
            }       
        );
        // start with the first meeting, put it to a meeting room
        heap.offer(intervals[0]);
    
        for (int i = 1; i < intervals.length; i++) {
            // get the meeting room that finishes earliest
            int[] interval = heap.poll();

            if (intervals[i][0] >= interval[1]) {
                // if the current meeting starts right after 
                // there's no need for a new room, merge the interval
                interval[1]= intervals[i][1];
            } else {
                // otherwise, this meeting needs a new room
                heap.offer(intervals[i]);
            }

            // don't forget to put the meeting room back
            heap.offer(interval);
        }

        return heap.size();
    }

