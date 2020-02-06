//log sort
/*
给定一个字符串列表 logs, 其中每个元素代表一行日志. 每行日志信息由第一个空格分隔成两部分, 前面是这一行日志的 ID, 后面是日志的内容(日志的内容中可能包含空格). 一条日志的内容要么全部由字母和空格组成, 要么全部由数字和空格组成.

现在你需要按照如下规则对列表中的所有日志进行排序:

内容为字母的日志应该排在内容为数字的日志之前
内容为字母的日志, 按照内容的字典序排序, 当内容相同时则按照 ID 的字典序
内容为数字的日志, 按照输入的顺序排序
*/
// 用index of 得到 content 的部分; 用substring
//if it is a digit between 0-9, digits are the last
// 为了减少空间，使用 ans[cnt--]， 就是从后面往前面遍历
// 再排序
//log sort
/*
给定一个字符串列表 logs, 其中每个元素代表一行日志. 每行日志信息由第一个空格分隔成两部分, 前面是这一行日志的 ID, 后面是日志的内容(日志的内容中可能包含空格). 一条日志的内容要么全部由字母和空格组成, 要么全部由数字和空格组成.

现在你需要按照如下规则对列表中的所有日志进行排序:

内容为字母的日志应该排在内容为数字的日志之前
内容为字母的日志, 按照内容的字典序排序, 当内容相同时则按照 ID 的字典序
内容为数字的日志, 按照输入的顺序排序
*/
// 用index of 得到 content 的部分; 用substring
//if it is a digit between 0-9, digits are the last
// 为了减少空间，使用 ans[cnt--]， 就是从后面往前面遍历
// 再排序
public class Solution {
    /**
     * @param logs: the logs
     * @return: the log after sorting
     */
    public String[] logSort(String[] logs) {
        // Write your code here
    	List<String> list = new ArrayList<String>();
    	String[] ans = new String[logs.length];
    	int count = 0;

    	for (int i = logs.length -1; i >= 0; i--) {
    		int ind1 = logs[i].indexOf(' ');
    		String content = logs[i].substring(ind1 + 1);
    		if (content.charAt(0) >= '0' && content.charAt(0) <= '9') {
    			count++;
    			ans[logs.length - count] = logs[i];
    		} else {
    			list.add(logs[i]);
    		}
    	}

    	Collections.sort(list, new MyComparator());
    
    	count = 0;

    	for (String s: list){
    		ans[count++] = s;
    	}
    	return ans;


    }


    class MyComparator implements Comparator{
    	@Override //upper case o
    	public int compare(Object o1, Object o2) {
    		String s1 = (String) o1;
    		String s2 = (String) o2;

    		int ind1 = s1.indexOf(' ');
    		int ind2 = s2.indexOf(' ');

    		String head1 = s1.substring(0, ind1);
    		String head2 = s2.substring(0, ind2);
    		String cont1 = s1.substring(ind1);
    		String cont2 = s2.substring(ind2);

    		if(cont1.equals(cont2)) {
    			return head1.compareTo(head2);
    			//this.charAt(k)-anotherString.charAt(k)
    		} else {
    			return cont1.compareTo(cont2);
    		}


    	}

    } 
}