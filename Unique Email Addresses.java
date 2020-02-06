public class Solution {
    /**
     * @param emails: 
     * @return: The number of the different email addresses
     */
    /*每封电子邮件都由一个本地名称和一个域名组成，以 @ 符号分隔。

	例如，在 alice@lintcode.com中， alice 是本地名称，而 lintcode.com 是域名。

		除了小写字母，这些电子邮件还可能包含 '.'' 或 '+'。

		如果在电子邮件地址的本地名称部分中的某些字符之间添加句点（'.'），则发往那里的邮件将会转发到本地名称中没有点的同一地址。例如，"alice.z@lintcode.com” 和 “alicez@lintcode.com” 会转发到同一电子邮件地址。 （请注意，此规则不适用于域名。）

		如果在本地名称中添加加号（'+'），则会忽略第一个加号后面的所有内容。这允许过滤某些电子邮件，例如 m.y+name@email.com 将转发到 my@email.com。 （同样，此规则不适用于域名。）
    */
    public int numUniqueEmails(String[] emails) {
        // write your code here
        Set<String> differentset = new HashSet<>();
        for(String email : emails) {
        	String[] parts = email.split("@"); //split the string by @, 123@456 --> 123, 456
        	String[] local = parts[0].split("\\+");// remember to add \\
        	differentset.add(local[0].replace(".", "") + "@" + parts[1]);//string.replace()
        }

        return differentset.size(); // hashset use size()
    }
    //-->O(n), n is the number of the emails
    //--> O(n)
}