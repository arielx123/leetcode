public class Solution{
	int[] dx = new int[]{0, 0, 0, 1, 1, 1, -1, -1, -1};
	int[ dy = new int[]{0, -1, 1, 0, 1, -1, 0, 1, -1};

	public char[][] updateBoard(char[][] board, int[] click) {
        // Write your code here
        search(board, click);
        return board;
    }
    private void search(char[][] board, int[] click){
    	int x = click[0], y = click[1];
    	int n = board.length;
    	int m = board[0].length;

    	if(board[x][y] == 'M'){
            board[x][y] = 'X';
        }else if(board[x][y] == 'E'){
        	int mineCount = 0;
        	for(int i = 1; i<9; i++){
                int nx = x + dx[i];
                int ny = y + dy[i];
                if(nx >= 0 && nx < n && ny >= 0 && ny < m && board[nx][ny] == 'M'){
                    mineCount++;
                }
            }

            if (mineCount == 0) {
            	 board[x][y] = 'B'; // update and search for the surrandings
            	 for(int i = 1; i<9; i++){
                    int nx = x + dx[i];
                    int ny = y + dy[i];
                    if(nx >= 0 && nx < n && ny >= 0 && ny < m){
                       search(board, new int[]{nx, ny});
                    }
                }
            } else {
            	board[x][y] = mineCount;
            }
        }
        return;
    }
}