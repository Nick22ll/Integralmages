public class SumRowIntegralImage extends Thread {
    int width, row_start, row_stop;
    int[][] integral_matrix;

    public SumRowIntegralImage(int[][] integral_matrix, int row_start, int row_stop, int width){
        this.integral_matrix = integral_matrix;
        this.row_start = row_start;
        this.row_stop = row_stop;
        this.width = width;
    }

    public void run() {
        for(int row = row_start; row < row_stop; row++)
            for (int col = 1; col < width; col++)
                integral_matrix[row][col] += integral_matrix[row][col-1];
    }
}
