public class PartialIntegralImage extends Thread {
    int width, height, start_x, start_y;
    int[][] integral_matrix;
    GreyImage original_image;

    public PartialIntegralImage(GreyImage originalImage, int start_x, int start_y, int end_x, int end_y, int[][] integral_matrix){
        original_image = originalImage;
        this.integral_matrix = integral_matrix;
        this.start_x = start_x;
        this.start_y = start_y;
        width = end_x - start_x;
        height = end_y - start_y;

    }

    public void run() {
        //Fill the first Cell
        integral_matrix[start_y][start_x] = original_image.getPixel(start_x, start_y);

        //Fill the first column
        for(int i = 1; i < height ; i++)
            integral_matrix[start_y + i][start_x] = integral_matrix[start_y+ (i-1)][start_x] + original_image.getPixel( start_x, i + start_y) ;

        //Fill the first row
        for(int j = 1; j < width ; j++)
            integral_matrix[start_y][start_x + j] = integral_matrix[start_y][start_x + (j-1)] + original_image.getPixel(j + start_x, start_y);

        for(int i=1; i < height; i++)
            for(int j=1; j < width ; j++)
                integral_matrix[start_y + i][start_x+ j] = original_image.getPixel(j+start_x,i + start_y) + integral_matrix[start_y + i][start_x + (j-1)] + integral_matrix[start_y + (i-1)][start_x + j] - integral_matrix[start_y + (i-1)][start_x + (j-1)];
    }
}
