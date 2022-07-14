public class PartialIntegralImage extends Thread {
    int width, height, start_x, start_y;
    int[][] integral_matrix;
    GreyImage original_image;

    public PartialIntegralImage(GreyImage originalImage, int start_x, int start_y, int end_x, int end_y){
        original_image = originalImage;
        this.start_x = start_x;
        this.start_y = start_y;
        width = end_x - start_x;
        height = end_y - start_y;
        integral_matrix = new int[height][width];
    }


    public void run() {
        //Fill the first Cell
        integral_matrix[0][0] = original_image.getPixel(start_x, start_y);

        //Fill the first column
        for(int i = 1; i < height ; i++)
            integral_matrix[i][0] = integral_matrix[i-1][0] + original_image.getPixel( start_x, i + start_y) ;

        //Fill the first row
        for(int j = 1; j < width ; j++)
            integral_matrix[0][j] = integral_matrix[0][j-1] + original_image.getPixel(j + start_x, start_y);

        for(int i=1; i < height; i++){
            for(int j=1; j < width ; j++){
                integral_matrix[i][j] = original_image.getPixel(j+start_x,i + start_y) + integral_matrix[i][j-1] + integral_matrix[i-1][j] - integral_matrix[i-1][j-1];
            }
        }
    }
}
