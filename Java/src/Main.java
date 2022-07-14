import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;

public class Main {

    public static void main(String[] args) throws InterruptedException {
        GreyImage image = new GreyImage("images/big.jpg");

        sequentialTest seqTest = new sequentialTest(image, 100);
        seqTest.run();
        seqTest.displayResults();

        Integer[] data = {2,4,8,16,20,24};
        Vector<Integer> num_threads = new Vector<Integer>(Arrays.asList(data));
        parallelTest parTest = new parallelTest(image,num_threads, 100);
        parTest.run_display();


        /*int[][] original_image = new int[image.height][image.width];
        for(int i=0; i<image.height; i++)
            for(int j=0; j<image.width; j++)
                original_image[i][j] = image.getPixel(j,i);

        int[][] sequential_integral_image = image.calculateIntegralImage();
        int[][] parallel_integral_image = image.calculateParallelIntegralImage(2);

        for(int i=0; i<image.height; i++)
            for(int j=0; j<image.width; j++)
                if(sequential_integral_image[i][j] != parallel_integral_image[i][j])
                    System.out.println("Error at cell: "+i+","+j);

*/


    }

}


