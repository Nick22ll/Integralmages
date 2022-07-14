import java.awt.*;
import java.awt.image.BufferedImage;

import java.io.*;
import java.util.Vector;

import javax.imageio.ImageIO;


public class GreyImage {
    BufferedImage image;

    int width;
    int height;

    public GreyImage(String image_path) {
        try {
            File input = new File(image_path);
            BufferedImage rgb_image = ImageIO.read(input);
            width = rgb_image.getWidth();
            height = rgb_image.getHeight();

            image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
            Graphics g = image.getGraphics();
            g.drawImage(rgb_image, 0, 0, null);

            g.dispose();

        } catch (Exception e) {
            System.out.println("File opening FAILED!");
        }
    }

    public int getPixel(int x, int y){
        return image.getRGB(x, y) & 0xFF;
    }

    public int[][] calculateIntegralImage(){
        int[][] integralImage = new int[height][width];

        //Fill the first Cell
        integralImage[0][0] = this.getPixel(0,0);

        //Fill the first column
        for(int i = 1; i < height ; i++)
            integralImage[i][0] = integralImage[i - 1][0] + this.getPixel(0,i);

        //Fill the first row
        for(int j = 1; j < width ; j++)
            integralImage[0][j] = integralImage[0][j-1] + this.getPixel(j, 0);

        for(int i=1; i < height; i++){
            for(int j=1; j < width ; j++){
                integralImage[i][j] = this.getPixel(j,i) + integralImage[i][j-1] + integralImage[i-1][j] - integralImage[i-1][j-1];
            }
        }

        return integralImage;
    }

    public int[][] calculateParallelIntegralImage(int numThreads) throws InterruptedException{
        int[][] integralImage = new int[height][width];
        Vector<PartialIntegralImage> threads = new Vector<PartialIntegralImage>();
        int columns_per_thread = (int) Math.floor(width/numThreads);

        for(int i=0;i<numThreads-1; i++)
            threads.add(new PartialIntegralImage(this, i*columns_per_thread, 0, (i*columns_per_thread) + columns_per_thread, height));
        threads.add(new PartialIntegralImage(this, (numThreads-1)*columns_per_thread, 0, width, height));

        for(PartialIntegralImage thread : threads)
            thread.start();

        //wait for first thread to stop and then simply copy the partial integral image obtained
        threads.get(0).join();
        for(int row = 0; row<height; row++)
            System.arraycopy(threads.get(0).integral_matrix[row], 0, integralImage[row], 0, threads.get(0).width);

        //wait for the next thread and then properly concatenate the partial integral image obtained with the previous one
        int last_idx = threads.get(0).width;
        for(int i=1; i<numThreads; i++) {
            threads.get(i).join();
            for(int row = 0; row<height; row++){
                System.arraycopy(threads.get(i).integral_matrix[row], 0, integralImage[row], last_idx, threads.get(i).width);
                for(int col = 0; col<threads.get(i).width; col++ )
                    integralImage[row][last_idx+col] += integralImage[row][last_idx-1];
            }
            last_idx += threads.get(i).width;
        }


        return integralImage;

    }

}


