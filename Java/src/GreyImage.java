import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.WritableRaster;
import java.io.*;
import java.util.stream.IntStream;
import javax.imageio.ImageIO;

public class GreyImage {
    BufferedImage image;
    int[][] image_matrix;
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

            image_matrix = new int[height][width];
            for(int i=0; i<height; i++)
                for(int j=0; j<width; j++)
                    image_matrix[i][j] = this.getPixel(j, i);

        } catch (Exception e) {
            System.out.println("File opening FAILED!");
        }
    }

    public GreyImage(GreyImage old_image) {
        ColorModel cm = old_image.image.getColorModel();
        boolean isAlphaPremultiplied = cm.isAlphaPremultiplied();
        WritableRaster raster = old_image.image.copyData(null);
        image = new BufferedImage(cm, raster, isAlphaPremultiplied, null);
    }

    public int getPixel(int x, int y) {
        return image.getRGB(x, y) & 0xFF;
    }

    public int[][] calculateIntegralImage() {
        int[][] integralImage = new int[height][width];

        //Fill the first Cell
        integralImage[0][0] = this.getPixel(0, 0);

        //Fill the first column
        for (int i = 1; i < height; i++)
            integralImage[i][0] = integralImage[i - 1][0] + this.getPixel(0, i);

        //Fill the first row
        for (int j = 1; j < width; j++)
            integralImage[0][j] = integralImage[0][j - 1] + this.getPixel(j, 0);

        for (int i = 1; i < height; i++)
            for (int j = 1; j < width; j++)
                integralImage[i][j] = this.getPixel(j, i) + integralImage[i][j - 1] + integralImage[i - 1][j] - integralImage[i - 1][j - 1];

        return integralImage;
    }
    public int[][] calculateParallelIntegralImage(int numThreads) throws InterruptedException {
        int[][] integral_image;
        int[][] transpose;
        int columns_per_thread = (int) Math.floor(width / numThreads);
        int rows_per_thread = (int) Math.floor(height / numThreads);

        transpose = transpose(this.image_matrix, height, width);
        SumRowIntegralImage[] threads = new SumRowIntegralImage[numThreads];

        for (int i = 0; i < numThreads - 1; i++)
            threads[i] = new SumRowIntegralImage(transpose, i * columns_per_thread, (i * columns_per_thread)+columns_per_thread, height);
        threads[numThreads-1] = new SumRowIntegralImage(transpose, (numThreads - 1) * columns_per_thread, width, height);

        for (SumRowIntegralImage thread : threads)
            thread.start();

        for (SumRowIntegralImage thread: threads)
            thread.join();

        integral_image = transpose(transpose, width, height);
        threads = new SumRowIntegralImage[numThreads];

        for (int i = 0; i < numThreads - 1; i++)
            threads[i] = new SumRowIntegralImage(integral_image, i * rows_per_thread, (i * rows_per_thread)+ rows_per_thread, width);
        threads[numThreads-1] = new SumRowIntegralImage(integral_image, (numThreads - 1) * rows_per_thread, height, width);

        for (SumRowIntegralImage thread : threads)
            thread.start();

        for (SumRowIntegralImage thread: threads)
            thread.join();

        return integral_image;
    }

    public int[][] transpose(int[][] matrix, int height, int width){
        int[][] transpose = new int[width][height];
        IntStream.range(0, width * height).parallel().forEach(i ->
        {
            int m = i / height;
            int n = i % height;
            transpose[m][n] = matrix[n][m];
        });
        return transpose;
    }
}

