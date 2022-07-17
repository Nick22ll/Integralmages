import java.util.ArrayList;
import java.util.Arrays;
import java.util.Vector;

public class Main {

    public static void main(String[] args) throws InterruptedException {

        String[] image_names = {"480", "720", "1080", "1440", "2160"};
        
        for (String name : image_names) {
            GreyImage image = new GreyImage("U:/Magistrale/Parallel Computing/IntegralImagesCUDA/images/" + name + ".jpg");

            System.out.println("STARTING SEQUENTIAL TEST WITH IMAGE: " + name + ".jpg! \n");
            sequentialTest seqTest = new sequentialTest(image, 100);
            seqTest.run();
            double mean_sequential = seqTest.mean_exec_time;
            seqTest.displayResults();

            int[] num_threads = {2,4,8,12,16, 20, 24};

            System.out.println("STARTING PARALLEL TEST WITH IMAGE: " + name + ".jpg! \n");
            parallelTest parTest = new parallelTest(image, num_threads, 100);
            Vector<Double> mean_values = parTest.run_display();
            for (int i = 0; i < num_threads.length; i++)
                System.out.println("Speedup with " + num_threads[i] + " threads is: " + mean_sequential / mean_values.get(i));
        }
    }
}


