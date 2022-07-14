import java.util.Vector;

public class parallelTest {
    int executions;
    GreyImage image;
    Vector<Integer> num_threads;

    parallelTest(GreyImage image,Vector<Integer> num_threads, int executions){
        this.image = image;
        this.num_threads = num_threads;
        this.executions = executions;

    }

    void run_display() {
        for (int threads : num_threads) {
            double min_exec_time = 10000, max_exec_time = 0, mean_exec_time = 0;
            for (int execution = 0; execution < executions; execution++) {
                long start = System.currentTimeMillis();
                try{
                    image.calculateParallelIntegralImage(threads);
                }
                catch(InterruptedException e){
                    System.out.println("Error!");
                }
                long diff = System.currentTimeMillis() - start;
                double exec_time = diff / 1000F; //convert time in seconds
                mean_exec_time += exec_time;

                //Check min
                if (exec_time < min_exec_time)
                    min_exec_time = exec_time;

                //Check max
                if (exec_time > max_exec_time)
                    max_exec_time = exec_time;
            }

            mean_exec_time /= executions;

            System.out.println("Parallel test with: "+threads+" threads.");
            System.out.println("Parallel Minimum Execution Time: " + min_exec_time + " seconds!");
            System.out.println("Parallel Maximum Execution Time: " + max_exec_time + " seconds!");
            System.out.println("Parallel Mean Execution Time: " + mean_exec_time + " seconds!");
        }
    }
}
