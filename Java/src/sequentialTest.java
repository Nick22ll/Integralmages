
public class sequentialTest {
    int executions;
    GreyImage image;
    double min_exec_time= 10000 , max_exec_time = 0, mean_exec_time = 0;

    sequentialTest(GreyImage image, int executions){
        this.image = image;
        this.executions = executions;
    }

    void run(){
        for(int execution=0; execution<executions; execution++){
            long start = System.currentTimeMillis();
            image.calculateIntegralImage();
            long diff = System.currentTimeMillis() - start;
            double exec_time = diff/1000F; //convert time in seconds
            mean_exec_time += exec_time;

            //Check min
            if(exec_time < min_exec_time)
                min_exec_time = exec_time;

            //Check max
            if(exec_time > max_exec_time)
                max_exec_time = exec_time;
        }
        mean_exec_time /= executions;
    }

    void displayResults(){
        System.out.println("Sequential Minimum Execution Time: " + min_exec_time + " seconds!");
        System.out.println( "Sequential Maximum Execution Time: " + max_exec_time + " seconds!");
        System.out.println( "Sequential Mean Execution Time: " + mean_exec_time+ " seconds!");
    }

}
