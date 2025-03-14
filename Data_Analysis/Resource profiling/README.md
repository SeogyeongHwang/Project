# Resource Profiling in Linux

### Introduction

There is a tool that measures the current usage (% throughput) of resources (CPU, Memory, Disk, Network) on Linux, which measures the usage of each resource when running a program and we will use Python to represent it as a graph.

### Step 1 - Resource Profiling

* Install sysstat, which Linux provides by default, on Ubuntu.
* After installation, monitor each resource using the following commands and save the results to a file.
  - sar -n DEV 1 > filename.txt (network)
  - sar -r 1 > filename.txt (Memory)
  - sar -d -p 1 > filename.txt (disk.dat)
  - sar -u -P ALL 1 > filename.txt (CPU)
* After executing each command, when you open the file, it is stored as a text file like the one in the folder 'step1'.
* If the time information is not output properly, add LC_TIME="en_US.UTF-8" to the ~/.bashrc file.

### Step 2 - Test 

1. Run the sleep command in the background for 20 seconds, such as "sleep 20 &", and then measure the CPU resources.
2. Create and run an infinite loop program below, then measure the same CPU resources.
   ```c
   int main(void) {
     for (;;)
       ;
   }
   ```
   
3. Similarly, write and run a program that invokes the getppid() function indefinitely, and measure CPU resources.   
   ```c
   int main(void) {
     for (;;)
       getppid();
   }
   ```
   
We can check the cpu mesurement results after executing cases 1, 2, and 3 in the folder 'step 2'. Let's compare and analyze how cpu usage varies in the above three cases.

|              |    infinite loop       | getppid() infinite call                      | sleep 20  |
|--------------|------------------------|----------------------------------------------|-----------|
| %user        | High (using 1 cpu)     | High (using 1 cpu)                           | Low       |
| %system      | Low                    | Little bit high because of using system call | Low       |
| %idle        | cpu in use: low        | cpu in use: low                              | High      |
| Resource     | Limited to other tasks | Limited to other tasks                       | Available |
| Availability | Limited to other tasks | Limited to other tasks                       | Available |
   
The infinite loop and the getppid() infinite call code all use one cpu, and either cpu 0 or 1 shows a high %user vaLue. However, the %system value was higher because the getppid() keeps system calling.
In contrast to the two infinite loop codes, when the sleep command is executed, the %user value is very low and resource availability is high because there is no cpu in use.

### Step 3 - Application Program

* Use wc (word count) to find out the number of words for large file.
* I used CAvideo.csv dataset from <https://www.kaggle.com/datasnaek/youtube-new>.
* Once the file is ready, run the following command.
  + wc textfilename.txt (filename)
   
1. When executing the wc program, the resource measurement code written in step 1 is executed simultaneously. All resource measurement result values are stored in a separate directory.
2. Run the web browser and play the video on YouTube. At the same time, execute the resource measurement code written in Step 1.
   
We can check the results of resource usage(CPU, Memory, Disk, Network) for three cases above in the folder 'step 3'. Let's analyze the results between three cases.

Compared to the execution of the wc command, all resources are used much more when playing YouTube videos. In particular, there was a big difference in CPU, memory, and network activities.   
While there is no network activity when executing the wc command, the network is used when playing Youtube videos.   
Youtube video playback measured 30% higher in average memory usage than when using the wc command.
There is some disk activity when playing YouTube videos, and there is no disk activity at all when using wc command, so the value is measured as 0. Most of the wc commands have very low CPU usage and are in an idle state, while YouTube has a remarkably high CPU usage rate and the value of %system is particularly high.
   
### Step 4 - Graph the storage result of Step 3 

+ The saved results are opened using Python, read by line, and then a graph is generated using the Matlab library.
+ Each file represent as one graph, and the data of interest for each graph are as follows. (The y-axis is the unit below.)
  - CPU : in 'all' low, (100-idle) (unit: %)
  - Memory : %memused (unit: %)
  - Disk : %util (unit: %)
  - Network : rxkB/s, txkB/s (unit: kB/s)
+ The X-axis becomes the time of each data. Since data is generated every second, the start time of the x-axis is 0 seconds to express the time elapsed.
   
#### Method

      1) network_plot : 
      2) mem_data : 
      3) disk_data : 
      4) cpu_data : 
      5) data_plot :

<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/ee138d53ac29d2f667f1f551a96a90f411899e9c/Data_Analysis/Resource%20profiling/Plots/Plot_networkWC.jpg" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/ee138d53ac29d2f667f1f551a96a90f411899e9c/Data_Analysis/Resource%20profiling/Plots/Plot_networkYoutube.jpg" width="49%" height="49%">
   </p>

   

