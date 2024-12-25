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
* After executing each command, when you open the file, it is stored as a text file like the one in the folder step1.
* If the time information is not output properly, add
<span style="color: green"> LC_TIME="en_US.UTF-8"</span>
to the ~/.bashrc file.


### Method

- .min(), .max() : To get the minimum and the maximum values between data
- .plot(kind:'box') : plot data as a box plot

### Normalization Box Plot

Change the range of the characteristic value to [0, 1] using this formula $X'=\frac{X - X_min}{X_max - X_min}$.   
The largest value converted to 1, and the smallest value to 0, allowing the characteristics to be placed in an equal position.
You can see the result below.

![Alt_text](https://github.com/SeogyeongHwang/Project/blob/94b095b9f26ce51ad775b9eeda907ebea02cca64/Data_Analysis/basic_analysis/Plots/Data_normalization_plot.jpg)

   
   
# Correlation

Correlation refers to the statistical relationship between the two entities. It measures the extent to which two variables are linearly related. We use it to determine whether there is a correlation between variables, and to determine the degree of correlation. We can also check increasing direction.   
Correlation coefficients only exist between -1 and +1.
+ a negative correlation   
+ No correlation   
+ positive correlation   

To visualize this, we use 'heatmap' plot.

![Alt_text](https://github.com/SeogyeongHwang/Project/blob/e08fe4e3c7f5b356ed51d7908733bde75987d660/Data_Analysis/basic_analysis/Plots/Data_heatmap_plot.jpg)

### Method

- plt.subplots(figsize=()) : to determine the plot size
- sns.heatmap(data, annot=True, cmap='RdYlBu_r') : to plot as a heatmap. With annot, we can show the data value in each cell. With cmap, we can mapping data values to color space.
- .corr() : to compute correlation of columns in dataframe



# Linear Regression

Regression is a way to explain the relationship between a dependent variable(Y) and one or more explanatory variables(X). Hence, linear regression is a simple approach to predict based on a data that follows a linear trend and find the 'best fit line' that gives good value nearest to that point like in above plots.   
   
Scatterplot can be used to explore potential relationships between pairs of variables and correlation provides a measure of the linear association between pairs of variables, but it doesn't tell us about more complex relationships like if the relationship. However, through regression, We can predict the values of a response variable based on the values of the important predictors. Or, we can use regression models for optimization, to determine settings of factors to optimize a response.



# Finding missing values

Missing values can bias the results of your machine learning models and can result in decreased accuracy. That is why we must handle these values in the correct way, so that the data is imputed correctly.
### How to check missing values
In pandas, missing data is represented by two values: None or NaN   
There are several useful functions for detecting, removing, and replacing null values in Pandas.
+ isnull() : To check null values
+ notnull() : To check non-null values
+ dropna() : To drop rows / columns with null values
+ fillna() : To replace NaN values with some value by users
+ replace() : To replace data values to other values
+ interpolate() : To fill NaN values in dataframe or series
