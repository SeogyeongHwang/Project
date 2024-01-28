# Box plot

### What is box plot?

Box plot is a type of data visualization that allows you to easily compare different data while simultaneously showing the distribution and outliers of the data. Instead of using the raw data as it is, the data is processed and visualized with a statistical concept called the ‘Five-number Summary’.    

Five-number summary is a method of presenting data with five statistics: the minimum and maximum values, and the first quartile(Q1), the second quartile(Q2, median), and the third quartile(Q3).

![Alt text](https://github.com/SeogyeongHwang/Project/blob/950cf78eaae2f3d1b5dc3c7382b6f4fc5b121c37/Data_Analysis/basic_analysis/box_plot.jpg)

### Why do people use box plot?

The average or standard deviation commonly used to produce statistics is more likely to convey distorted meaning if there is an outlier in the data, so it is useful to use the box plot to easily determine how many outliers are included and also easy to compare between datasets.

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
