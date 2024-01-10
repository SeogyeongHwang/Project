# Box plot

### What is box plot?

Box plot is a type of data visualization that allows you to easily compare different data while simultaneously showing the distribution and outliers of the data. Instead of using the raw data as it is, the data is processed and visualized with a statistical concept called the ‘Five-number Summary’.    

Five-number summary is a method of presenting data with five statistics: the minimum and maximum values, and the first quartile(Q1), the second quartile(Q2, median), and the third quartile(Q3).

![Alt text](https://github.com/SeogyeongHwang/Project/blob/950cf78eaae2f3d1b5dc3c7382b6f4fc5b121c37/Data_Analysis/basic_analysis/box_plot.jpg)

### Why do people use box plot?

The average or standard deviation commonly used to produce statistics is more likely to convey distorted meaning if there is an outlier in the data, so it is useful to use the box plot to easily determine how many outliers are included and also easy to compare between datasets.

### Normalization Box Plot

Change the range of the characteristic value to [0, 1] using this formula $X'=\frac{X - X_min}{X_max - X_min}$.   
The largest value converted to 1, and the smallest value to 0, allowing the characteristics to be placed in an equal position.


## Method

- .min(), .max() : 
