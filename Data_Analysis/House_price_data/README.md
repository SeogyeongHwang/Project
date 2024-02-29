# Analysis of Boston Housing Price data

## 1. Introduction

In this report, we will see how each column affects the Boston housing price based on the 'MEDV(median value of owner-occupied homes in $1000's)' column and see what affects the housing price the most.

## 2. Method

Seven methods were used in this code to download the data.

      1) pd.read_csv() : To read csv file into dataframe. The default delimiter is ',' but can be used interchangeably.
      2) to_list() : To change the values of a particular column into a list.
      3) df[::2] or df[1::2] : To extract even or odd rows from dataframe.
      4) reset_index(drop=True) : To reset index.
      5) pd.concat([df1, df2], axis=1) : To concat two dataframe. Depending on whether axis is 0 or 1, you can choose which direction to combine the data between rows and columns.
      6) dropna(axis=1) : To remove columns that have NaN values
      7) to_csv() : To make dataframe as a csv file

## 3. Result

![Alt_text](https://github.com/SeogyeongHwang/Project/blob/5954d8907f8abf7766e871f20e0436d6cb2d7231/Data_Analysis/House_price_data/Plots/CRIM.jpg)
#### < Correlation with per capita crime rate >
+ The closer the crime rate per capita is to 0, the higher the house price, and the higher the crime rate, the lower the house price is.

![Alt_text](https://github.com/SeogyeongHwang/Project/blob/5954d8907f8abf7766e871f20e0436d6cb2d7231/Data_Analysis/House_price_data/Plots/ZN.jpg)

#### < Correlation with proportion of residential land in a specific section >
+ This graph tend to go up.
+ The higher the residential land ratio, the higher the house price is formed.

![Alt_text](https://github.com/SeogyeongHwang/Project/blob/5954d8907f8abf7766e871f20e0436d6cb2d7231/Data_Analysis/House_price_data/Plots/INDUS.jpg)

#### < Correlation with proportion of non-retail business acres >
+ Not only the slope of the graph, but also the distribution of the data shows that the y value decreases when the x value is close to 0, and the y value decreases when the x value increases.

![Alt_text](https://github.com/SeogyeongHwang/Project/blob/5954d8907f8abf7766e871f20e0436d6cb2d7231/Data_Analysis/House_price_data/Plots/NOX.jpg)

#### < Correlation with nitric oxides concentration >
+ It also shows a downard trend.
+ As the x value exceeds 0.7, the overall distribution of the y value goes downwards.

![Alt_text](https://github.com/SeogyeongHwang/Project/blob/5954d8907f8abf7766e871f20e0436d6cb2d7231/Data_Analysis/House_price_data/Plots/RM.jpg)

#### < Correlation with average number of rooms per dwelling >
+ It is increasing rapidly than other columns.
+ There are many similar y-values between 5 and 7 with x values.

![Alt_text](https://github.com/SeogyeongHwang/Project/blob/5954d8907f8abf7766e871f20e0436d6cb2d7231/Data_Analysis/House_price_data/Plots/AGE.jpg)


#### < Correlation with proportion of units built before 1940 >
+ As the x value increases, the y value tends to decrease.
+ When x value is 100, it has both minimum and maximum values.

![Alt_text](https://github.com/SeogyeongHwang/Project/blob/5954d8907f8abf7766e871f20e0436d6cb2d7231/Data_Analysis/House_price_data/Plots/DIS.jpg)

#### < Correlation with distance to five Boston employment centres >
+ 
