# Analysis of Boston Housing Price data

## 1. Introduction

In this project, we will develop and evaluate the performance and prediction of data with multiple plots. The dataset contains these following items: 'CRIM: per capita crime rate by town', 'ZN: proportion of residential land zoned for lots over 25,000 sq.ft.', 'INDUS: proportion of non-retail business acres per town', 'CHAS: Charles River dummy variable(= 1 if tract bounds river; 0 otherwise)', 'NOX: nitric oxides concentration (parts per 10 million)', 'RM: average number of rooms per dwelling', 'AGE: proportion of owner-occupied units built prior to 1940', 'DIS: weighted distances to five Boston employment centres', 'RAD: index of accessibility to radial highways', 'TAX: full-value property-tax rate per $10,000', 'PTRATIO: pupil-teacher ratio by town', 'B: 1000$$(Bk - 0.63)^2$$ where Bk is the proportion of blacks by town', 'LSTAT: % lower status of the population'. In this report, we will see how each column affects the Boston housing price based on the 'MEDV(median value of owner-occupied homes in $1000's)' column and see what affects the housing price the most.

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

Before plotting the data, we can see that there was no missing value in the dataframe through 'Finding_missing_values' function.

![Alt_text](https://github.com/SeogyeongHwang/Project/blob/d111795bfb28579892a0ac0df41e8bbcfc104b6c/Data_Analysis/House_price_data/Plots/Data_normalization_plot.jpg)
#### < Normalization data Box Plot >
Since the data is spread differently, it is difficult to see data that are not normalized together. We can see the distribution of each variable from this image.
+ 'CRIM' has a very low distribution, so it can be seen that most of them are close to zero and there are outliers up there.
+ In contrast, 'INDUS', 'NOX', 'AGE', 'RAD', 'TAX' are widely distributed and have no outliers.
+ Through this graph, 'CHAS' column is hard to get information because it shows that this data composed of only two different values.

#### < Correlation between two data using scatter plot >
<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/a9f503a28c40ca765ae6497001d2a39a74609566/Data_Analysis/House_price_data/Plots/B.jpg" width="33%" height="33%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/5954d8907f8abf7766e871f20e0436d6cb2d7231/Data_Analysis/House_price_data/Plots/DIS.jpg" width="33%" height="33%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/5954d8907f8abf7766e871f20e0436d6cb2d7231/Data_Analysis/House_price_data/Plots/RM.jpg" width="33%" height="33%">
   </p>

+ All three graphs have a positive slope.
+ When proportion of population that is black is high, the values are variously distributed.
+ We can see if the distance from the employment centres is shorter, the price of the house could be very high and very low. In addition, very high priced houses are distributed where the value of 'DIS' is under 6.
+ THe more rooms there are, the price also go up. But when the number of rooms are around 6, have many median values.

<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/5954d8907f8abf7766e871f20e0436d6cb2d7231/Data_Analysis/House_price_data/Plots/AGE.jpg" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/cbf880b5416d0da660f9335514ae69f181a22601/Data_Analysis/House_price_data/Plots/LSTAT.jpg" width="49%" height="49%">
   </p>

+ These are graphs have a negative slope.
+ It can be seen that expensive houses are distributed from new houses to very old houses, but the correlation does not seem to be significant until 50 years, while later houses are relatively inexpensive.
+ Proportion of population that is lower status is less than 10, it has the highest house price.

<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/b82419361da60d6379e486a4e7d59090137ffe72/Data_Analysis/House_price_data/Plots/bivariate_boxplot_CHAS.jpg" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/b82419361da60d6379e486a4e7d59090137ffe72/Data_Analysis/House_price_data/Plots/bivariate_boxplot_RAD.jpg" width="49%" height="49%">
   </p>

#### < Bivariate Box Plot >
These columns are not invisible to obtain meaningful information from scatter plot.
+ The column 'CHAS' value is 1 if tract bounds river, otherwise it has 0. Overall, the distribution of house prices is higher when the 'CHAS' value is 1 than when the value is 0. 
+ When the index of accessibility to radial highways has highest value, it generally shows low house price.
   
## 4. Conclusion

From the above results, we can draw the following conclusions. In the beginning, I expected that these following items would affect the house price a lot: crime rate, room numbers, how old the house is, distance from the employment centres. I cannot say that these conclusions are exact statistics, because There may be various reasons why the x value is low or high in plotting the values. One-dimensional reasoning can be made that if a house is old, the price will fall, but for example, 50 years ago, preference may decrease because of the use of harmful ingredients when building a house, or because the perception of the neighborhood (crime rate, etc.) is poor over time. Therefore, if we just look at the reformulation of the graph and draw the conclusion, we can get these result.
As we can see from the graphs, the house price is high, when…
+ the crime rate is low.
+ proportion of residential land zoned has high value.
+ proportion of non-retail business is less.
+ nitric oxides concentration is low.
+ it has more rooms.
+ the more recent the house was made.
+ the full-value tax rate is lower.
+ pupil-teacher ratio has smaller value.
+ proportion of blacks is high.
+ the lower status of the population is less.
