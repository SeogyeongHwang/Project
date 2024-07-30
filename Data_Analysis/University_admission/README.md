# Analysis of University Admission data

## 1. Introduction

In this project, we will analyze the given data for Admission in the University and predict the probability of the admission of the student in particular university based on various parameters. The different entities or parameters in the dataset are 'Serial No: To uniquely identify students', 'GRE Score: Score of GRE test which is an important test for admission in the graduate school or business school application process globally', 'TOEFL Score: Test of English as a Foreign Language exam score', 'University Rating: Rating of the University out of 5', 'SOP: Related to Statement of Purpose(SOP) for applying to a particular course or university', 'LOR: Some score related to LOR i.e a letter of recommendation', 'CGPA: CGPA is a past performance measure of a aspirant', 'Chance of admit: Probability of the student to get admission the university'.

## 2. Method

Seven methods were used in this code to download the data.

      1) pd.read_csv() : To read csv file into dataframe. The default delimiter is ',' but can be used interchangeably. I switched to '\t'.
      2) to_csv() : To make dataframe as a csv file.
      3) os.path.exists() :  To check whether the directory or file exists or not
      4) os.mkdir() : To create a directory.
      5) ggplot() : To plot data.
             For the graph type, I added geom_point() to create scatter plots with stat_smooth(method = lm) to display the results with linear model.
      6) ggsave() : To save a ggplot with sensible defaults.
      7) isnull() : To check null in the list

## 3. Result

Before plotting the data, we can see that there was no missing value in the dataframe through 'Finding_missing_values' function.

![Alt_text](https://github.com/SeogyeongHwang/Project/blob/bda21fd2263898710ac970d95254807b052177db/Data_Analysis/University_admission/Plots/Data_normalization_plot.jpg)
#### < Normalization data Box Plot >
+ Most variables have a median around 0.6.
+ There are a few outliers, particularly in the CGPA and Chance of Admit, LOR.
+ In contrast, 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP' are widely distributed and have no outliers.
+ Through this graph, 'University Rating' column is hard to get information because it shows that this data composed of only five different values.

#### < Correlation between two data using scatter plot >
<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/bda21fd2263898710ac970d95254807b052177db/Data_Analysis/University_admission/Plots/GRE%20Score.jpg" width="33%" height="33%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/bda21fd2263898710ac970d95254807b052177db/Data_Analysis/University_admission/Plots/TOEFL%20Score.jpg" width="33%" height="33%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/bda21fd2263898710ac970d95254807b052177db/Data_Analysis/University_admission/Plots/CGPA.jpg" width="32%" height="33%">
   </p>
<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/bda21fd2263898710ac970d95254807b052177db/Data_Analysis/University_admission/Plots/LOR%20.jpg" width="33%" height="33%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/bda21fd2263898710ac970d95254807b052177db/Data_Analysis/University_admission/Plots/SOP.jpg" width="33%" height="33%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/bda21fd2263898710ac970d95254807b052177db/Data_Analysis/University_admission/Plots/University%20Rating.jpg" width="33%" height="33%">
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
