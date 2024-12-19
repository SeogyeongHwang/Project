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
This box plot graph provids a clear overview of the central tendencies and variability of each variable relevant to university admissions.
+ Most variables have a median around 0.6.
+ There are a few outliers, particularly in the CGPA and Chance of Admit, LOR.
+ In contrast, 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP' are widely distributed and have no outliers.
+ Through this graph, 'University Rating' column is hard to get information because it shows that this data composed of only five different values.

#### < Correlation between two data using scatter plot >
<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/bda21fd2263898710ac970d95254807b052177db/Data_Analysis/University_admission/Plots/GRE%20Score.jpg" width="32.5%" height="33%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/bda21fd2263898710ac970d95254807b052177db/Data_Analysis/University_admission/Plots/TOEFL%20Score.jpg" width="32.5%" height="33%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/bda21fd2263898710ac970d95254807b052177db/Data_Analysis/University_admission/Plots/CGPA.jpg" width="32.5%" height="33%">
   </p>
<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/bda21fd2263898710ac970d95254807b052177db/Data_Analysis/University_admission/Plots/LOR%20.jpg" width="32.5%" height="33%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/bda21fd2263898710ac970d95254807b052177db/Data_Analysis/University_admission/Plots/SOP.jpg" width="32.5%" height="33%">
   </p>
These scatter plots with regression lines visualize the relationship between various factors and 'Chance of Admit'.
+ All five graphs have a positive slope.
+ 'CGPA' appears to have the strongest correlation with chance of admission.
+ From the graph, we can get the information that every each factors increase, the chance of admission generally increases.
+ 'LOR', 'SOP' have only 9 values to show from the graph. And these have some variability, but the genera trend is upward.


#### < Bivariate Box Plot >
<p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/8004ebb9b86cd610ae0635338cf623164a223159/Data_Analysis/University_admission/Plots/University%20Rating.jpg" width="49%" height="49%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/8004ebb9b86cd610ae0635338cf623164a223159/Data_Analysis/University_admission/Plots/bivariate_boxplot_University%20Rating.jpg" width="49%" height="49%">
   </p>
   
From scatter plot, we can see individual data points and their dispersion, while the box plot provides a summary of the distribution, median. and variability. 
'University of Rating' column only have 5 values to see, so we show scatter plot and box plot as well to see more details.
+ This plots also show a positive correlation between the University Rating and the Chance of Admit.
+ There are several outliers at each rating level, particularly at the lower ratings(1 and 2) showing some applicants with very low chances of admit despite the university rating.
+ Fewer outliers are present at higher ratings(4 and 5), suggesting a stronger consistency in higher chances of admit.

   
## 4. Conclusion

From the above results, we can draw the following conclusions. In the beginning, through my experience in university entry examinations, I knew the results already that every factors are important. I cannot say that these conclusions are exact statistics, because There may be various reasons why the x value is low or high in plotting the values. Every above graphs are showing that if factors (GRE Score, TOEFL Score, CGPA, LOR, SOP, University Rating) goes high, chance of admit is going up and that means my expectation was right. Therefore, if we just look at the reformulation of the graph and draw the conclusion, we can get these result.
As we can see from the graphs, the house price is high, when…
+ every factors(GRE Score, TOEFL Score, CGPA, LOR, SOP, University Rating) are high.
