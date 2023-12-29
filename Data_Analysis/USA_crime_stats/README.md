# Analysis of the correlation with the crime rate


## 1. Introduction

   Crime is always happening around us. There are various criminalities in the world, and there may or may not be reasons why crimes occur. Therefore, we tried to visualize and analyze crime-related data from Statis.org through graphs to see which items affect the crime rate. This has been studied using aggregate data on 47 states of the USA for 1960. The data set contains the following items: ‘M: percentage of males aged 14-24 in total state populations’, ‘Ed: mean years of schooling of the population aged 25 years or over’, ‘Po1: per capita expenditure on police protection in 1960’, ‘Po2: per capita expenditure on police protection in 1959’, ‘LF: labour force participation rate of civilian urban males in the age-group 14-24’, ‘M.F: number of males per 100 females’, ‘Pop: state population in 1960 in hundred thousands’, ‘NW: percentage of nonwhite in the population’, ‘U1: unemployment rate of urban males 14-24’, ‘U2: unemployment rate or urban males 35-39’, ‘Wealth: median value of transferable assets or family income’, ‘Ineq: income equality; percentage of families earning below half the median income’, ‘Prob: probability of imprisonment; ratio of number of commitments to number of offenses’, ‘Time: average time in months served by offenders in state prisons before their first release’.
In this report, we checked how much each item affected the crime rate and which one affected the crime rate the most. 

## 2. Method

   Six methods were used in this code.
   
      1) pd.read_csv() : To read csv file into dataframe. The default delimiter is ',' but can be used interchangeably. I switched to '\t'.
      2) to_csv() : To make dataframe as a csv file.
      3) os.path.exists() :  To check whether the directory or file exists or not
      4) os.mkdir() : To create a directory.
      5) ggplot() : To plot data.
                 For the graph type, I added geom_point() to create scatter plots with stat_smooth(method = lm) to display the results with linear model.
      6) ggsave() : To save a ggplot with sensible defaults.

## 3. Result

   < rate of change with Percentage of 14-24 aged men >

   ![Alt text](https://github.com/SeogyeongHwang/Project/blob/8d31bf164e23d3a1715c8437ac26cc3dd609daef/Data_Analysis/USA_crime_stats/Plots/M.jpg)
   
   + When the percentage of males aged 14-24 is between 12 to 14, it generally shows a high crime rate.*
   + It is a very fine slope, but the crime rate shows a downward trend.

   < rate of change with Means years of Education aged 25 years or over >

   ![Alt text](https://github.com/SeogyeongHwang/Project/blob/a3ab64523fb51e5cc0d1c123a1de94c08c40ed83/Data_Analysis/USA_crime_stats/Plots/Ed.jpg)

   + Overall, The graph shows a trend where the crime rate increases, as the mean years increase.

   < rate of change with per capita expenditure on Police protection >

   <p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/a3ab64523fb51e5cc0d1c123a1de94c08c40ed83/Data_Analysis/USA_crime_stats/Plots/Po1.jpg" width="49%" height="50%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/51299959a29601e91d2853c8fc5a040ce4c00e28/Data_Analysis/USA_crime_stats/Plots/Po2.jpg" width="49%" height="50%">
   </p>

   + Left graph is in 1960, the right graph is in 1959. However, these have almost the same trend.
   + Both of them show that the larger the expenditure on police protection is, the crime rate increases.

   < rate of change with Labour Force participation rate of 14-24 aged males >

   ![Alt text](https://github.com/SeogyeongHwang/Project/blob/18c1d8b9adfa01e50818b4cab62300e351f68261/Data_Analysis/USA_crime_stats/Plots/LF.jpg)

   + In this graph, when X data is 0.55~0.60, the average crime rate is higher than other sections.
   + It is increasing very minutely.

   < rate of change with number of Males per 100 Females >
   
   ![Alt text](https://github.com/SeogyeongHwang/Project/blob/683557ff9e6f6abfacc41364da7a4459b39a104c/Data_Analysis/USA_crime_stats/Plots/M.F.jpg)

   + At the end, when the number of males are going up, crime rate is also going up.
   
   < rate of change with state Population in 1960 >

   ![Alt text](https://github.com/SeogyeongHwang/Project/blob/683557ff9e6f6abfacc41364da7a4459b39a104c/Data_Analysis/USA_crime_stats/Plots/Pop.jpg)

   + The unit is hundred thousands.
   + In the begining, It goes up and down in large amounts like a frequency.

   < rate of change with percentage of Nonwhites in the population >

   ![Alt text](https://github.com/SeogyeongHwang/Project/blob/3a1da6efcb7405cee18df5a94dd7de5c693a869d/Data_Analysis/USA_crime_stats/Plots/NW.jpg)

   + It increases very slightlyl, but it has a maximum crime rate value when the x-value is around 10.

   < rate of change with Unemployment rate or urban males >

   <p float="left">
   <img src="[https://github.com/SeogyeongHwang/Project/blob/a3ab64523fb51e5cc0d1c123a1de94c08c40ed83/Data_Analysis/USA_crime_stats/Plots/Po1.jpg](https://github.com/SeogyeongHwang/Project/blob/3a1da6efcb7405cee18df5a94dd7de5c693a869d/Data_Analysis/USA_crime_stats/Plots/U1.jpg)" width="49%" height="50%">
   <img src="[https://github.com/SeogyeongHwang/Project/blob/51299959a29601e91d2853c8fc5a040ce4c00e28/Data_Analysis/USA_crime_stats/Plots/Po2.jpg](https://github.com/SeogyeongHwang/Project/blob/3a1da6efcb7405cee18df5a94dd7de5c693a869d/Data_Analysis/USA_crime_stats/Plots/U2.jpg)https://github.com/SeogyeongHwang/Project/blob/3a1da6efcb7405cee18df5a94dd7de5c693a869d/Data_Analysis/USA_crime_stats/Plots/U2.jpg" width="49%" height="50%">
   </p>

   + Left graph has 14-24 age group, and the right graph has 35-39 age group.
   + For the 14-24 age group, the graph is showing downward trend.
   + For the 35-39 age group, this graph goes from low to high, with the center of the unemployment rate value having the highest crime rate.
   + These two graphs are going in an opposite way.

   < rate of change with Wealth >

   ![Alt text](https://github.com/SeogyeongHwang/Project/blob/3a1da6efcb7405cee18df5a94dd7de5c693a869d/Data_Analysis/USA_crime_stats/Plots/Wealth.jpg)

   + This graph is showing a trend that when the wealth value goes up, the crime rate value also goes up.

   < rate of change with income Inequality >

   ![Alt text](https://github.com/SeogyeongHwang/Project/blob/3a1da6efcb7405cee18df5a94dd7de5c693a869d/Data_Analysis/USA_crime_stats/Plots/Ineq.jpg)

   + It has a gradually lowering shape. The percentage of families earning below half the median income is around 16%, it has the highest crime rate value, but when the x value goes down, the crime rate also goes down.

   < rate of change with Probability of imprisonment >

   ![Alt text](https://github.com/SeogyeongHwang/Project/blob/3a1da6efcb7405cee18df5a94dd7de5c693a869d/Data_Analysis/USA_crime_stats/Plots/Prob.jpg)

   + If the ratio of probability of imprisonment is less than 0.05, the crime rate is the highest.
   + The graph is gradually going down, but when the x-value is around 0.125, the crime rate value suddenly jumped.

   < ratio of change with average Time served by offenders in state prisons before their first release >

   ![Alt text](https://github.com/SeogyeongHwang/Project/blob/3a1da6efcb7405cee18df5a94dd7de5c693a869d/Data_Analysis/USA_crime_stats/Plots/Time.jpg)

   + 

