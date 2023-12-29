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
   
      1) When the percentage of males aged 14-24 is between 12 to 14, it generally shows a high crime rate.*
      2) It is a very fine slope, but the crime rate shows a downward trend.


   < rate of change with Means years of Education aged 25 years or over >

   ![Alt text](https://github.com/SeogyeongHwang/Project/blob/a3ab64523fb51e5cc0d1c123a1de94c08c40ed83/Data_Analysis/USA_crime_stats/Plots/Ed.jpg)

      1) Overall, The graph shows a trend where the crime rate increases, as the mean years increase.


   < rate of change with per capita expenditure on Police protection >

   <p float="left">
   <img src="https://github.com/SeogyeongHwang/Project/blob/a3ab64523fb51e5cc0d1c123a1de94c08c40ed83/Data_Analysis/USA_crime_stats/Plots/Po1.jpg" width="49%" height="50%">
   <img src="https://github.com/SeogyeongHwang/Project/blob/51299959a29601e91d2853c8fc5a040ce4c00e28/Data_Analysis/USA_crime_stats/Plots/Po2.jpg" width="49%" height="50%">
   </p>

      1) Left graph is in 1960, the right graph is in 1959. However, these have almost the same trend.
      2) Both of them show that the larger the expenditure on police protection is, the crime rate increases.


   < rate of change with Labour Force participation rate of 14-24 aged males >

   ![Alt text](https://github.com/SeogyeongHwang/Project/blob/18c1d8b9adfa01e50818b4cab62300e351f68261/Data_Analysis/USA_crime_stats/Plots/LF.jpg)

      1) 
