# Analysis of Boston Housing Price data

## 1. Introduction



## 2. Method

Seven methods were used in this code to download the data.

      1) pd.read_csv() : To read csv file into dataframe. The default delimiter is ',' but can be used interchangeably.
      2) to_list() : To change the values of a particular column into a list.
      3) df[::2] or df[1::2] : To extract even or odd rows from dataframe.
      4) reset_index(drop=True) : To reset index.
      5) pd.concat([df1, df2], axis=1) : To concat two dataframe. Depending on whe6ther axis is 0 or 1, you can choose which direction to combine the data between rows and columns.
      6) dropna(axis=1) : To remove columns that have NaN values
      7) to_csv() : To make dataframe as a csv file

## 
