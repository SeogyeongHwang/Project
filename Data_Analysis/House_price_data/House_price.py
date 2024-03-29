import pandas as pd
from plotnine import ggplot, aes, geom_point, stat_smooth
import os
import matplotlib.pyplot as plt
import seaborn as sns
import ssl

# To use unverified ssl
ssl._create_default_https_context = ssl._create_unverified_context

# url for csv data
url = 'https://lib.stat.cmu.edu/datasets/boston'

class BasicAnalysis:
    
    def __init__(self):
        self.csv_data = 'data.csv'
        self.normalized_data = 'normlized_data.csv'
        
        return
    
    def download_to_csv(self, url):
        # download header
        df_header = pd.read_csv(url, sep=" ", usecols=[1], skiprows=7, nrows=14, header=None)        
        header_name = df_header[1].to_list()

        # download the data
        pre_df = pd.read_csv(url, skiprows=21, header=None, delimiter=r"\s+", names = header_name)

        # reprocess data
        df = self.reprocessing_data(pre_df, header_name)

        # save dataframe to csvfile
        df.to_csv(self.csv_data, index=False)

        return

    def reprocessing_data(self, pre_df, header_name):
        Even_df = pre_df[::2].reset_index(drop=True)
        Odd_df = pre_df[1::2].reset_index(drop=True)
 
        # merge two data
        merge_df = pd.concat([Even_df, Odd_df], axis=1)
        new_df = merge_df.dropna(axis=1)
       
        # reset header
        new_df.columns = header_name
        
        return new_df
    
    def scatter_plot(self):
        # load the data to a df
        data = pd.read_csv(self.csv_data)
        
        # make folder to put plots
        if not os.path.exists("Plots"):
            os.mkdir("Plots")
        
        for column in data.iloc[:, :-1].columns:
            print(column)
            gg = (ggplot(data, aes(x= data[column], y=data.columns[-1])) + geom_point() + stat_smooth(method = 'lm'))
            plot_name = column + ".jpg"
            # save plots in Plots directory
            gg.save(filename = plot_name, path = "Plots")
            
        return
    
    def normalize_data(self, data):       
         for column in data.columns:
             data[column] = (data[column]-data[column].min())/(data[column].max()-data[column].min()) 

         return

    def box_plot(self):
         # load the data to a df
         data = pd.read_csv(self.csv_data)
         # copy the data to normalize
         normalization = data.copy()
         
         # normalize data
         self.normalize_data(normalization)
         
         # save new data to csv file
         normalization.to_csv(self.normalized_data)
         
         # make folder to put plots
         if not os.path.exists("Plots"):
             os.mkdir("Plots")
             
         normalization.plot(kind='box', subplots=False, sharey=False, figsize=(20,10))
         # save plot in Plots directory
         plt.savefig('Plots\Data_normalization_plot.jpg')
         
         return
     
    def bivariate_boxplot(self):
        # load the data to a df
        data = pd.read_csv(self.csv_data)
        
        # make folder to put plots
        if not os.path.exists("Plots"):
            os.mkdir("Plots")
        
        col = ['CHAS', 'RAD']

        for column in col:        
            sns.boxplot(x=column, y=data.columns[-1], data=data)
            # save plot
            plt.savefig('Plots\\bivariate_boxplot_'+column +'.jpg')        
            
        return
     
    def heatmap_plot(self):
         # load the data to a df
         data = pd.read_csv(self.csv_data)
         
         fig, ax = plt.subplots(figsize=(15,10))
         sns.heatmap(data = data.corr(), annot=True, cmap='RdYlBu_r')
         
         # make folder to put plots
         if not os.path.exists("Plots"):
             os.mkdir("Plots")
         # save plots in Plots directory
         plt.savefig('Plots\Data_heatmap_plot.jpg')
         
         return

    def find_missing_values(self):
        # load the data to a df
        data = pd.read_csv(self.csv_data)
        
        # make dictionary
        missings = {}
        values = []

        for item in data.columns:
            for i in data.index:
                if data.isnull()[item][i] == True:
                   values.append(i)
                   # add key, value in a dict
                   missings[item] = values
            # reset list
            values = []

        print(missings)        
    
if __name__ == "__main__":
    ba = BasicAnalysis()
    ba.download_to_csv(url)
    #ba.scatter_plot()
    #ba.box_plot()
    ba.bivariate_boxplot()
    #ba.heatmap_plot()
    ba.find_missing_values()