import pandas as pd
from plotnine import ggplot, aes, geom_point, stat_smooth
import os
import matplotlib.pyplot as plt

# url for csv data
url = 'http://www.statsci.org/data/general/uscrime.txt'

class BasicAnalysis:
    
    def __init__(self):
        self.csv_data = 'data.csv'
        self.normalized_data = 'normlized_data.csv'
        return
    
    def download_to_csv(self, url):
        # download data
        df = pd.read_csv(url, delimiter = '\t')

        # save dataframe to csvfile
        df.to_csv(self.csv_data, index=False)
        
        return
    
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
            #save plots in Plots directory
            gg.save(filename = plot_name, path = "Plots")
            
        return
    
    def normalize_data(self, data):       
        for column in data.columns:
            data[column] = (data[column]-data[column].min())/(data[column].max()-data[column].min()) 
        return
        
    def box_plot(self):
        # load the data to a df
        data = pd.read_csv(self.csv_data)
        
        # normalize data
        self.normalize_data(data)
        
        #save new data to csv file
        data.to_csv(self.normalized_data)
        
        # make folder to put plots
        if not os.path.exists("Plots"):
            os.mkdir("Plots")
            
        data.plot(kind='box', subplots=False, sharey=False, figsize=(20,10))
        # save plot in Plots directory
        plt.savefig('Plots\Data_normalization_plot.jpg')

        return
          
if __name__ == "__main__":
    ba = BasicAnalysis()
    ba.download_to_csv(url)
    #ba.scatter_plot()
    ba.box_plot()