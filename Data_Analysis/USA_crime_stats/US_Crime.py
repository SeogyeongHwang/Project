import pandas as pd
from plotnine import ggplot, aes, geom_point, stat_smooth
import os

# url for csv data
url = 'http://www.statsci.org/data/general/uscrime.txt'

class BasicAnalysis:
    
    def __init__(self):
        self.csv_data = 'data.csv'
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
        if not os.path.exists("C:/Users/HSG/.spyder-py3/spyder lab/lab6/Plots"):
            os.mkdir("Plots")
        
        for column in data.iloc[:, :-1].columns:
            print(column)
            gg = (ggplot(data, aes(x= data[column], y=data.columns[-1])) + geom_point() + stat_smooth(method = 'lm'))
            plot_name = column + ".jpg"
            #save plots in plots directory
            gg.save(filename = plot_name, path = "Plots")
            
        return
    
if __name__ == "__main__":
    ba = BasicAnalysis()
    ba.download_to_csv(url)
    ba.scatter_plot()