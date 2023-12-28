import pandas as pd
from plotnine import ggplot, aes, geom_point, stat_smooth

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
        # remove 'crime' column
        data2 = data.iloc[:, :-1]
        
        for column in data2.columns:
            print(column)
            gg = (ggplot(data, aes(x= data[column], y=data.columns[-1])) + geom_point() + stat_smooth(method = 'lm'))
            plot_name = column + ".jpg"
            #save plots in plots directory
            gg.save(filename = plot_name, path = "Plot images")
            
        return
    
if __name__ == "__main__":
    ba = BasicAnalysis()
    ba.download_to_csv(url)
    ba.scatter_plot()