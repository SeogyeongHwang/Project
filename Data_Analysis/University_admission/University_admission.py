import pandas as pd
from plotnine import ggplot, aes, geom_point, stat_smooth
import os
import matplotlib.pyplot as plt
import seaborn as sns

class BasicAnalysis:
    
    def __init__(self):
        self.csv_data = 'data.csv'
        self.normalized_data = 'normalized_data.csv'     
        return
    
    def download_to_csv(self):
        # download data
        df = pd.read_csv(self.csv_data)
        
        
        return
    
    def scatter_plot(self):
        # load the data to a df
        data = pd.read_csv(self.csv_data)
        
        for column in data.iloc[:, :-1].columns:
            print(column)
            gg = (ggplot(data, aes(x=data[column])))
    
if __name__ == "__main__":
    ba = BasicAnalysis()
    ba.download_to_csv()
    
    