import pandas as pd
from plotnine import ggplot, aes, geom_line

# download data
url = 'http://www.statsci.org/data/general/uscrime.txt'
df = pd.read_csv(url, delimiter = '\t')

# change dataframe to csvfile
df.to_csv('data.csv', index=False)

#plot data using ggplot library
data = pd.read_csv('data.csv')

gg = (ggplot(data, aes(x='Time', y='Crime')) + geom_line())
print(gg)