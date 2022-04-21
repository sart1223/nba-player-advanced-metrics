import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, r2_score
import csv

data = pd.read_csv(r'C:\Users\Sarthak Tayal\PycharmProjects\nba-player-advanced-metrics\nba-data-historical.csv')
print(data)

data_2010_2020 = data[data['year_id'] > 2010]
print(data_2010_2020)

data_2020 = data[data['year_id'] == 2020]
df2 = data_2020.mean(axis=0)
threepa_2020 = df2['3PAr']
ts_2020 = df2['TS%']
tov_2020 = df2['TOV%']

data_2019 = data[data['year_id'] == 2019]
df3 = data_2019.mean(axis=0)
threepa_2019 = df3['3PAr']
ts_2019 = df3['TS%']
tov_2019 = df3['TOV%']

data_2018 = data[data['year_id'] == 2018]
df4 = data_2018.mean(axis=0)
threepa_2018 = df4['3PAr']
ts_2018 = df4['TS%']
tov_2018 = df2['TOV%']

data_2017 = data[data['year_id'] == 2017]
df5 = data_2017.mean(axis=0)
threepa_2017 = df5['3PAr']
ts_2017 = df5['TS%']
tov_2017 = df5['TOV%']

data_2016 = data[data['year_id'] == 2016]
df6 = data_2016.mean(axis=0)
threepa_2016 = df6['3PAr']
ts_2016 = df6['TS%']
tov_2016 = df6['TOV%']

data_2015 = data[data['year_id'] == 2015]
df7 = data_2015.mean(axis=0)
threepa_2015 = df7['3PAr']
ts_2015 = df7['TS%']
tov_2015 = df7['TOV%']

data_2014 = data[data['year_id'] == 2014]
df8 = data_2014.mean(axis=0)
threepa_2014 = df8['3PAr']
ts_2014 = df8['TS%']
tov_2014 = df8['TOV%']

data_2013 = data[data['year_id'] == 2013]
df9 = data_2013.mean(axis=0)
threepa_2013 = df9['3PAr']
ts_2013 = df9['TS%']
tov_2013 = df9['TOV%']

data_2012 = data[data['year_id'] == 2012]
df10 = data_2012.mean(axis=0)
threepa_2012 = df10['3PAr']
ts_2012 = df10['TS%']
tov_2012 = df10['TOV%']

data_2011 = data[data['year_id'] == 2011]
df11 = data_2011.mean(axis=0)
threepa_2011 = df11['3PAr']
ts_2011 = df11['TS%']
tov_2011 = df11['TOV%']

data_2010 = data[data['year_id'] == 2010]
df12 = data_2010.mean(axis=0)
threepa_2010 = df12['3PAr']
ts_2010 = df12['TS%']
tov_2010 = df12['TOV%']

years = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]

threepa_list = [threepa_2010, threepa_2011, threepa_2012, threepa_2013, threepa_2014,
                threepa_2015, threepa_2016, threepa_2017, threepa_2018, threepa_2019,
                threepa_2020]

fig, threepa = plt.subplots()
threepa.plot(years, threepa_list,
             marker = 'o',
             color = "red",
             alpha = 0.5)

threepa.set(title = "League Average 3PAr per Season\n(2010 - 2020)",
       xlabel = "Years",
       ylabel = "Average 3PAr")

plt.setp(threepa.get_xticklabels(), rotation = 45)
threepa.set_facecolor('xkcd:sky blue')

plt.show()

ts_list = [ts_2010, ts_2011, ts_2012, ts_2013, ts_2014,
           ts_2015, ts_2016,ts_2017, ts_2018, ts_2019, ts_2020]

fig, ts = plt.subplots()
ts.plot(years, ts_list,
        marker = 'o',
        color = "red",
        alpha = 0.5)

ts.set(title = "League Average TS% per Season\n(2010 - 2020)",
       xlabel = "Years",
       ylabel = "Average TS%")

plt.setp(ts.get_xticklabels(), rotation = 45)
ts.set_facecolor('xkcd:sky blue')

plt.show()

tov_list = [tov_2010, tov_2011, tov_2012, tov_2013, tov_2014,
            tov_2015, tov_2016, tov_2017, tov_2018, tov_2019, tov_2020]

fig, tov = plt.subplots()
tov.plot(years, tov_list,
         marker = 'o',
         color = "red",
         alpha = 0.5)

tov.set(title = "League Average TOV% per Season\n(2010 - 2020)",
       xlabel = "Years",
       ylabel = "Average TOV%")

plt.setp(tov.get_xticklabels(), rotation = 45)
tov.set_facecolor('xkcd:sky blue')

plt.show()