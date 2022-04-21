import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data2021 = pd.read_csv(r'C:\Users\Sarthak Tayal\Downloads\NBA Stats.csv')

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df1 = pd.read_excel(r'C:\Users\User\Desktop\NBA Stats 202122 All Player Statistics in one Page-2 (2).xlsx')
df1 = df1.astype({'GP': 'float', 'MPG': 'float', 'FTA': 'float', '2PA': 'float', '3PA': 'float', 'PPG': 'float', 'RPG': 'float', 'APG': 'float', 'SPG': 'float', 'BPG': 'float', 'TOPG': 'float', 'ORTG': 'float', 'DRTG': 'float'})
print(df1.head())
df1["TOPG"] = -1 * df1["TOPG"]

#Only run one at a time
#For MLR1
x = df1[['GP', 'MPG', 'TOPG', 'RPG', 'SPG', 'BPG', 'FT%', 'APG']]
y = df1['PPG']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 100)
mlr = LinearRegression()
mlr.fit(x_train, y_train)
y_pred_mlr= mlr.predict(x_test)
print("Prediction for test set: {}".format(y_pred_mlr))
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
mlr_diff.head()
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared: {:.2f}'.format(mlr.score(x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)



#For MLR2
x = df1[['3PA', '2PA', 'MPG', 'TOPG', 'RPG', 'SPG', 'BPG', 'APG', 'FTA']]
y = df1['PPG']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
mlr = LinearRegression()
mlr.fit(x_train, y_train)
y_pred_mlr= mlr.predict(x_test)
print("Prediction for test set: {}".format(y_pred_mlr))
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
mlr_diff.head()
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
print('R squared: {:.2f}'.format(mlr.score(x, y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)
