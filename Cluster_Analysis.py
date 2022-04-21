import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

data2021 = pd.read_excel(r'C:\Users\Sarthak Tayal\Downloads\NBA Stats 202122 All Player Statistics in one Page-2.xlsx')
print(data2021)

sns.pairplot(data2021[['PPG', 'APG', 'RPG']])
plt.show()

correlation = data2021[['PPG', 'APG', 'RPG']].corr()
sns.heatmap(correlation, annot=True)

kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = data2021._get_numeric_data().dropna(axis=1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
labels

pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:, 0], y=plot_columns[:, 1], c=labels)
plt.show()

# Find player LeBron
LeBron = good_columns.loc[data2021['FULL NAME'] == 'LeBron James', :]

#Find player Durant
Durant = good_columns.loc[data2021['FULL NAME'] == 'Kevin Durant', :]

#print the players
print(LeBron)
print(Durant)

#Change the dataframes to a list
Lebron_list = LeBron.values.tolist()
Durant_list = Durant.values.tolist()

plot_columns

#Predict which group LeBron James and Kevin Durant belongs
LeBron_Cluster_Label = kmeans_model.predict(Lebron_list)
Durant_Cluster_Label = kmeans_model.predict(Durant_list)

print(LeBron_Cluster_Label)
print(Durant_Cluster_Label)

data2021.corr()

x_train, x_test, y_train, y_test = train_test_split(data2021[['PPG']], data2021[['APG']], test_size=0.2, random_state=42)

#Create the Linear Regression Model
lr = LinearRegression() # Create the model
lr.fit(x_train, y_train) #Train the model
predictions = lr.predict(x_test) #Make predictions on the test data
print(predictions)
print(y_test)

lr_confidence = lr.score(x_test, y_test)
print("lr confidence (R^2): ", lr_confidence)

print("Mean Squared Error (MSE): ", mean_squared_error(y_test, predictions))

