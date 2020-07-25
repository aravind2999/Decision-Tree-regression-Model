#import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

#Import Datasets
AgeGroupDetails = pd.read_csv('C:\\Users\\ARAVIND\\Desktop\\AgeGroupDetails.csv')
covid_19_india = pd.read_csv('C:\\Users\\ARAVIND\\Desktop\\covid_19_india.csv')

#convert dates in dataset to a numerical number
from datetime import date
covid_19_india["Date"] = pd.to_datetime(covid_19_india["Date"], dayfirst = True)
covid_19_india['Date'] = covid_19_india['Date'].map(datetime.date.toordinal)

#load all state_names into a list
state_names = covid_19_india['State/UnionTerritory'].unique().tolist()

#create a dictionary by grouping the information of Different states
d = dict(tuple(covid_19_india.groupby('State/UnionTerritory')))

state_cases = {}
X_set = {}
y_set = {}
X_train = {}
X_test = {}
y_train = {}
y_test = {}



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Standard Scaler
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()

#DecisionTree Regression
from sklearn.tree import DecisionTreeRegressor
regressor = {}

for i in range(0, 37):
    df = d[state_names[i]]
    df = df.iloc[:].values
    state_cases[state_names[i]] = df
    X_set[state_names[i]] = state_cases[state_names[i]][:,1:2]
    y_set[state_names[i]] = state_cases[state_names[i]][:,-1]
    X_train[state_names[i]], X_test[state_names[i]], y_train[state_names[i]], y_test[state_names[i]] = train_test_split(X_set[state_names[i]], y_set[state_names[i]], test_size = 0.2, random_state = 0)

#Regression
for i in range(0,37):
    regressor[state_names[i]] = DecisionTreeRegressor()
    regressor[state_names[i]].fit(X_train[state_names[i]], y_train[state_names[i]])


#Prediction
y_pred = {}
for i in range(0,37):
    y_pred[state_names[i]] = regressor.predict(X_test[state_names[i]])

#Error in predictions of corona cases in Kerela State
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test['Kerala'], regressor["Kerala"].predict(X_test["Kerala"]))

#Visualization
X_grid = {}
for i in range(0,37):

    X_grid[state_names[i]] = np.arange(min(X_set[state_names[i]]), max(X_set[state_names[i]]), 0.1)
    X_grid[state_names[i]] = X_grid[state_names[i]].reshape(len(X_grid[state_names[i]]),1)
    plt.plot(X_grid[state_names[i]], regressor[state_names[i]].predict(X_grid[state_names[i]]), color = 'red')
    plt.scatter(X_set[state_names[i]], y_set[state_names[i]])
    plt.title(state_names[i])
    plt.xlabel("Date")
    plt.ylabel("Number of cases")
    plt.show()
    
