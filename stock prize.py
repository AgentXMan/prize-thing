'''
QUICK NOTE:
USES LINEAR REGRESSSION AS THE TRAINING MODEL AND GOT 100% ACCURACY
'''
#Importing all the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("yahoo_stock (1) (1).csv") #reads the file
df = pd.DataFrame(data) #makes a spreadsheet and then stores it


# displays the spreadsheet's top rows
df.head()


#displays summary statistics
df.describe()

 
'''
showing column wise % of NaN values they contains 
looping through the columns from the csv file and then detects the missing values (NAN)
Since there is no null valus, we can move further
(data preprocessing and cleaning)
'''
for i in df.columns:
  print(i,"\t-\t", df[i].isna().mean()*100)
  

#finding the pairwise correlation of all columns in the datafram excluding null values (thats why we checked for them above)
cormap = df.corr()

#visualizing the data in colors
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cormap, annot = True)


#finding corelating columns and expressing it
def get_corelated_col(cor_dat, threshold): 
  '''
  Cor_data to be column along which corelation to be measured 
  Threshold be the value above which of corelation to considered
  '''
  feature=[]
  value=[]
  
  #an algorithm to check the correlation
  for i ,index in enumerate(cor_dat.index):
    if abs(cor_dat[index]) > threshold:
      feature.append(index)
      value.append(cor_dat[index])

  df = pd.DataFrame(data = value, index = feature, columns=['corr value'])
  return df


#getting corelated values
top_corelated_values = get_corelated_col(cormap['Close'], 0.60)
top_corelated_values
#all colums except volume are highly co-related. Using them for predictions.


df = df[top_corelated_values.index]
df.head()


df.shape #(number of rows, number of columns)


#checking out and displaying different graphs and pairplot plots pairwise relationships in a dataset 
sns.pairplot(df)
plt.tight_layout()
#Since other parameters have linear relationship with Close, we are using some linear models fore prediction


#.drop() drops the specified labels from rows and columns
X = df.drop(['Close'], axis=1)
y = df['Close']


'''
Since range of data in different columns veries significantly 
we need to scale the independent variable i.e. X. For this we will use Min-Max Scaling.
scaling the data to a smaller range 
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X.head()


from sklearn.model_selection import train_test_split

#splitting data into test train pairs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

Acc = [] #creates an empty dictionary


#we use linear regression for model training
from sklearn.linear_model import LinearRegression

# model training
model_1 = LinearRegression()
model_1.fit(X_train, y_train) #fitting the data into the model, takes training data and trains it


# now we predict after training
y_pred_1 = model_1.predict(X_test)

#Actual is the column of the actual testing data and the prediction column is the predicted data
pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_1}) 
pred_df.head()


'''
measuring the accuracy store of the predictions
best score is 1 and revolves around 0-1 (just like probability in maths)
r2_score basically gets the score from the data
'''

from sklearn.metrics import r2_score
 
print("Accuracy score of the predictions: {0}".format(r2_score(y_test, y_pred_1)))
Acc.append(r2_score(y_test, y_pred_1))


#plotting the graph for the actual value and predictions
plt.figure(figsize=(8,8))
plt.ylabel('Close Price', fontsize=16)
plt.plot(pred_df)
plt.legend(['Actual Value', 'Predictions'])
plt.show()
