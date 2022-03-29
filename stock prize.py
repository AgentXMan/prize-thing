#Importing all the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("yahoo_stock (1) (1).xls")
df = pd.DataFrame(data) #makes a spreadsheet and then stores it

# displays the spreadsheet's top rows
df.head()

#displays summary statistics
df.describe()

# showing column wise %ge of NaN values they contains 

# looping through the columns from the csv file and then detects the missing values (NAN)
# Since there is no null valus, we can move further
for i in df.columns:
  print(i,"\t-\t", df[i].isna().mean()*100)
  
  
#finding the pairwise correlation of all columns in the datafram excluding null values (thats why we checked for them above)
cormap = df.corr()

#visualizing the data in colors
fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cormap, annot = True)


#finding corelating columns and expressing it
def get_corelated_col(cor_dat, threshold): 
  # Cor_data to be column along which corelation to be measured 
  #Threshold be the value above which of corelation to considered
  feature=[]
  value=[]

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


df.shape #(number of rows, number of columns)

#checking out different graphs and pairplot plots pairwise relationships in a dataset 
sns.pairplot(df)
plt.tight_layout()

#Since other parameters have linear relationship with Close, we are using some linear models fore prediction

#.drop() drops the specified labels from rows and columns
X = df.drop(['Close'], axis=1)
y = df['Close']

#Since range of data in different columns veries significantly we need to scale the independent variable i.e. X. For this we will use Min-Max Scaling.
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X.head()


