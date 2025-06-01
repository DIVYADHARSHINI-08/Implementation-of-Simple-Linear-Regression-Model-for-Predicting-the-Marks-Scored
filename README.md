# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DIVYA DHARSHINI R
RegisterNumber: 212223040042

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error,mean_squared_error
#read csv file
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())

# Segregating data to variables
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

#displaying predicted values
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

#graph plot for training data
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#find mae,mse,rmse
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```
## Output:
## Head Values
![image](https://github.com/user-attachments/assets/a6511a6d-418d-4f4f-bc5a-cb0124f79cef)
## Tail Values
![image](https://github.com/user-attachments/assets/7f73b303-4f27-4e6a-ae46-5b23465754a0)
## Compare Set
![image](https://github.com/user-attachments/assets/4c7f656a-059d-454b-ab11-eeb5426a982b)
## Prediction Value of Xand Y
![image](https://github.com/user-attachments/assets/c8d5fba2-ed85-4e56-b7d9-c36cf612a7d3)
## Training Set
![image](https://github.com/user-attachments/assets/5d377388-31fe-48b2-8d42-6f37dd266208)
## Testing Set
![image](https://github.com/user-attachments/assets/71326706-7d22-4d24-84c7-84da25b75b11)
## MSE,MAE and RMSE
![image](https://github.com/user-attachments/assets/0f4128ce-317f-41cb-ba25-fb849c155b83)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
