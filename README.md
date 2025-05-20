# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries-Load necessary Python libraries.

2.Load Data-Read the dataset containing house details.

3.Preprocess Data-Clean and solit the data into training and testing sets.

4.Select Features & Target-Choose input variables(features) and output variables(house price,occupants).

5.Train Mode-Use SGDRegressor() to train the model.

6.Make Predictions-Use the model to predict house price and occupants.

7.Evaluate Performance-Check accuracy using error metrics. 

8.Improve Model-Tune settings for better accuracy.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: HARISH P K
RegisterNumber:  212224040104
*/
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
print(data)
```
![image](https://github.com/user-attachments/assets/5aa13ac6-eca5-4309-bba4-fc16f5e35747)
```
df=pd.DataFrame(data.data,columns = data.feature_names)
df['target'] = data.target
print(df.head)
print(df.tail)
print(df.info())
```
![image](https://github.com/user-attachments/assets/f7f1f4b1-f4ec-48ba-9fd0-8638b11470e0)
```
X=df.drop(columns=['AveOccup','target'])
X.info()
Y = df['target']
```
![image](https://github.com/user-attachments/assets/29f965b7-645d-4559-ae22-fb9ed531fa2c)
```
print(X.shape)
print(Y.shape)
```
![image](https://github.com/user-attachments/assets/0da46cfa-c1fc-4f65-97cf-98032c01ba8a)
```
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=11)
X.head()
```
![image](https://github.com/user-attachments/assets/90f788c5-4630-4d01-8821-e3aa7cee141e)
```
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
```
![image](https://github.com/user-attachments/assets/81669e8e-ea69-4e42-9c8f-62aaeb144236)
```
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
print(X_train)
```
![image](https://github.com/user-attachments/assets/654d2ce9-d29e-473b-be35-4f5605306434)
```
sgd = SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
```
![image](https://github.com/user-attachments/assets/d2d9fa1d-dab2-4837-9e6b-a5e344b8a93d)
```
Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",Y_pred[:5])
```
## Output:
![image](https://github.com/user-attachments/assets/8e08b1ac-5f58-435e-8d0f-14a17e38ef1a)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
