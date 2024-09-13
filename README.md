# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset and print the values.
3. Define X and Y array and display the value.
4. Find the value for cost and gradient.
5. Plot the decision boundary and predict the Regression value.

## Program:
Program to implement the the Logistic Regression Using Gradient Descent.

Developed by: Senthil kumaran c

RegisterNumber: 212223220103

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Placement_Data.csv')
dataset
dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
def gradient_descent(theta, X, y, alpha, num_iterations):
  m = len(y)
  for i in range(num_iterations):
    h = sigmoid(X.dot(theta))
    gradient = X.T.dot(h - y) / m
    theta -= alpha * gradient
  return theta
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
  h = sigmoid(X.dot(theta))
  y_pred = np.where(h >= 0.5,1, 0)
  return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict (theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict (theta, xnew)
print(y_prednew)
```

## Output:
dataset
![Image-1](https://github.com/user-attachments/assets/be38beee-3062-4a18-81aa-eceebfdf17c3)
![Image-2](https://github.com/user-attachments/assets/82cc1b6b-badd-4720-8fb6-b2c7e91927ff)
![Image-3](https://github.com/user-attachments/assets/16942f4e-d8bc-4231-972c-530a929cbcf2)
![Image-4](https://github.com/user-attachments/assets/6f2e7df0-0623-4152-a6c1-b509c1710c57)

Accuracy and Predicted Values

![Image-8](https://github.com/user-attachments/assets/b6baa93d-7a2d-4887-82eb-706d1c809085)
![Image-5](https://github.com/user-attachments/assets/c491d7c8-3a02-4e87-bd23-e7e24ecbc650)
![Image-6](https://github.com/user-attachments/assets/73483700-b74f-42e6-8c83-1e02a6cda013)
![Image-7](https://github.com/user-attachments/assets/928f8db1-59db-4077-903c-b25d6c676c80)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

