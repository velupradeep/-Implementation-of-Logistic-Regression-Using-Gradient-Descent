# EX-05 Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.
 
 ## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: PRADEEP V
RegisterNumber: 212223240119
*/
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:/Users/admin/Downloads/Placement_Data.csv")
dataset

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

Y

theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 /(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(x.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred 
y_pred=predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)


```

## Output:
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/bccc7736-a315-4ed6-b17f-93f1e18a0b2f)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/d9939646-1198-4cd8-91a7-9f064465d3d3)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/f7b59e6a-cd06-4929-8c83-4ce963342c28)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/6c74869b-492e-4b79-a875-b604186b91bd)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/f7d9225b-cbd1-4f63-ba15-b9d65700097f)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/8262bc86-c9ec-4af8-a2ce-7d016b984d05)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/ca8a0fc4-6c6d-464c-b70b-272131cd4cae)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/63f8eaf2-d715-4dd1-9669-b5f5a210134b)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/cdaace1a-c388-4a00-8ed1-73d9710e29bd)

![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/f326e34b-48fa-4bec-abd4-d447e346dbe8)

![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/4c16def5-f46d-479d-83b5-1570cdb09f12)

























## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

