# Implementation-of-Logistic-Regression-Using-Gradient-Descent

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
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/d9272053-300f-46a6-a72f-0818369ad9eb)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/597f6d16-3cf8-401c-8023-d2a924240a3f)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/8344bfa0-5c1a-471b-a885-0d7d95a450f9)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/e10333cb-c34b-4cfd-915b-70823a5765d4)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/22e5c742-9ba9-4989-abdd-3b129452f432)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/6b86a604-ab6b-4ff7-bda0-55841f95aedd)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/0f295f34-3d17-4fab-bd3a-37f4504a1fa7)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/a9321798-1be6-4cd1-bd16-250171e0d00b)
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/e08172ec-2640-4eb5-8129-4d9405121410)








## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

