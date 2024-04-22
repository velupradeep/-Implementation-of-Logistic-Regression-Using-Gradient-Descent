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
![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/8bf85e52-39e1-4c6b-b04b-666eaa74d95c)

![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/86e80108-cdec-46d1-98e7-8f1e1ab65660)

![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/214c0417-882a-4d53-9bb8-68b5cee06b89)

![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/256b861a-e01f-440c-a408-cd3c57f3296d)

![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/452fd814-9eb6-4b74-b68f-b2b2ef97699f)

![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/ce6d744d-8a8b-4cdd-a910-a0cc3d53c68c)

![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/fe9fd3c2-def9-479f-aaf2-c583f3a6a79f)

![image](https://github.com/velupradeep/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150329341/61823f33-fe75-486f-924d-75e9fee0f0e2)













## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

