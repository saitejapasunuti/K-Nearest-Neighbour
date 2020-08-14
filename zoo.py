# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 18:51:01 2020

@author: saiteja pasunuti
"""

import numpy as np
#used for numerical data
import pandas as pd
#pandas is used for data manipulation
import matplotlib.pyplot as plt
#Matplotlib generally consists of bars, pies, lines, scatter plots
import seaborn as sns
#Seaborn is a library for making statistical graphics in Python
import warnings
warnings.filterwarnings("ignore")


zoo=pd.read_csv("D:/360digiTMG/unsupervised/mod18 K Nearest Neighbour/zoo dataset/Zoo.csv")

zoo.head()
zoo.info()
zoo.describe()

zoo.drop("animal name",axis=1,inplace=True)

color_list=[("red" if i==1 else "blue" if i==0 else "yellow") for i in zoo.hair]

unique_list = list(set(color_list))
unique_list

sns.countplot(x="hair",data=zoo)
plt.xlabel("hair")
plt.ylabel("count")
plt.show()

zoo.loc[:,"hair"].value_counts()
#0    58
#1    43

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
x,y=zoo.loc[:,zoo.columns != "hair"],zoo.loc[:,"hair"]
knn.fit(x,y)
prediction=knn.predict(x)
print("prediction=",prediction)


#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
knn=KNeighborsClassifier(n_neighbors=1)
x,y = zoo.loc[:,zoo.columns != 'hair'], zoo.loc[:,'hair']
knn.fit(x_train,y_train)
prediction=knn.predict(x_test)
print('With KNN (K=1) accuracy is: ',knn.score(x_test,y_test))
#With KNN (K=1) accuracy is:  0.967741935483871

k_values=np.arange(1,25)
train_acc=[]
test_acc=[]

for i ,k in enumerate(k_values):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    train_acc.append(knn.score(x_train,y_train))
    test_acc.append(knn.score(x_test,y_test))
    
####plot
plt.figure(figsize=[13,8])
plt.plot(k_values,test_acc,label ="test accuracy")
plt.plot(k_values,train_acc,label="train accuracy")
plt.legend()
plt.title("value vs accuracy")
plt.xlabel("number of neighbor")
plt.ylabel("accuracy")
plt.xticks(k_values)
plt.show()
print("best accuracy is {} with k={}".format(np.max(test_acc),1+test_acc.index(np.max(test_acc))))
#best accuracy is 0.967741935483871 with k=1