# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 22:19:07 2020

@author: saiteja pasunuti
"""

################## K NEAREST NEIGHBOR ######################

import pandas as pd
#pandas is data manipulation
import numpy as np
#deals with numerical data 
import matplotlib.pyplot as plt
#used for basic data visulazations like scatter plot etc

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
#used for splitting the data into train and test datasets
from sklearn.preprocessing import StandardScaler
#StandardScaler standardizes a feature by subtracting the mean and then scaling to unit variance.
import seaborn as sns
#seaborn is used for graphical visualization
from sklearn.metrics import classification_report,accuracy_score
#A Classification report is used to measure the quality of predictions from a classification
from sklearn.model_selection import cross_val_score


#load dataset
glass=pd.read_csv("D:/360digiTMG/unsupervised/mod18 K Nearest Neighbour/glass dataset/glass.csv")
glass.head()

glass.Type.value_counts()

#data visualization
cor=glass.corr()
sns.heatmap(cor)

sns.scatterplot(glass["RI"],glass["Na"],hue=glass["Type"])
sns.pairplot(glass,hue="Type")

scaler=StandardScaler()
scaler.fit(glass.drop("Type",axis=1))
StandardScaler(copy=True, with_mean=True, with_std=True)

#perform transformation
scaled_features = scaler.transform(glass.drop('Type',axis=1))
scaled_features

glass_feat = pd.DataFrame(scaled_features,columns=glass.columns[:-1])
glass_feat.head()

#APPLYING KNN ALGORITHM
dff = glass_feat.drop(['Ca','K'],axis=1) #Removing features - Ca and K 
X_train,X_test,y_train,y_test  = train_test_split(dff,glass['Type'],test_size=0.3,random_state=45)
 #setting random state ensures split is same eveytime, so that the results are comparable

knn = KNeighborsClassifier(n_neighbors=4,metric='manhattan')
knn.fit(X_train,y_train)

KNeighborsClassifier(algorithm="auto",leaf_size=30,metric="manhattan",metric_params=None,n_jobs=None,n_neighbors=4,p=2,weights="uniform")

y_pred=knn.predict(X_test)

print(classification_report(y_test,y_pred))

accuracy_score(y_test,y_pred)
#0.7384615384615385
#we got accuracy of 73.84%

#find best k value
k_range=range(1,25)
k_score=[]
error_rate=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    #kscore -accuracy
    scores=cross_val_score(knn,dff,glass["Type"],cv=5,scoring="accuracy")
    k_score.append(scores.mean())
    
    #error rate
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    error_rate.append(np.mean(y_pred!=y_test))
    
plt.plot(k_range,k_score)
plt.xlabel('value of k - knn algorithm')
plt.ylabel('Cross validated accuracy score')
plt.show()

#plot k vs error rate
plt.plot(k_range,error_rate)
plt.xlabel('value of k - knn algorithm')
plt.ylabel('Error rate')
plt.show()

#we can see that k=4 produces the most accurate results
