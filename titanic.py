#this is the basic machine learning project on predicting the survival rate of
#person on basis of the data provided.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data=pd.read_csv("C:\\Users\\arjun\\Downloads\\train.csv")

test_data=pd.read_csv("C:\\Users\\arjun\\Downloads\\test.csv")
corr_matrix=data.corr(method="pearson")

#                PassengerId  Survived    Pclass       Age     SibSp     Parch      Fare
#PassengerId     1.000000 -0.005007 -0.035144  0.036847 -0.057527 -0.001652  0.012658
#Survived       -0.005007  1.000000 -0.338481 -0.077221 -0.035322  0.081629  0.257307
#Pclass         -0.035144 -0.338481  1.000000 -0.369226  0.083081  0.018443 -0.549500
#Age             0.036847 -0.077221 -0.369226  1.000000 -0.308247 -0.189119  0.096067
#SibSp          -0.057527 -0.035322  0.083081 -0.308247  1.000000  0.414838  0.159651
#Parch          -0.001652  0.081629  0.018443 -0.189119  0.414838  1.000000  0.216225
#Fare            0.012658  0.257307 -0.549500  0.096067  0.159651  0.216225  1.000000

#observing the correaltion matrix for survival it is clear that
#the survival of the person doesn't much depend on the passengerID
#so we can drop the feature altogether. similarly we can drop the name of the person
new_data=data.drop(["Name","PassengerId"],axis=1)

#the remaining features that were not considered in the initial correlation  matrix
#this is because they have null data in them and they do not have integral data in them.
param=data[["Sex","Ticket","Cabin","Embarked"]]

encoder=LabelEncoder()

#add the gender information in the form of binary data instead of alphabets
#and remove the Sex column from the feature table
var_sex=encoder.fit_transform(new_data["Sex"])
new_data=new_data.assign(gender=var_sex)
new_data=new_data.drop(["Sex"],axis=1)

#this shows the location in the embarked column 
param_data=new_data
param_data[param_data["Embarked"].isnull()].index.tolist()
#[61, 829]
#since there are only two rows of data, we can just drop the rows of data
param_data=param_data.drop(param_data.index[829])
param_data=param_data.drop(param_data.index[61])

k=param_data.isnull().any()
#Survived    False
#Pclass      False
#Sex         False
#Age          True
#SibSp       False
#Parch       False
#Ticket      False
#Fare        False
#Cabin        True
#Embarked    False
#gender      False
#multi       False
t=param_data["Ticket"].unique()
#print(t.size)
#680
#since the tickets are almost unique we can drop this column for a basic model

param_corr=param_data.corr(method="pearson")
#           Survived    Pclass     SibSp     Parch      Fare    gender
#Survived  1.000000 -0.335346 -0.034692  0.082157  0.255918 -0.543297
#Pclass   -0.335346  1.000000  0.081852  0.017154 -0.548372  0.128099
#SibSp    -0.034692  0.081852  1.000000  0.414350  0.161041 -0.116753
#Parch     0.082157  0.017154  0.414350  1.000000  0.217845 -0.248300
#Fare      0.255918 -0.548372  0.161041  0.217845  1.000000 -0.179797
#gender   -0.543297  0.128099 -0.116753 -0.248300 -0.179797  1.000000

var=param_data["Embarked"]
encoder=LabelEncoder()
var=encoder.fit_transform(param_data["Embarked"])
param_data["embark"]=var
param_data=param_data.drop(["Embarked"],axis=1)
param_corr=param_data.corr(method="pearson")
#           Survived    Pclass     SibSp     Parch      Fare    gender    embark
#Survived  1.000000 -0.335346 -0.034692  0.082157  0.255918 -0.543297 -0.170746
#Pclass   -0.335346  1.000000  0.081852  0.017154 -0.548372  0.128099  0.164972
#SibSp    -0.034692  0.081852  1.000000  0.414350  0.161041 -0.116753  0.068635
#Parch     0.082157  0.017154  0.414350  1.000000  0.217845 -0.248300  0.039963
#Fare      0.255918 -0.548372  0.161041  0.217845  1.000000 -0.179797 -0.226187
#gender   -0.543297  0.128099 -0.116753 -0.248300 -0.179797  1.000000  0.109889
#embark   -0.170746  0.164972  0.068635  0.039963 -0.226187  0.109889  1.000000

new_data=param_data

#now we need to seperate the labels as this is an example of a supervised learning
#it is a classification problem

#now it is time to train the models but first seperate the labels and the features
X=new_data
y=new_data["Survived"]
X=X.drop(["Survived"],axis=1)
X=X.drop(["Ticket"],axis=1)
X=X.drop(["Cabin"],axis=1)
X=X.drop(["Age"],axis=1)


#print(X)
#print(y)
#model=LogisticRegression()
#model.fit(X,y)
#do the similar data wrangling for the test set

test_data=test_data.drop(["Name"],axis=1)
new_data=test_data.drop(["PassengerId"],axis=1)
encoder=LabelEncoder()
var_sex=encoder.fit_transform(new_data["Sex"])
new_data=new_data.assign(gender=var_sex)
new_data=new_data.drop(["Sex"],axis=1)
#print(new_data)
param_data=new_data
var=param_data["Embarked"]
encoder=LabelEncoder()
var=encoder.fit_transform(param_data["Embarked"])
param_data["embark"]=var
param_data=param_data.drop(["Embarked"],axis=1)
param_data=param_data.drop(["Ticket"],axis=1)
param_data=param_data.drop(["Cabin"],axis=1)
param_data=param_data.drop(["Age"],axis=1)

Xtest=param_data
Xtest[Xtest["Fare"].isnull()].index.tolist()
#[152]
#one test data has a null value so we can just discard it
Xtest=Xtest.drop(index=152)

X_validate=X.iloc[700:888]
X["target"]=y
X_validate=X.iloc[700:888]
y_validate=X_validate["target"]
X_validate=X_validate.drop(["target"],axis=1)
model=LogisticRegression()
Xtrain=X.iloc[:700]
ytrain=Xtrain["target"]
Xtrain=Xtrain.drop(["target"],axis=1)
model.fit(Xtrain,ytrain)
y_predicted=model.predict(X_validate)
print(accuracy_score(y_predicted,np.array(y_validate),normalize=True))

#the model score is 0.80 which is satisfactory for first model
