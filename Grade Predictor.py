import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from tkinter import *


##Create dataframe using student-mat.csv
df = pd.read_csv("student-mat.csv",sep=';')
df=df[['G1','G2','G3','age','Medu','Fedu','studytime','failures','famrel','Walc','health']]
valPredict = "G3" ##Value we want to predict; Student final grade

X = np.array(df.drop(["G3"],axis = 1))
Y = np.array(df[valPredict])
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y, test_size = 0.2)

##Previously used loop to save the model with the best score into 'gradingmodel.pickle' to be used later.

# best = 0
# for i in range(1000):
#     x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y, test_size = 0.2) ##Use 20% of given data for testing.
#
#     model = linear_model.LinearRegression() ##Using basic linear regression model
#     model.fit(x_train,y_train) ##Fit the model to the x and y training sets
#
#     score = model.score(x_test,y_test)
#     print(score)
#
#     if score > best:
#         best = score
#         with open("gradingmodel.pickle","wb") as file:
#             pickle.dump(model, file)

##Loads in the saved model from gradingmodel.pickle
pickle_in = open("gradingmodel.pickle","rb")
model = pickle.load(pickle_in)

score = model.score(x_test,y_test)

####### Used this code to show the score, coefficients, and y-intercepts of the model being trained, as well as output it's predictions #######
# print(score)
# print("Coef: ", model.coef_)
# print("Intercept: ", model.intercept_)
#
# predictions = model.predict(x_test)
# for x in range(len(predictions)):
#     print(predictions[x], x_test[x], y_test[x])



##Get user inputs from terminal
print("first period score (1-20): ")
userPer1 = int(input())
print("second period score (1-20): ")
userPer2 = int(input())
print("age (in years): ")
userAge = int(input())
print("mother's education level (0: none, 1: primary education (4th grade), 2: 5th to 9th grade, 3: secondary education or 4: higher education): ")
userMedu = int(input())
print("father's education level (0: none, 1: primary education (4th grade), 2: 5th to 9th grade, 3: secondary education or 4: higher education): ")
userFedu = int(input())
print("weekly study time (1: <2 hours, 2: 2 to 5 hours, 3: 5 to 10 hours, or 4: >10 hours): ")
userStudy = int(input())
print("number of past class failures (0-4): ")
userFail = int(input())
print("quality of family relations (1: very bad, 5: excellent): ")
userFamrel = int(input())
print("weekend alcohol consumption (1: very low, 5: very high): ")
userWalc = int(input())
print("current health status (1: very bad, 5: very good): ")
userHealth = int(input())

userInputs=[[userPer1,userPer2,userAge,userMedu,userFedu,userStudy,userFail,userFamrel,userWalc,userHealth]]

prediction = model.predict(userInputs)
print("Predicted final score (from 1-20): ", prediction[0])

