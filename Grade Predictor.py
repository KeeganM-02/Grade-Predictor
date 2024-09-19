import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle


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

print(score)
print("Coef: ", model.coef_)
print("Intercept: ", model.intercept_)

predictions = model.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
