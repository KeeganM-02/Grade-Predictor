import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

##Create dataframe using student-mat.csv
df = pd.read_csv("student-mat.csv",sep=';')
df=df[['G1','G2','G3','age','Medu','Fedu','traveltime','studytime','failures','freetime','famrel','goout','Dalc','Walc','health','absences']]
valPredict = "G3" ##Value we want to predict; Student final grade

X = np.array(df.drop(["G3"],axis = 1))
Y = np.array(df[valPredict])

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y, test_size = 0.2) ##Use 20% of given data for testing.

model = linear_model.LinearRegression() ##Using basic linear regression model
model.fit(x_train,y_train) ##Fit the model to the x and y training sets

score = model.score(x_test,y_test)
predictions = model.predict(x_test)

print(score)
print("Coef: ", model.coef_)
print("Intercept: ", model.intercept_)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
