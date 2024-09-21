import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from customtkinter import *

def Predict():
    userInputs = [[int(Input_G1.get()),int(Input_G2.get()),int(Input_Age.get()),int(Input_Medu.get()),int(Input_Fedu.get()),int(Input_Study.get()),int(Input_Fail.get()),
                   int(Input_Famrel.get()),int(Input_Walc.get()),int(Input_Health.get())]]
    prediction = model.predict(userInputs)
    Output_text.configure(state="normal")
    Output_text.delete(1.0, END)
    Output_text.insert(END, round(prediction[0],2))
    Output_text.configure(state="disabled")



##Create dataframe using student-mat.csv
df = pd.read_csv("student-mat.csv",sep=';')
df=df[['G1','G2','G3','age','Medu','Fedu','studytime','failures','famrel','Walc','health']]
valPredict = "G3" ##Value we want to predict; Student final grade

X = np.array(df.drop(["G3"],axis = 1))
Y = np.array(df[valPredict])
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y, test_size = 0.2)

##Previously used loop to save the model with the best score into 'gradingmodel.pickle' to be used later.##
####################If you would like to retrain the model, uncomment following loop and run as many iterations as you'd like!####################

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


###Create tkinter object
root = CTk()
set_appearance_mode("dark")
root.geometry('1100x500')
root.resizable(True,True)
root.title("Grade Predictor")

##General labels
CTkLabel(root, text="Grade Predictor", font = ('Arial',50,"bold")).pack()
CTkLabel(root, text="Input Your Information Here", font = ('Arial',30,"bold")).place(x=115,y=90)
CTkLabel(root, text="Predicted Period 3 Grade", font = ('Arial',30,"bold")).place(x=715, y = 90)

##Label pointing to student.txt for more information on variables
CTkLabel(root, text = "*Go to student.txt for more\n "
                   "information on variables and scoring system*",
      font = ('Arial',10)).place(x=0,y=0)

##Labels for each variable
CTkLabel(root, text="First Period Score (1-20)", font = ('Arial',18)).place(x=70, y = 145)
CTkLabel(root, text="Second Period Score (1-20)", font = ('Arial',18)).place(x=350, y = 145)
CTkLabel(root, text="Mother's Education Level (1-4)", font = ('Arial',18)).place(x=70, y = 215)
CTkLabel(root, text="Father's Education Level (1-4)", font = ('Arial',18)).place(x=350, y = 215)
CTkLabel(root, text="Weekly Study time (1-4)", font = ('Arial',18)).place(x=70, y=285)
CTkLabel(root, text="Number of Past Class Failures (0-4)",font = ('Arial',18)).place(x=350, y=285)
CTkLabel(root, text="Quality of Family Relations (1-5)", font = ('Arial',18)).place(x=70, y=355)
CTkLabel(root, text="Weekend Alcohol Consumption (1-5)", font = ('Arial',18)).place(x=350, y=355)
CTkLabel(root, text="Current Health Status (1-5)", font = ('Arial',18)).place(x=70, y=425)
CTkLabel(root, text="Current Age (15-22)", font = ('Arial',18)).place(x=350, y=425)


##Input boxes for each variable
Input_G1 = CTkEntry(root, width = 100)
Input_G1.place(x = 70, y=170)
Input_G2 = CTkEntry(root, width = 100)
Input_G2.place(x = 350, y = 170)
Input_Medu = CTkEntry(root, width = 100)
Input_Medu.place(x = 70, y = 240)
Input_Fedu = CTkEntry(root, width = 100)
Input_Fedu.place(x = 350, y = 240)
Input_Study = CTkEntry(root, width = 100)
Input_Study.place(x = 70, y = 310)
Input_Fail = CTkEntry(root, width = 100)
Input_Fail.place(x = 350, y = 310)
Input_Famrel = CTkEntry(root, width = 100)
Input_Famrel.place(x = 70, y = 380)
Input_Walc = CTkEntry(root, width = 100)
Input_Walc.place(x = 350, y = 380)
Input_Health = CTkEntry(root, width = 100)
Input_Health.place(x = 70, y = 450)
Input_Age = CTkEntry(root, width = 100)
Input_Age.place(x=350, y=450)


Output_text = CTkTextbox(root, font=("Arial",50), height=5, wrap=WORD, padx=5, pady=5, width=245, corner_radius = 20, state="disabled")
Output_text.place(x=772, y=130)

button = CTkButton(root, text = "Generate Prediction", font=("Arial",20), fg_color = "purple",command=Predict, corner_radius = 10)
button.place(x=795, y = 250)


root.mainloop()

