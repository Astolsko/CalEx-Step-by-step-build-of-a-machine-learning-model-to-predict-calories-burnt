import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.model_selection import train_test_split  # ML Model
from xgboost import XGBRegressor
from sklearn import metrics

#Importing data
calories_data = pd.read_csv("calories.csv")
exercise_data = pd.read_csv("exercise.csv")

#Combining the datasets
combined_data = pd.concat([exercise_data, calories_data['Calories']], axis=1)
combined_data.replace({'Gender':{'male':0,'female':1}},inplace=True)
#Rearranging
X = combined_data.drop(columns=['User_ID','Calories'], axis=1)
Y = combined_data['Calories']
# Spliting into train and test
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2) 
#test_size = 20% of total data so 80 % is training data.
print("Training Data Shape:")
print(X.shape,X_train.shape,X_test.shape)

#loading the model
model_3 = XGBRegressor()
#training the model with X_train
model_3.fit(X_train,Y_train)
XGB_prediction = model_3.predict(X_test)

#Evaluation Metrics
MAE = metrics.mean_absolute_error(Y_test, XGB_prediction)
print("Mean Absolute Error = ",MAE)
rss = metrics.r2_score(Y_test,XGB_prediction )
print("R-Squared Score = ", rss)
mse = metrics.mean_squared_error(Y_test,XGB_prediction)
print("Mean Squared Error = ", mse)
rmse = metrics.mean_squared_error(Y_test, XGB_prediction, squared=False)
print("Root Mean Squared Error = ", rmse)

# User input
gender = input("Enter your gender : ")
if gender.lower() == "male":
    gender = 0
elif gender.lower() == "female":
    gender = 1
age = int(input("Enter your age : "))
height = float(input("Enter your height : "))
weight = float(input("Enter your weight : "))
exercise_duration = float(input("Enter your exercise duration : "))
heart_rate = float(input("Enter your heart rate : "))
body_temp = float(input("Enter your body temperature : "))
# Create a NumPy array from the user input
user_input = np.array([[gender, age, height, weight, exercise_duration, heart_rate, body_temp]],dtype=np.float32)
XGB_prediction = model_3.predict(user_input)
print("Calories burnt : ",XGB_prediction)