import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#check filepath for where you store filtered dataset
data = pd.read_csv("./Final_Project/Dataset/filtered_data.csv", delimiter=",")
X = data[['creatinine.enzymatic.method', 'LVEF']].values  #features
y = data['death'].values                #target

#splitting data 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# logistic regression classifier
logisticModel = LogisticRegression()
logisticModel.fit(X_train_scaled, y_train)
y_pred1 = logisticModel.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred1)
print(f"Accuracy for logistic regression is: {accuracy}")


