import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


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
logisticModel = LogisticRegression(random_state=0)
logisticModel.fit(X_train_scaled, y_train)
y_pred1 = logisticModel.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred1)
print(f"Accuracy for logistic regression is: {accuracy}")

#correlation matrix
cmatrix = confusion_matrix(y_test, y_pred1)
plt.figure(figsize=(6, 4))
sns.heatmap(cmatrix, annot=True, cmap='coolwarm', xticklabels=['Alive', 'Dead'], yticklabels=['x', 'y'])
plt.title('Correlation Heatmap between x and y')
plt.show()

#evaluation metrics
f1_macro = f1_score(y_test, y_pred1, average='macro')
precision = precision_score(y_test, y_pred1)
recall = recall_score(y_test, y_pred1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Macro Score']
scores = [accuracy, precision, recall, f1_macro]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, scores, color='skyblue', edgecolor='black')
plt.ylim(0, 1.1)
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
