import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#check filepath for where you store filtered dataset
data = pd.read_csv("./Final_Project/Dataset/filtered_data.csv", delimiter=",")

#separating features
mts_ChronicKidney = data.iloc[:,0]
arf = data.iloc[:,1]
creatinine = data.iloc[:,2]
sodium = data.iloc[:,3]
lvef = data.iloc[:,4]
features = [mts_ChronicKidney, arf, creatinine, sodium, lvef]
feature_names = ["mts_ChronicKidney", "arf", "creatinine", "sodium", "lvef"]

'''
# plotting histograms of features and desc stats
for feature, name in zip(features, feature_names):
    plt.subplot(2, 3, feature_names.index(name) + 1)
    plt.hist(feature, bins='auto', color='skyblue', edgecolor='black')
    plt.xlabel("Bin")
    plt.ylabel("Frequency Count")
    plt.title(name)
    # descriptive stats
    low = feature.min()
    high = feature.max()
    mean = np.mean(feature)
    median = np.median(feature)
    freq = (feature != 0).sum()
    print(f"{name} descriptive stats: \n lowest val: {low}, highest val: {high}, mean: {mean}, median: {median}, freq: {freq}")
plt.tight_layout()
plt.show()
'''
plt.hist(creatinine, bins='auto', color='skyblue', edgecolor='black')
plt.xlabel("Bin")
plt.ylabel("Frequency Count")
plt.title("creatinine.enzymatic.method")
plt.show()

plt.hist(lvef, bins='auto', color='skyblue', edgecolor='black')
plt.xlabel("Bin")
plt.ylabel("Frequency Count")
plt.title("LVEF")
plt.show()
