import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

df = pd.read_csv("Bank_Customer_Churn_Prediction.csv")

df.head()
print(df)

'''plt.figure(figsize=(10,8))
#sns.heatmap(df.corr(numeric_only=True), annot=True)
#sns.heatmap(df.corr(numeric_only=True)['churn'].sort_values(), annot = True)
plt.title("Correlation Heatmap")
plt.show()'''

list1 = df.corr(numeric_only=True)['balance'].sort_values()
print(list1)
print("###########################")
selected_features = list1[(list1 > 0.1) | (list1 < -0.1)]
print(selected_features)

print(df['churn'].value_counts())



def upsample(features, target, repeat):

    features_0 = features[target == 0]
    features_1 = features[target == 1]

    target_0 = target[target == 0]
    target_1 = target[target == 1]

    features_up = pd.concat([features_0] + [features_1]*repeat)
    target_up = pd.concat([target_0] + [target_1]*repeat)

    features_up, target_up = shuffle(features_up, target_up, random_state=42)

    return features_up, target_up

features = df.drop("churn", axis=1)
target = df["churn"]

features_up, target_up = upsample(features, target, 3)

print(target_up.value_counts())