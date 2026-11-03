Lab # 6 Activity
Data Preprocessing for Machine Learning (Python)
Course: Programming for AI
Lab Duration: 1 hour 40 minutes
Lab Objectives
After completing this lab, students will be able to:
•	Perform feature selection using correlation
•	Detect and remove outliers
•	Handle imbalanced datasets
•	Apply categorical encoding
•	Perform data splitting
•	Upload a complete ML preprocessing project to GitHub
________________________________________
Dataset
Students will use the Bank Customer Churn Dataset
Example columns:
CustomerId
CreditScore
Geography
Gender
Age
Balance
NumOfProducts
IsActiveMember
EstimatedSalary
Exited
Exited → Target variable (1 = left bank, 0 = stayed)
________________________________________
Lab Setup (10 minutes)
Step 1: Install Required Libraries
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn pyod
________________________________________
Step 2: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
________________________________________
Step 3: Load Dataset
df = pd.read_csv("bank_churn.csv")

df.head()
________________________________________
Part 1 — Feature Selection (20 minutes)
Feature selection identifies important variables that influence model performance.
________________________________________
Task 1 (Guided)
Generate a correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()
Question
1.	Which features are highly correlated?
________________________________________
Task 2 (Guided)
Find correlation with the target variable
df.corr(numeric_only=True)['balance'].sort_values()
________________________________________
Task 3 (Student Exercise)
Select features whose correlation is:
> 0.1  OR  < -0.1
Write code to display these features.
________________________________________
Task 4 (Challenge)
Remove highly correlated redundant features.
Example:
df = df.drop(columns=['CustomerId'])
Explain why removing redundant features prevents overfitting.
________________________________________
Part 2 — Handling Imbalanced Data (20 minutes)
Imbalanced data occurs when one class dominates the dataset.
________________________________________
Task 1 (Guided)
Check class distribution.
df['Exited'].value_counts()
________________________________________
Task 2 (Guided)
Convert to percentage.
df['Exited'].value_counts(normalize=True) * 100
Question:
Is the dataset balanced or imbalanced?
________________________________________
Task 3 (Guided) — Upsampling
def upsample(features, target, repeat):

    features_0 = features[target == 0]
    features_1 = features[target == 1]

    target_0 = target[target == 0]
    target_1 = target[target == 1]

    features_up = pd.concat([features_0] + [features_1]*repeat)
    target_up = pd.concat([target_0] + [target_1]*repeat)

    features_up, target_up = shuffle(features_up, target_up, random_state=42)

    return features_up, target_up
________________________________________
Apply upsampling:
features = df.drop("Exited", axis=1)
target = df["Exited"]

features_up, target_up = upsample(features, target, 3)
________________________________________
Task 4 (Student Exercise)
Print the class distribution after upsampling.
________________________________________
Task 5 (Challenge)
Implement downsampling function.
________________________________________
Part 3 — Encoding Categorical Variables (20 minutes)
Machine learning algorithms require numerical input.
Categorical features:
Geography
Gender
________________________________________
Task 1 (Guided)
Identify categorical columns.
df.select_dtypes(include='object')
________________________________________
Task 2 (Guided)
Apply One-Hot Encoding
df = pd.get_dummies(df, columns=['Geography','Gender'], drop_first=True)
________________________________________
Task 3 (Exercise)
Print the dataset again.
df.head()
Question:
How many new columns were created?
________________________________________
Task 4 (Challenge)
Explain the difference between:
•	Ordinal Encoding
•	One-hot Encoding
________________________________________
Part 4 — Data Splitting (15 minutes)
Data splitting divides the dataset into:
•	Training set
•	Validation set
•	Test set
________________________________________
Task 1 (Guided)
Split dataset into train + test
train_valid, test = train_test_split(df, test_size=0.2, random_state=42)
________________________________________
Task 2 (Guided)
Create train and validation sets
train, valid = train_test_split(train_valid, test_size=0.25, random_state=42)
________________________________________
Task 3 (Exercise)
Create features and target.
X_train = train.drop("Exited", axis=1)
y_train = train["Exited"]

X_valid = valid.drop("Exited", axis=1)
y_valid = valid["Exited"]

X_test = test.drop("Exited", axis=1)
y_test = test["Exited"]
________________________________________
Task 4 (Challenge)
Print shapes of all datasets.
Expected output example:
Training set
Validation set
Test set
________________________________________
Final Task — Upload Project to GitHub (15 minutes)
Students must upload their complete preprocessing pipeline to GitHub.
________________________________________
Step 1 — Create Project Folder
ML_Preprocessing_Lab
Inside folder:
dataset/
bank_churn.csv

notebooks/
preprocessing_lab.ipynb

scripts/
preprocessing.py

README.md
________________________________________
Step 2 — Initialize Git
git init
________________________________________
Step 3 — Add Files
git add .
________________________________________
Step 4 — Commit
git commit -m "ML preprocessing lab implementation"
________________________________________
Step 5 — Create GitHub Repository
Go to:
https://github.com
Create repository:
ml-preprocessing-lab
________________________________________
Step 6 — Connect Local Repo
git remote add origin https://github.com/username/ml-preprocessing-lab.git
________________________________________
Step 7 — Push Code
git push -u origin main
________________________________________
Deliverables (Submission)
Students must submit:
1.	GitHub repository link
2.	Python notebook (.ipynb)
3.	Screenshots of results
________________________________________
Lab Reflection Questions
1.	Why is feature selection important?
2.	What problems arise with imbalanced datasets?
3.	Why is one-hot encoding preferred for nominal variables?
4.	Why should test data never be used during training?
________________________________________

