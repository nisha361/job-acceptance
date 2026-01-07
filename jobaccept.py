import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("HR_Job_Placement_Dataset (1).csv")

# Basic checks
print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())
num_cols = df.select_dtypes(include=["int64","float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
Encode Target Variable
Copy code
Python
df["status"] = df["status"].map({"Placed":1, "Not Placed":0})
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in cat_cols:
    if col != "status":
        df[col] = le.fit_transform(df[col])
        from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
import matplotlib.pyplot as plt
import seaborn as sns

# Placement distribution
sns.countplot(x="status", data=df)
plt.title("Job Acceptance Distribution")
plt.show()

# Skills match vs acceptance
sns.boxplot(x="status", y="skills_match_percentage", data=df)
plt.title("Skills Match vs Job Acceptance")
plt.show()

# Company tier impact
sns.barplot(x="company_tier", y="status", data=df)
plt.title("Company Tier vs Acceptance Rate")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()
df["experience_level"] = pd.cut(
    df["years_of_experience"],
    bins=[-1,1,5,30],
    labels=[0,1,2] # Fresher, Junior, Senior
)
from sklearn.model_selection import train_test_split

X = df.drop("status", axis=1)
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
importances = pd.Series(rf.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(10))
