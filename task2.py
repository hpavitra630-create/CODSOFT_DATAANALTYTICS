import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("Cleaned_Titanic.csv")

print("========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

print("\n========== CORRELATION ==========")
print(df.corr(numeric_only=True))

# Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class Distribution")
plt.show()

# Age Distribution
plt.figure(figsize=(6,4))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Outlier Detection
plt.figure(figsize=(6,4))
sns.boxplot(x=df["Age"])
plt.title("Age Box Plot")
plt.show()

# Scatter Plot
plt.figure(figsize=(6,4))
sns.scatterplot(x="Age", y="Fare", data=df)
plt.title("Age vs Fare")
plt.show()

print("\n========== BUSINESS INSIGHTS ==========")

print("\nPassenger Class Distribution:")
print(df["Pclass"].value_counts())

print("\nSurvival Percentage:")
print(df["Survived"].value_counts(normalize=True) * 100)

print("\nAverage Age:", df["Age"].mean())

print("Average Fare:", df["Fare"].mean())

print("\nEDA Completed Successfully!")