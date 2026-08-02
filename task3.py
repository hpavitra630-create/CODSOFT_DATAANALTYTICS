import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
df = pd.read_csv("Titanic-Dataset.csv")
print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

print(df.head())

plt.figure(figsize=(6,5))
sns.countplot(x='Survived', data=df, palette='Set2')
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")
plt.show()

plt.figure(figsize=(6,5))
sns.countplot(x='Pclass', data=df, palette='viridis')
plt.title("Passenger Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,5))
df['Sex'].value_counts().plot.pie(
    autopct='%1.1f%%',
    startangle=90,
    colors=['skyblue','pink']
)
plt.title("Gender Distribution")
plt.ylabel("")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['Age'].dropna(), bins=30, kde=True, color='green')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['Fare'], bins=30, kde=True, color='orange')
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,5))
sns.countplot(x='Sex', hue='Survived', data=df, palette='Set1')
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Survived")
plt.show()

plt.figure(figsize=(6,5))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='Dark2')
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.legend(title="Survived")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(
    x='Age',
    y='Fare',
    hue='Survived',
    data=df,
    palette='coolwarm'
)
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10,))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title("Correlation Heatmap")
plt.show()

print("=" * 50)
print("Task 3 Completed Successsfully")
print("=" * 50)