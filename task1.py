import pandas as pd

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv"
)

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())

if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column if it exists
if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

# Save cleaned dataset
df.to_csv("Cleaned_Titanic.csv", index=False)

print("\nData Cleaning Completed Successfully!")
print("Cleaned dataset saved as Cleaned_Titanic.csv")