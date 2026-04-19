import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Display basic info
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ----------------------------
# Data Cleaning
# ----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# ----------------------------
# Data Analysis
# ----------------------------

# Average scores by gender
gender_scores = df.groupby("gender")[["math score", "reading score", "writing score"]].mean()
print("\nAverage Scores by Gender:")
print(gender_scores)

# Average scores by parental education
parent_scores = df.groupby("parental level of education")[["math score", "reading score", "writing score"]].mean()
print("\nScores by Parental Education:")
print(parent_scores)

# ----------------------------
# Visualization
# ----------------------------

# Set style
sns.set(style="whitegrid")

# Distribution plots
plt.figure(figsize=(8,5))
sns.histplot(df["math score"], kde=True)
plt.title("Math Score Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["reading score"], kde=True)
plt.title("Reading Score Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["writing score"], kde=True)
plt.title("Writing Score Distribution")
plt.show()

# Bar plot (Gender comparison)
gender_scores.plot(kind="bar", figsize=(8,5))
plt.title("Average Scores by Gender")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df[["math score", "reading score", "writing score"]].corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Between Scores")
plt.show()

# ----------------------------
# Insights (Printed)
# ----------------------------
print("\nInsights:")
print("- Reading and Writing scores are usually strongly correlated.")
print("- Gender differences may exist in subject performance.")
print("- Parental education level can influence student scores.")

print("\nAnalysis completed successfully!")