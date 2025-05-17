import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load Excel file
file_path = 'natural_disaster_data.xlsx'
df = pd.read_excel(file_path)

# Filter only Earthquakes and Tsunamis
relevant_df = df[df['DisasterType'].isin(['Earthquake', 'Tsunami'])].copy()

# Create label: 1 if Tsunami, else 0
relevant_df['WillCauseTsunami'] = (relevant_df['DisasterType'] == 'Tsunami').astype(int)

# Select features and drop missing values
features = ['Magnitude', 'Duration(Hours)', 'AffectedPopulation', 'ResponseTime(Hours)', 'DamageCost(USD Millions)']
relevant_df = relevant_df[features + ['WillCauseTsunami']].dropna()

# Features and label
X = relevant_df[features]
y = relevant_df['WillCauseTsunami']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature Importance Plot
importances = rf.feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Magnitude Distribution
X_plot = relevant_df.copy()
X_plot['DisasterType'] = ['Tsunami' if x == 1 else 'Earthquake' for x in X_plot['WillCauseTsunami']]
plt.figure(figsize=(8, 5))
sns.histplot(data=X_plot, x='Magnitude', hue='DisasterType', kde=True, bins=10)
plt.title("Magnitude Distribution: Earthquakes vs Tsunamis")
plt.xlabel("Magnitude")
plt.ylabel("Frequency")
plt.show()