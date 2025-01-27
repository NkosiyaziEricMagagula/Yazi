# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Step 2: Load and Explore Data
# Load the Titanic dataset
data = sns.load_dataset('titanic')
print(data.head())  # Display the first few rows

# Explore basic statistics and data structure
print(data.describe())
print(data.info())

# Step 3: Data Cleaning and Preprocessing
# Check for missing values
print("Missing values before cleaning:")
print(data.isnull().sum())

# Handle missing values
data['age'].fillna(data['age'].median(), inplace=True)  # Fill missing ages with the median
data.dropna(subset=['embarked'], inplace=True)  # Drop rows with missing 'embarked'

# Encode categorical variables
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['embarked'], drop_first=True)

# Select relevant features for prediction
features = data[['pclass', 'sex', 'age', 'fare', 'embarked_Q', 'embarked_S']]
target = data['survived']

# Step 4: Train Logistic Regression Model
# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Fit a logistic regression model
model = LogisticRegression(max_iter=200)  # Increased max_iter to ensure convergence
model.fit(X_train, y_train)

# Evaluate the model's performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy Score: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)

# Step 5: Visualization
# Feature Analysis
# Plot the distribution of age for survivors and non-survivors
plt.figure(figsize=(12, 6))
sns.histplot(data[data['survived'] == 1]['age'], bins=30, color='blue', label='Survived', kde=True)
sns.histplot(data[data['survived'] == 0]['age'], bins=30, color='red', label='Not Survived', kde=True)
plt.title('Age Distribution of Survivors and Non-Survivors')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Create a count plot for the pclass column
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='pclass', hue='survived')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.show()

# Model Results
# Plot a confusion matrix heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Create a ROC curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Insights and Interpretations
# Print insights
print(f"ROC AUC Score: {roc_auc}")
