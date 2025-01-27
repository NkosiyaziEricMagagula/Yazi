# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# Function to create and explore the custom dataset
def create_and_explore_data():
    # Create a simple custom dataset including the 'name' column
    data = pd.DataFrame({
        'name': ['John', 'Jane', 'Alice', 'Bob', 'Charlie', 'David'],
        'pclass': [1, 2, 1, 3, 2, 1],
        'sex': ['male', 'female', 'female', 'male', 'male', 'male'],
        'age': [22, 24, 35, 26, 40, 50],
        'fare': [7.25, 71.28, 8.05, 8.75, 11.13, 19.20],
        'embarked': ['S', 'C', 'Q', 'S', 'C', 'Q'],
        'survived': [0, 1, 1, 0, 1, 0]
    })

    print(data.head())  # Display the first few rows
    print(data.describe())  # Basic statistics
    print(data.info())  # Data structure and types
    return data

# Function to clean and preprocess the data
def preprocess_data(data):
    print("Missing values before cleaning:")
    print(data.isnull().sum())

    # Handle missing values
    data['age'] = data['age'].fillna(data['age'].median())  # Fill missing ages with median
    data.dropna(subset=['embarked'], inplace=True)  # Drop rows with missing 'embarked'

    print("Missing values after cleaning:")
    print(data.isnull().sum())

    # Encode categorical variables
    data['sex'] = data['sex'].map({'male': 0, 'female': 1})  # Binary encoding for 'sex'

    # One-hot encoding for 'embarked'
    data = pd.get_dummies(data, columns=['embarked'], drop_first=True)  # One-hot encoding

    return data

# Function to prepare features and target
def prepare_features_target(data):
    features = data[['pclass', 'sex', 'age', 'fare', 'embarked_Q', 'embarked_S']]
    target = data['survived']
    return features, target

# Function to train a logistic regression model
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=200)  # Increased max_iter for convergence
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy Score: {accuracy:.2f}")
    print("Confusion Matrix:\n", conf_matrix)
    return accuracy, conf_matrix

# Function to plot visualizations
def plot_visualizations(data, conf_matrix, y_test, model, X_test):
    # Age distribution of survivors and non-survivors
    plt.figure(figsize=(12, 6))
    sns.histplot(data[data['survived'] == 1]['age'], bins=30, color='blue', label='Survived', kde=True)
    sns.histplot(data[data['survived'] == 0]['age'], bins=30, color='red', label='Not Survived', kde=True)
    plt.title('Age Distribution of Survivors and Non-Survivors')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Count plot for passenger class
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x='pclass', hue='survived')
    plt.title('Survival Count by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Count')
    plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
    plt.show()

    # Confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC curve
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

    print(f"ROC AUC Score: {roc_auc:.2f}")

# Main function to orchestrate the workflow
def main():
    # Step 1: Create and explore the custom dataset
    data = create_and_explore_data()

    # Step 2: Clean and preprocess the data
    data = preprocess_data(data)

    # Step 3: Prepare features and target
    features, target = prepare_features_target(data)

    # Step 4: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Step 5: Train the logistic regression model
    model = train_logistic_regression(X_train, y_train)

    # Step 6: Evaluate the model
    accuracy, conf_matrix = evaluate_model(model, X_test, y_test)

    # Step 7: Plot visualizations
    plot_visualizations(data, conf_matrix, y_test, model, X_test)

# Run the main function
if __name__ == "__main__":
    main()