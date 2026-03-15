import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data and initial observation
print("--- Loading Dataset ---")
try:
    df = pd.read_csv('diabetes.csv')
    print(f"Dataset loaded successfully. Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")
except FileNotFoundError:
    print("Error: Please ensure 'diabetes.csv' is in the same directory as this notebook.")

if 'df' in locals():
    # Check for actual duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Number of exact duplicate rows: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print("Duplicates dropped.\n")

    # 2. Pre-processing and null value handling
    # Domain Knowledge: In this dataset, values like Blood Pressure, Glucose, BMI, etc., cannot be 0.
    # A value of 0 actually indicates a missing value (null point).
    cols_with_hidden_nulls = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Replace 0s with actual NaNs so pandas can count them properly
    df[cols_with_hidden_nulls] = df[cols_with_hidden_nulls].replace(0, np.nan)
    
    null_counts = df.isnull().sum()
    print("\nActual missing values per feature (after replacing 0s):")
    print(null_counts[null_counts > 0])

    # Visualizing the missing values
    plt.figure(figsize=(8, 4))
    null_counts[null_counts > 0].plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Count of Missing Values (Originally 0s) per Feature')
    plt.ylabel('Number of Nulls')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Imputation: Fill missing values with the median of each respective column
    # Median is better than mean here to avoid skewing from outliers (like extreme insulin levels)
    for col in cols_with_hidden_nulls:
        df[col] = df[col].fillna(df[col].median())
        
    print("Missing values successfully imputed with column medians.\n")

    # 3. Correlation and feature selection
    # Calculate correlation of all features specifically with the 'Outcome' (Diabetes label)
    correlations = df.corr()['Outcome'].drop('Outcome').sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    correlations.plot(kind='bar', color='teal')
    plt.title('Feature Correlation with Diabetes Outcome')
    plt.ylabel('Correlation Coefficient')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Select top 3 features with the highest absolute correlation
    top_3_features = correlations.abs().nlargest(3).index.tolist()
    print(f"Top 3 selected features based on correlation strength: {top_3_features}\n")

    # 4. Logistic regression from scratch
    # Prepare X (features) and y (target)
    X = df[top_3_features].values
    y = df['Outcome'].values.reshape(-1, 1)

    # Standardization (Crucial for Logistic Regression to converge)
    def standardize_features(X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / (std + 1e-8)

    X_scaled = standardize_features(X)

    # Custom Logistic Regression Class
    class LogisticRegressionCustom:
        def __init__(self, learning_rate=0.01, iterations=2000):
            self.lr = learning_rate
            self.iters = iterations
            self.weights = None
            self.bias = None
            self.cost_history = []
            
        def sigmoid(self, z):
            # Sigmoid activation function to map predictions between 0 and 1
            # Using np.clip to prevent overflow errors in exp
            z = np.clip(z, -250, 250)
            return 1 / (1 + np.exp(-z))
            
        def fit(self, X, y):
            n_samples, n_features = X.shape
            # Initialize parameters to zeros
            self.weights = np.zeros((n_features, 1))
            self.bias = 0
            
            # Gradient Descent Optimization
            for i in range(self.iters):
                # Linear combination
                linear_model = np.dot(X, self.weights) + self.bias
                
                # Apply sigmoid activation
                y_predicted = self.sigmoid(linear_model)
                
                # Calculate gradients
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
                db = (1 / n_samples) * np.sum(y_predicted - y)
                
                # Update weights and bias
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                
                # Calculate Log Loss (Binary Cross-Entropy) for tracking
                # Adding epsilon to log to avoid log(0)
                epsilon = 1e-9
                cost = -(1 / n_samples) * np.sum(y * np.log(y_predicted + epsilon) + (1 - y) * np.log(1 - y_predicted + epsilon))
                self.cost_history.append(cost)
                
        def predict_prob(self, X):
            linear_model = np.dot(X, self.weights) + self.bias
            return self.sigmoid(linear_model)
            
        def predict(self, X, threshold=0.5):
            probabilities = self.predict_prob(X)
            return (probabilities >= threshold).astype(int)

    # Train the model
    print("Training Custom Logistic Regression Model...")
    clf = LogisticRegressionCustom(learning_rate=0.1, iterations=1000)
    clf.fit(X_scaled, y)
    print("Training Complete.")

    # Show the learning curve
    plt.figure(figsize=(8, 4))
    plt.plot(clf.cost_history, color='purple')
    plt.title('Binary Cross-Entropy Loss over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost (Loss)')
    plt.show()

    # Quick evaluation on the training set to prove it works
    predictions = clf.predict(X_scaled)
    accuracy = np.mean(predictions == y) * 100
    print(f"Model Training Accuracy (on selected 3 features): {accuracy:.2f}%")