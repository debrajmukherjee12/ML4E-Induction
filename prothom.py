import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset
# Make sure 'nyc-rolling-sales.csv' is in the exact same folder as your script/notebook
try:
    df = pd.read_csv('nyc-rolling-sales.csv')
except FileNotFoundError:
    print("Error: Could not find the file. Please ensure it's named 'nyc-rolling-sales.csv' and is in the same directory.")
    # Exit or handle the error gracefully depending on your setup

if 'df' in locals():
    print("--- Initial Observations ---")
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")

    # 2. Finding and Removing Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")
    df = df.drop_duplicates()
    print("Duplicates removed.")

    # 3. Finding and Cleaning Null Points
    # The NYC dataset often has non-numeric characters (like '-' or '$') in numeric columns
    cols_to_clean = ['SALE PRICE', 'GROSS SQUARE FEET', 'LAND SQUARE FEET']
    for col in cols_to_clean:
        if col in df.columns:
            # Force string conversion, remove unwanted characters, then force back to numeric (creating NaNs for bad data)
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[\$,-]', '', regex=True).str.strip(), errors='coerce')

    null_counts = df.isnull().sum()
    print("\nNull values per column:\n", null_counts[null_counts > 0])

    # Plotting the null values
    plt.figure(figsize=(10, 5))
    null_counts[null_counts > 0].plot(kind='bar', color='salmon')
    plt.title('Count of Null Values per Column')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 4. Pre-processing Nulls
    # We MUST drop rows where the target variable (SALE PRICE) is missing
    df = df.dropna(subset=['SALE PRICE'])

    # Fill missing values in all other numeric columns with the median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    print("\nData shape after cleaning:", df.shape)

    # 5. Correlation & Feature Selection
    corr_matrix = df[numeric_cols].corr()
    
    # Isolate correlation with SALE PRICE, drop SALE PRICE itself from the list, and sort
    target_corr = corr_matrix['SALE PRICE'].drop('SALE PRICE').sort_values(ascending=False)

    # Plotting correlations
    plt.figure(figsize=(10, 6))
    target_corr.plot(kind='bar', color='skyblue')
    plt.title('Correlation of Features with SALE PRICE')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Selecting the top 4 features based on absolute correlation strength
    top_4_features = target_corr.abs().nlargest(4).index.tolist()
    print("\nSelected 4 Features for prediction based on correlation:", top_4_features)

    # 6. Building Linear Regression from Scratch
    # Prepare Data
    X = df[top_4_features].values
    y = df['SALE PRICE'].values.reshape(-1, 1)

    # Feature Scaling (Standardization)
    def standardize(X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        # Adding a tiny epsilon to prevent division by zero
        return (X - mean) / (std + 1e-8), mean, std

    X_scaled, X_mean, X_std = standardize(X)

    # Custom Linear Regression Class
    class LinearRegressionCustom:
        def __init__(self, learning_rate=0.01, iterations=1000):
            self.lr = learning_rate
            self.iters = iterations
            self.weights = None
            self.bias = None
            self.loss_history = []
            
        def fit(self, X, y):
            n_samples, n_features = X.shape
            
            # Initialize weights and bias
            self.weights = np.zeros((n_features, 1))
            self.bias = 0
            
            # Gradient Descent
            for i in range(self.iters):
                # Hypothesis equation: y = Xw + b
                y_pred = np.dot(X, self.weights) + self.bias
                
                # Gradients calculation
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
                db = (1 / n_samples) * np.sum(y_pred - y)
                
                # Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                
                # Calculate Mean Squared Error (MSE) loss for tracking
                loss = (1 / n_samples) * np.sum((y_pred - y) ** 2)
                self.loss_history.append(loss)
                
        def predict(self, X):
            return np.dot(X, self.weights) + self.bias

    # Train the custom model
    print("\nTraining Custom Linear Regression Model...")
    model = LinearRegressionCustom(learning_rate=0.1, iterations=500)
    model.fit(X_scaled, y)

    print("Model Training Complete.")
    print("Final Weights:", model.weights.flatten())
    print("Final Bias:", model.bias)

    # Plotting the loss to visualize Gradient Descent convergence
    plt.figure(figsize=(8, 4))
    plt.plot(model.loss_history, color='green')
    plt.title('Cost Reduction over Iterations (Gradient Descent)')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error Loss')
    plt.show()