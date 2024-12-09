# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Housing.csv')

# Explore the dataset
print(data.head())
print(data.info())

# Select relevant features and target
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad']]  # Adjust as per dataset columns
y = data['price']

# Preprocess the data
X = pd.get_dummies(X, drop_first=True)  # Convert categorical features to numerics
y = y.fillna(y.mean())  # Fill missing values in target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-Squared:", r2)

# Visualise predictions vs actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()