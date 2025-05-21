# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and separate features (R&D, Admin, Marketing) and target (Profit).
2. Scale features and target using StandardScaler for better gradient descent performance.
3. Train the model using gradient descent to minimize the cost function.
4. Use the trained model to make predictions for new input data.
5. Inverse scale the predicted profit to get the final output in the original scale.
6. 
## Program:
```

# Program to implement the linear regression using gradient descent.
# Developed by: SHAHIN J
# Register Number: 212223040190

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Linear Regression using Gradient Descent
def linear_regression(features, target, learning_rate=0.01, num_iters=1000):
    X = np.c_[np.ones(len(features)), features]  
    theta = np.zeros((X.shape[1], 1))  
    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - target
        theta -= (learning_rate / len(features)) * X.T.dot(errors)
    return theta

# Reading the dataset
data = pd.read_csv('/content/50_Startups (2).csv')

# Extracting input features (R&D Spend, Administration, Marketing Spend)
features = data[['R&D Spend', 'Administration', 'Marketing Spend']].values

# Converting to float
features = features.astype(float)

# Extracting target output (Profit)
profit = data[['Profit']].values

# Feature scaling
feature_scaler = StandardScaler()
profit_scaler = StandardScaler()

scaled_features = feature_scaler.fit_transform(features)
scaled_profit = profit_scaler.fit_transform(profit)

# Training the model
theta = linear_regression(scaled_features, scaled_profit)

# Predicting for new input
new_startup_data = np.array([[165349.2, 136897.8, 471784.1]])  
scaled_new_data = feature_scaler.transform(new_startup_data)

# Making prediction
scaled_input = np.append([[1]], scaled_new_data, axis=1) 
scaled_prediction = scaled_input.dot(theta)

# Inverse transform to get actual predicted profit
predicted_profit = profit_scaler.inverse_transform(scaled_prediction)

# Output the predicted profit
print(f"Predicted Profit for input {new_startup_data[0]}: ${predicted_profit[0][0]:.2f}")

```

## Output:
![Screenshot 2025-04-07 153327](https://github.com/user-attachments/assets/b640132d-364b-4a65-b8da-98b93a269a17)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
