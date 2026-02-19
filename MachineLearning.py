import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
heart_data = pd.read_csv('heart.csv')

# Split features & target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)  # Increased iterations
model.fit(X_train, Y_train)

# Accuracy evaluation
train_accuracy = accuracy_score(model.predict(X_train), Y_train)
test_accuracy = accuracy_score(model.predict(X_test), Y_test)
print(f'Accuracy on Training data: {train_accuracy:.2f}')
print(f'Accuracy on Test data: {test_accuracy:.2f}')

# Prediction on new data
input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# Convert input_data into a DataFrame with feature names to avoid warnings
input_data_df = pd.DataFrame([input_data], columns=X.columns)

# Scale the input data
input_data_scaled = scaler.transform(input_data_df)

# Make prediction
prediction = model.predict(input_data_scaled)

# Display result
result = "The Person has Heart Disease" if prediction[0] == 1 else "The Person does not have Heart Disease"
print(result)