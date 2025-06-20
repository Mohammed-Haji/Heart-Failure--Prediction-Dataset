
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Load the dataset
data = pd.read_csv('heart.csv')

# Use the correct column name 'HeartDisease'
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# One-hot encode categorical features to handle text data
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Align columns - this is important after get_dummies
train_cols = X_train.columns
test_cols = X_test.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0
X_test = X_test[train_cols]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- THE FIX IS HERE ---
# Save ALL the necessary files, including the scaler
os.makedirs('processed_data', exist_ok=True)

with open('processed_data/X_train.pkl', 'wb') as f:
    pickle.dump(X_train_scaled, f)
with open('processed_data/X_test.pkl', 'wb') as f:
    pickle.dump(X_test_scaled, f)
with open('processed_data/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('processed_data/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

# This line saves the scaler object
with open('processed_data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# This saves the column order after dummy creation
with open('processed_data/train_cols.pkl', 'wb') as f:
    pickle.dump(train_cols, f)


print(f"Success: 'preprocess.py' has been updated and now saves the scaler and columns.")
