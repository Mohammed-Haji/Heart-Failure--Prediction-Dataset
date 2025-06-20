
# 1. Import necessary libraries
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 2. Load the preprocessed data that you saved earlier
data_dir = 'processed_data'
try:
    with open(os.path.join(data_dir, 'X_train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(data_dir, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)
except FileNotFoundError:
    print("Error: Training data not found.")
    exit()

# 3. Initialize and train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# 4. Save the trained model to a 'models' folder
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'heart_disease_model.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Recipe written! The file 'train_model.py' has been created successfully.")
