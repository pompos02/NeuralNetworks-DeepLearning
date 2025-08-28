import numpy as np
import cupy as cp
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10 # type: ignore
from tqdm import tqdm
import random
cp.random.seed(42)  # Replace 42 with any fixed integer
np.random.seed(42)
random.seed(42)

# Load the dataset
(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

# Flatten label arrays
y_train_full = y_train_full.flatten()
y_test_full = y_test_full.flatten()

# Define the classes
class_map = {5: 'dog', 3: 'cat'}

# Filter training data
train_filter = np.isin(y_train_full, list(class_map.keys()))
x_train = x_train_full[train_filter]
y_train = y_train_full[train_filter]

# Filter test data
test_filter = np.isin(y_test_full, list(class_map.keys()))
x_test = x_test_full[test_filter]
y_test = y_test_full[test_filter]

# Map labels to +1 and -1
label_map = {5: 1, 3: -1}
y_train = np.vectorize(label_map.get)(y_train)
y_test = np.vectorize(label_map.get)(y_test)

# Flatten the images
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)




x_train = cp.array(x_train)
y_train = cp.array(y_train)
x_test = cp.array(x_test)
y_test = cp.array(y_test)


from sklearn.svm import SVC

x_train_np = cp.asnumpy(x_train)
y_train_np = cp.asnumpy(y_train)
x_test_np = cp.asnumpy(x_test)
y_test_np = cp.asnumpy(y_test)

clf = SVC(kernel='rbf', gamma='auto', C=0.2)
clf.fit(x_train_np, y_train_np)

y_pred_train = clf.predict(x_train_np)
y_pred_test = clf.predict(x_test_np)

train_accuracy = np.mean(y_pred_train == y_train_np)
test_accuracy = np.mean(y_pred_test == y_test_np)

print(f"scikit-learn SVM Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"scikit-learn SVM Test Accuracy: {test_accuracy * 100:.2f}%")

from sklearn.model_selection import GridSearchCV
n_features = x_train.shape[1]
var_X = np.var(x_train)

gamma = float(1 / n_features)
gamma_withVar = float(1 / (n_features * var_X))

print(f"gamma: {gamma}")
# Expand the range for C and gamma
param_grid = {
    'C': [0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10],  # Wider range for regularization parameter
    'gamma': [gamma, gamma_withVar, 0.0001, 0.001, 0.01, 0.1, 1, 10],  # Wider range for RBF kernel parameter
    'kernel': ['rbf']
}

# Initialize SVM classifier
svc = SVC()

# Perform grid search
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, verbose=3)
grid_search.fit(cp.asnumpy(x_train), cp.asnumpy(y_train))

# Best parameters and accuracy
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate on test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(cp.asnumpy(x_test), cp.asnumpy(y_test))
print(f"Test Accuracy with Best Model: {test_accuracy * 100:.2f}%")

