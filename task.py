# Roll No: 22F-BSAI-06
# Project: Student Performance Predictor

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score

# Load and Initial Data Exploration

csv = pd.read_csv('student-mat.csv', sep=';')
print("Initial data preview:")
print(csv.head())

# Data Preprocessing: Check for missing values
print("\nChecking for missing values:")
print(csv.isnull().sum())
# The output confirms no missing values, simplifying the preprocessing step.

# Feature Selection and Target Preparation

# Features selected for predicting performance:
# 'studytime': Directly relates to student effort (time spent studying).
# 'absences': Relates to consistency and engagement.
# 'G1', 'G2': Previous period grades are strong predictors of the final grade (G3).
features = ['studytime', 'absences', 'G1', 'G2']
target = 'G3'

# Prepare features and target
X = csv[features]
y = csv[target]

# Create binary target (Pass/Fail) for Classification Models (Logistic Regression, Decision Tree)
y_binary = (y >= 10).astype(int)

# Data Scaling and Splitting

# Apply Standard Scaling to features.
# Scaling is crucial for Linear and Logistic Regression models to prevent features
# with larger ranges (like G1/G2) from disproportionately influencing the model coefficients.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

print("\nScaled features preview (used for classification models):")
print(X_scaled.head())

# Split the scaled data for Classification (X_scaled, y_binary)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42
)

# Split the original (unscaled) data for Linear Regression (X, y)
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression Model (Predicting Final Marks - G3)

# Model Choice: Linear Regression is used because the target 'G3' is a continuous, 
# numerical variable (final mark), making this a regression task.
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)
y_pred_lr = lr_model.predict(X_test_lr)

# Evaluate Linear Regression using R-squared (Coefficient of Determination)
lr_score = r2_score(y_test_lr, y_pred_lr)
print(f'\nLinear Regression Performance (Predicting G3)')
print(f'R² Score (Model Fit): {lr_score:.3f}')

# Logistic Regression Model (Classifying Pass/Fail)

# Model Choice: Logistic Regression is used for binary classification (Pass/Fail).
# It models the probability of passing using a sigmoid function.
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train_cls, y_train_cls)
y_pred_log = log_model.predict(X_test_cls)

# Evaluate Logistic Regression
log_accuracy = accuracy_score(y_test_cls, y_pred_log)
log_precision = precision_score(y_test_cls, y_pred_log)
log_recall = recall_score(y_test_cls, y_pred_log)

# Decision Tree Classifier Model (Classifying Pass/Fail)

# Model Choice: Decision Tree is a non-linear, non-parametric model used for 
# classification. It can capture complex non-linear relationships in the data.
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_cls, y_train_cls)
y_pred_dt = dt_model.predict(X_test_cls)

# Evaluate Decision Tree
dt_accuracy = accuracy_score(y_test_cls, y_pred_dt)
dt_precision = precision_score(y_test_cls, y_pred_dt)
dt_recall = recall_score(y_test_cls, y_pred_dt)

# Model Comparison and Visualization (Classification)

print("\nClassification Model Comparison (Pass/Fail) ---")

# Combine results into a DataFrame for clear comparison
comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall'],
    'Logistic Regression': [log_accuracy, log_precision, log_recall],
    'Decision Tree': [dt_accuracy, dt_precision, dt_recall]
})

print(comparison.set_index('Metric').round(3))

models = ['Logistic Regression', 'Decision Tree']
accuracies = [log_accuracy, dt_accuracy]
precisions = [log_precision, dt_precision]
recalls = [log_recall, dt_recall]
x = range(len(models))

# Visualization of Model Comparison
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.bar(x, accuracies, width=0.6, color='skyblue')
plt.xticks(x, models, rotation=15)
plt.ylabel('Score')
plt.title('Accuracy Comparison')

plt.subplot(1, 3, 2)
plt.bar(x, precisions, width=0.6, color='lightcoral')
plt.xticks(x, models, rotation=15)
plt.ylabel('Score')
plt.title('Precision Comparison')

plt.subplot(1, 3, 3)
plt.bar(x, recalls, width=0.6, color='lightgreen')
plt.xticks(x, models, rotation=15)
plt.ylabel('Score')
plt.title('Recall Comparison')

plt.tight_layout()
plt.suptitle('Performance Comparison of Classification Models', y=1.02, fontsize=16)
plt.show()