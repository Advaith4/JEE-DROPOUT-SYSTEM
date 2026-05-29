import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

# Create static directory if it doesn't exist for saving graphs
if not os.path.exists('static'):
    os.makedirs('static')

# 1. Load Data
# Assuming the file is in the 'archive' folder as seen in the file list
file_path = os.path.join('archive', 'JEE_Dropout_After_Class_12.csv')
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# 2. Preprocessing
print("Preprocessing data...")
# Identify columns
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
numerical_cols = numerical_cols.drop('dropout') # Remove target

# Handle categorical data with Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Store numerical columns names for later use in scaling
numerical_features = numerical_cols.tolist()

# Split X and y
X = df.drop('dropout', axis=1)
y = df['dropout']

# Scale numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training & Evaluation
print("Training models...")
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

results = {}
best_model_name = ""
best_model_score = 0
best_model_obj = None

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    # We'll use F1 Score to select the best model
    if f1 > best_model_score:
        best_model_score = f1
        best_model_name = name
        best_model_obj = model

print(f"\nBest Model: {best_model_name} with F1 Score: {best_model_score:.4f}")

# 4. Save Artifacts
print("Saving artifacts...")
joblib.dump(best_model_obj, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("Models and preprocessors saved.")

# 5. Generate Graphs
print("Generating graphs...")
# Convert results to DataFrame for plotting
metrics_df = pd.DataFrame(results).T

# Plot Performance Metrics Comparison
plt.figure(figsize=(10, 6))
metrics_df.plot(kind='bar', figsize=(12, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join('static', 'model_comparison.png'))
print("Saved metric comparison graph to static/model_comparison.png")

# Confusion Matrix for Best Model
y_pred_best = best_model_obj.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Dropout'], yticklabels=['Stayed', 'Dropout'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join('static', 'confusion_matrix.png'))
print("Saved confusion matrix to static/confusion_matrix.png")

print("Training pipeline completed successfully.")
