# Step : Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Step 1: Sample Dataset
data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female'],
    'SeniorCitizen': [0, 1, 0, 0, 1, 1, 0, 0],
    'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70, 89.10, 25.50, 99.75],
    'Tenure': [1, 34, 2, 45, 5, 10, 1, 60],
    'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year', 'Month-to-month', 'Month-to-month', 'Month-to-month', 'Two year'],
    'Churn': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No']
}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('churn.csv', index=True)
print("Churn created successfully!")

# Step 2: Data Preprocessing
# Handle missing values
df.dropna(inplace=True)

#Encode categorical columns
label_encoders = {}
for column in ['Gender', 'Contract', 'Churn']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Step 3: Feature Selection and Scaling
X = df[['Gender', 'SeniorCitizen', 'MonthlyCharges', 'Tenure', 'Contract']]
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 5: Train Models
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(max_depth=4)
tree_model.fit(X_train, y_train)

# Step 6: Predictions
log_pred = log_model.predict(X_test)
tree_pred = tree_model.predict(X_test)

# # Step 7: Evaluation
# def evaluate_model(y_test, predictions, model_name):
#     print(f"\n🔍 Evaluation for {model_name}")
#     print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
#     print("\nClassification Report:\n", classification_report(y_test, predictions))

# evaluate_model(y_test, log_pred, "Logistic Regression")
# evaluate_model(y_test, tree_pred, "Decision Tree")

# # Step 8: ROC Curve for Logistic Regression
# y_prob_log = log_model.predict_proba(X_test)[:, 1]
# fpr, tpr, _ = roc_curve(y_test, y_prob_log)
# roc_auc = roc_auc_score(y_test, y_prob_log)

# plt.figure(figsize=(7, 5))
# plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve - Logistic Regression")
# plt.legend()
# plt.grid(True)
# plt.show()

# Step 9: Better Visualization - Feature Importance (Decision Tree)
plt.figure(figsize=(6, 4))
sns.barplot(
    x=tree_model.feature_importances_,
    y=['Gender', 'SeniorCitizen', 'MonthlyCharges', 'Tenure', 'Contract']
)
plt.title("Feature Importance - Decision Tree")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show() 

# --- Step 10: Predict Churn for a New Employee ---

# Example new employee data
new_employee = {
    'Gender': 'Male',
    'SeniorCitizen': 0,
    'MonthlyCharges': 65.0,
    'Tenure': 12,
    'Contract': 'Month-to-month'
}

# Convert to DataFrame
new_df = pd.DataFrame([new_employee])

# Encode using existing label encoders
for column in ['Gender', 'Contract']:
    if column in new_df:
        le = label_encoders[column]
        new_df[column] = le.transform(new_df[column])

# Arrange columns in the same order as training data
X_new = new_df[['Gender', 'SeniorCitizen', 'MonthlyCharges', 'Tenure', 'Contract']]

# Scale the data
X_new_scaled = scaler.transform(X_new)

# Predict churn
prediction = log_model.predict(X_new_scaled)[0]  # 0 or 1
prediction_label = label_encoders['Churn'].inverse_transform([prediction])[0]  # 'Yes' or 'No'

print(f"\n🔮 Prediction for New Employee:")
print("Churn Prediction:", prediction_label)
# # Apply KMeans
# kmeans = KMeans(n_clusters=3)
# df['cluster'] = kmeans.fit_predict(X)

# # Scatter Plot
# plt.scatter(df['Contract'], df['Churn'], c=df['cluster'], cmap='viridis')
# plt.xlabel("Contract")
# plt.ylabel("Churn")
# plt.title("Workers segments")
# plt.show()
