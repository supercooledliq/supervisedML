#creditCardApproval.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data from CSV
df = pd.read_csv('credit_approval_data.csv')

# Prepare features and target
X = df[['Income', 'Credit_Score', 'Card_Amount']]
y = df['Approval_Flag']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Predicted Defaults:", y_pred)
print("Actual Defaults:", y_test.values)
print("Accuracy:", accuracy_score(y_test, y_pred))

df['Card_Approved']=df['Credit_Score'].apply(lambda score: 'No' if score<400 else 'Yes')
print(df[['Credit_Score','Card_Approved']])