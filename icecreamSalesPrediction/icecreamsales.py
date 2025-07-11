
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
 
from sklearn.metrics import accuracy_score


# Load data from CSV
df = pd.read_csv('icecream_sales.csv')

# Optional: fix column name if needed (Excel sometimes corrupts names)
df.rename(columns={"Avaerage+AF8-Sales": "Average_Sales"}, inplace=True)

# Prepare features and target
X = df[['Temperature', 'Sunny', 'Average_Sales',]]
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train KNN classifier
# model = KNeighborsClassifier(n_neighbors=5)
# model.fit(X_train, y_train)

# # Train Decision Tree Classifier
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

#Train Logistic regression Classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Predicted Defaults:", y_pred)
print("Actual Defaults:", y_test.values)
print("Accuracy:", accuracy_score(y_test, y_pred))
