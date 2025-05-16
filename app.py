import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import h5py

# Load the Excel file
df = pd.read_excel("/Users/geethika/Desktop/IML/ML.xlsx", sheet_name="Sheet1")

# Drop unused columns
df.drop(columns=["Order ID", "Customer ID", "Order Date & Time", "Customer Feedback"], inplace=True)

# Encode target variable
df["Delivery Delay"] = df["Delivery Delay"].map({"Yes": 1, "No": 0})

# Encode categorical features
label_encoders = {}
for col in ["Platform", "Product Category", "Refund Requested"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare X and y
X = df.drop("Delivery Delay", axis=1)
y = df["Delivery Delay"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("✅ logistic_model.h5 created successfully!")
print(f"📊 Accuracy: {accuracy:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Delay", "Delay"], yticklabels=["No Delay", "Delay"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()
# Save model and scaler using joblib
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save model and scaler into .h5
with h5py.File("logistic_model.h5", "w") as h5f:
    h5f.create_dataset("model", data=np.void(open("logistic_model.pkl", "rb").read()))
    h5f.create_dataset("scaler", data=np.void(open("scaler.pkl", "rb").read()))
