import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("train.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove loan_id
df.drop("loan_id", axis=1, inplace=True)

# Encode categorical columns
encoder = LabelEncoder()

df["education"] = encoder.fit_transform(df["education"])
df["self_employed"] = encoder.fit_transform(df["self_employed"])
df["loan_status"] = encoder.fit_transform(df["loan_status"])

# Features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

best_model = None
best_accuracy = 0

print("\nModel Accuracies")
print("-" * 40)

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(f"{name}: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print("\nBest Model:", type(best_model).__name__)
print("Best Accuracy:", best_accuracy)

joblib.dump(best_model, "model.pkl")

print("\nmodel.pkl saved successfully!")