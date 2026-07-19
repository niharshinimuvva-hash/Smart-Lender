import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("train.csv", skipinitialspace=True)

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Remove spaces from string values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# ==========================
# Handle Missing Values
# ==========================

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

# ==========================
# Drop loan_id
# ==========================

df.drop("loan_id", axis=1, inplace=True)

# ==========================
# Manual Encoding
# ==========================

df["education"] = df["education"].map({
    "Graduate": 1,
    "Not Graduate": 0
})

df["self_employed"] = df["self_employed"].map({
    "Yes": 1,
    "No": 0
})

# Encode target
label_encoder = LabelEncoder()
df["loan_status"] = label_encoder.fit_transform(df["loan_status"])
print(label_encoder.classes_)

# ==========================
# Features and Target
# ==========================

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Models
# ==========================

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
}

best_model = None
best_accuracy = 0

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for name, model in models.items():

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Training Accuracy : {train_acc:.4f}")
    print(f"Testing Accuracy  : {test_acc:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, test_pred))

    print("\nClassification Report")
    print(classification_report(y_test, test_pred))

    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_model = model
print("\nFeature Order:")
print(X.columns.tolist())

print("\nBest Model:")
print(best_model)

print("\nBest Accuracy:")
print(best_accuracy)

joblib.dump(best_model, "model.pkl")


print("\nBest Accuracy:", round(best_accuracy * 100, 2), "%")
print("model.pkl saved successfully!")