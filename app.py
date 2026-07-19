from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get form inputs
        no_of_dependents = int(request.form["no_of_dependents"])

        education = 1 if request.form["education"] == "Graduate" else 0

        self_employed = 1 if request.form["self_employed"] == "Yes" else 0

        income_annum = int(request.form["income_annum"])

        loan_amount = int(request.form["loan_amount"])

        loan_term = int(request.form["loan_term"])

        cibil_score = int(request.form["cibil_score"])

        residential_assets_value = int(
            request.form["residential_assets_value"]
        )

        commercial_assets_value = int(
            request.form["commercial_assets_value"]
        )

        luxury_assets_value = int(
            request.form["luxury_assets_value"]
        )

        bank_asset_value = int(
            request.form["bank_asset_value"]
        )

        # Arrange features in same order as training dataset
        data = np.array([[
            no_of_dependents,
            education,
            self_employed,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]])

        # Predict
        prediction = model.predict(data)[0]

        if prediction == 0:
            result = "Loan Approved ✅"
        else:
            result = "Loan Rejected ❌"

        return render_template(
            "result.html",
            prediction=result
        )

    except Exception as e:

        return render_template(
            "result.html",
            prediction=f"Error: {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)