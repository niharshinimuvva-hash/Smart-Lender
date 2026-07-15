from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    no_of_dependents = int(request.form["no_of_dependents"])

    education = 1 if request.form["education"] == "Graduate" else 0

    self_employed = 1 if request.form["self_employed"] == "Yes" else 0

    income_annum = int(request.form["income_annum"])
    loan_amount = int(request.form["loan_amount"])
    loan_term = int(request.form["loan_term"])
    cibil_score = int(request.form["cibil_score"])
    residential_assets_value = int(request.form["residential_assets_value"])
    commercial_assets_value = int(request.form["commercial_assets_value"])
    luxury_assets_value = int(request.form["luxury_assets_value"])
    bank_asset_value = int(request.form["bank_asset_value"])

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

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Loan Approved ✅"
    else:
        result = "Loan Rejected ❌"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)