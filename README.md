# Smart-Lender
# Smart Lender - Loan Eligibility Prediction System

## Overview

Smart Lender is a Machine Learning based web application that predicts whether a loan applicant is eligible for loan approval. The system helps banks and financial institutions make quick, data-driven lending decisions by analyzing applicant information such as income, loan amount, CIBIL score, education, employment status, and asset values.

The application is built using Python, Flask, Scikit-learn, and XGBoost, and provides real-time loan eligibility prediction through a simple web interface.

---

## Features

- Predicts loan approval status instantly
- User-friendly Flask web application
- Compares multiple Machine Learning algorithms
- Automatically selects the best performing model
- Clean and responsive interface
- Real-time prediction using a trained ML model

---

## Technologies Used

- Python
- Flask
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Joblib
- HTML
- CSS

---

## Machine Learning Models Used

- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- XGBoost Classifier

The application evaluates all four models and automatically saves the best-performing model for prediction.

---

## Project Structure

```
Smart-Lender/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── train_model.py
├── model.pkl
├── train.csv
├── requirements.txt
├── README.md
└── Smart-Lender-Demo-compressed.mp4
```

---

## Dataset Features

The model uses the following applicant details:

- Number of Dependents
- Education
- Self Employed
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets Value
- Commercial Assets Value
- Luxury Assets Value
- Bank Assets Value

Target Variable:

- Loan Status (Approved / Rejected)

---

## Installation

Clone the repository

```bash
git clone https://github.com/niharshinimuvva-hash/Smart-Lender.git
```

Move into the project directory

```bash
cd Smart-Lender
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run

```bash
python train_model.py
```

This trains multiple Machine Learning models, compares their performance, and saves the best model as `model.pkl`.

---

## Run the Application

Start the Flask server

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

Enter applicant details and click **Predict Loan Status** to receive the prediction.

---

## Sample Prediction

### Input

| Feature | Value |
|---------|------:|
| Dependents | 2 |
| Education | Graduate |
| Self Employed | No |
| Annual Income | 8000000 |
| Loan Amount | 2000000 |
| Loan Term | 15 |
| CIBIL Score | 850 |

### Output

```
Loan Approved ✅
```

---

## Future Enhancements

- IBM Cloud Deployment
- Probability Score for Predictions
- User Authentication
- Loan Recommendation System
- Improved Dashboard and Analytics
- REST API Integration
- Database Connectivity

---

## Team Members

- Priya Veera Abhishek Bezzanki (Team Lead)
- Muvva Niharshini
- Vadlamani Sravani
- Bacchu Venkata Srinivasa Reddy
- Venkata Srinivasa Reddy Bacchu

---

## Repository

GitHub Repository:

https://github.com/niharshinimuvva-hash/Smart-Lender

---

## License

This project is developed for educational and internship purposes under the SmartBridge Skill Wallet Internship Program.
