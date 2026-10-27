# Telecom Customer Churn Prediction

A beginner-friendly machine-learning app that estimates whether a telecom customer may stop using a service.

## Features
- Bundled **synthetic demonstration data** so the app can run immediately
- Optional CSV upload for a compatible labelled dataset
- Data preprocessing for numeric and categorical columns
- Logistic Regression classification
- Held-out test metrics: accuracy, precision, recall, and confusion matrix
- Interactive churn-probability demonstration
- Streamlit interface

## Tech stack
Python, Pandas, Scikit-learn, Logistic Regression, Streamlit.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   streamlit run app.py
   ```

The bundled dataset is synthetic and is included only to demonstrate the end-to-end workflow. Its results must not be presented as real telecom performance.

## Use a public dataset
You can optionally download the IBM Telco Customer Churn sample dataset from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Upload its CSV in the app. The target column must be named `Churn` and contain values such as `Yes` and `No`. The app handles the common `customerID` column by dropping it.

Always check the dataset's license and terms before redistributing it. Do not upload private customer data.

## How it works
1. Read customer rows and the `Churn` target.
2. Split data into training and test sets.
3. Impute missing numeric and categorical values.
4. Scale numeric features and one-hot encode categorical features.
5. Train Logistic Regression on the training set.
6. Evaluate on the held-out test set.
7. Display a demonstration probability for an editable customer profile.

## Metrics in plain language
- **Accuracy:** fraction of all test predictions that are correct.
- **Precision:** among customers predicted to churn, the fraction who actually churn.
- **Recall:** among customers who actually churn, the fraction identified by the model.
- **Confusion matrix:** counts correct and incorrect predictions by class.

## Limitations
- Bundled data is synthetic; its performance is not evidence of real-world accuracy.
- The editable profile begins with a row from the dataset and changes common fields. It is a demonstration, not a complete production input form.
- Real deployment would require representative data, robust validation, fairness/privacy review, monitoring, and business cost analysis.
- A high churn probability does not prove that a customer will leave.

## Suggested improvements
- Use a properly licensed, representative real dataset.
- Compare Logistic Regression with a tree-based baseline.
- Tune the classification threshold using business costs.
- Add cross-validation and class-wise metrics.
- Build a complete form for all input features.
- Track model drift and evaluate fairness across groups.

## Learning sources
- IBM Telco Customer Churn sample dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Scikit-learn preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- Streamlit documentation: https://docs.streamlit.io/

## Interview summary
“I built a Python application that estimates telecom customer churn using a supervised machine-learning workflow. It preprocesses numerical and categorical data, trains Logistic Regression, and reports accuracy, precision, recall, and a confusion matrix on held-out test data. The included dataset is synthetic for demonstration; I would validate on representative licensed data and perform privacy and fairness checks before any business use.”
