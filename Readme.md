# Customer Churn Prediction using Machine Learning

## Project Overview
This project predicts whether a customer is likely to leave a company (Churn) or continue using its services (Not Churn) using Machine Learning.

It uses classification algorithms to analyze customer information and identify customers who are at risk of leaving the company.

The main objective is to help businesses understand customer behavior and improve customer retention.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Imbalanced-learn (if used)
- Joblib
- Streamlit (if deployed as a web application)

## Project Features
- Data cleaning and preprocessing
- Handling missing values
- Exploratory Data Analysis (EDA)
- Categorical data encoding
- Feature scaling
- Handling imbalanced datasets (if applicable)
- Customer churn prediction using Machine Learning
- Model evaluation using classification metrics
- Model saving using Joblib (if applicable)

## Machine Learning Workflow
1. Load the customer churn dataset.
2. Perform data cleaning and preprocessing.
3. Handle missing values and duplicate records.
4. Perform Exploratory Data Analysis (EDA).
5. Encode categorical features into numerical values.
6. Split the dataset into training and testing sets.
7. Apply feature scaling where required.
8. Train Machine Learning classification models.
9. Evaluate model performance.
10. Save the trained model using Joblib (if applicable).

## Project Files
- `Customer_Churn_Prediction.ipynb` — Jupyter Notebook containing data analysis, preprocessing, model training, and evaluation.
- `customer_churn.csv` — Customer churn dataset.
- `model.pkl` — Saved trained model (if applicable).
- `requirements.txt` — Required Python libraries.

## Installation
Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Run the Project
Open the terminal in the project directory and run:

```bash
jupyter notebook
```

Open the Customer Churn Prediction notebook and execute the cells.

## Model Evaluation
The model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics help evaluate how effectively the model identifies customers who are likely to leave the company.

## Model Saving
The trained model can be saved using Joblib:

```python
import joblib

joblib.dump(model, 'model.pkl')
```

## Future Improvements
- Compare multiple Machine Learning algorithms.
- Improve model performance through hyperparameter tuning.
- Apply techniques to handle class imbalance.
- Identify important factors affecting customer churn.
- Develop an interactive Streamlit web application.
- Deploy the application online.

## Author
Machine Learning Project