# 🛒 Outlet Sales Prediction Using Machine Learning

## 📌 Project Overview

**Outlet Sales Prediction** is a Machine Learning regression project developed to predict the sales of products at different retail outlets.

The project uses product-related and outlet-related information such as product type, product visibility, product weight, MRP, outlet size, outlet location, and outlet type to predict the expected **Item Outlet Sales**.

The main Machine Learning algorithm used in this project is **K-Nearest Neighbors Regression (KNN Regressor)**.

---

## 🎯 Problem Statement

Retail businesses generate large amounts of sales data. Predicting the expected sales of a product can help businesses make better decisions related to:

* Inventory management
* Product placement
* Stock planning
* Outlet management
* Sales forecasting
* Business strategy

The objective of this project is to develop a regression model that can predict product sales based on the available product and outlet features.

---

## 🎯 Project Objectives

The major objectives of this project are:

1. Understand and analyze the sales dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Identify missing values and handle them appropriately.
4. Convert categorical variables into numerical values.
5. Perform feature preprocessing.
6. Normalize numerical features using `MinMaxScaler`.
7. Split the dataset into training and testing sets.
8. Build a KNN Regression model.
9. Predict outlet sales.
10. Evaluate the performance of the model.
11. Understand the factors affecting product sales.

---

## 📂 Dataset Information

The dataset contains information about products and retail outlets.

### Important Features

| Feature                     | Description                                         |
| --------------------------- | --------------------------------------------------- |
| `Item_Identifier`           | Unique identification number of the product         |
| `Item_Weight`               | Weight of the product                               |
| `Item_Fat_Content`          | Fat content of the product                          |
| `Item_Visibility`           | Percentage of display area allocated to the product |
| `Item_Type`                 | Category/type of the product                        |
| `Item_MRP`                  | Maximum Retail Price of the product                 |
| `Outlet_Identifier`         | Unique identification number of the outlet          |
| `Outlet_Establishment_Year` | Year in which the outlet was established            |
| `Outlet_Size`               | Size of the outlet                                  |
| `Outlet_Location_Type`      | Location category of the outlet                     |
| `Outlet_Type`               | Type/category of the outlet                         |
| `Item_Outlet_Sales`         | Sales of the product at the outlet                  |

---

## 🔍 Data Preprocessing

Before training the Machine Learning model, the dataset is preprocessed.

### 1. Handling Missing Values

Missing values can negatively affect Machine Learning algorithms. Therefore, missing values are identified and replaced using suitable techniques such as:

* Mean/median imputation for numerical columns
* Mode imputation for categorical columns

### 2. Categorical Encoding

Machine Learning algorithms require numerical input.

Categorical columns are converted into numerical values using **One-Hot Encoding**.

Example:

```python
X = pd.get_dummies(X, drop_first=True)
```

Boolean values generated during encoding can then be converted into numerical values.

### 3. Feature Scaling

KNN is a distance-based algorithm, so feature scaling is very important.

The project uses:

```python
from sklearn.preprocessing import MinMaxScaler

norm_scaler = MinMaxScaler()
X_nor = norm_scaler.fit_transform(X)
```

Min-Max scaling transforms the features into a common range, generally between **0 and 1**.

---

## 📊 Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the structure and characteristics of the dataset.

The analysis can include:

* Checking dataset shape
* Checking data types
* Finding missing values
* Detecting duplicate records
* Studying numerical features
* Studying categorical features
* Finding relationships between variables
* Understanding sales distribution
* Identifying important sales-related features

Example:

```python
df.shape
df.info()
df.describe()
df.isnull().sum()
```

---

## 🤖 Machine Learning Algorithm

### K-Nearest Neighbors Regression

**KNN Regression** is a supervised Machine Learning algorithm used for predicting continuous numerical values.

For a new data point, KNN:

1. Calculates the distance between the new point and training data.
2. Finds the nearest `K` observations.
3. Uses the values of these neighboring observations.
4. Calculates their average.
5. Uses the average as the predicted value.

### Why KNN?

KNN is useful for this project because:

* It is simple to understand.
* It can model non-linear relationships.
* It works well with properly scaled numerical features.
* It does not require a complex mathematical model.

However, KNN can be sensitive to feature scaling and missing values.

---

## 🧠 Model Implementation

The KNN regression model can be created using:

```python
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor()

knn_reg.fit(X_train_nor, y_train)

y_pred = knn_reg.predict(X_test_nor)
```

---

## ✂️ Train-Test Split

The dataset is divided into two parts:

* **Training Dataset** – used to train the Machine Learning model.
* **Testing Dataset** – used to evaluate the model on unseen data.

Example:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_nor,
    y,
    test_size=0.2,
    random_state=42
)
```

---

## 📏 Model Evaluation

Regression models can be evaluated using different performance metrics.

### Mean Absolute Error — MAE

MAE measures the average absolute difference between actual and predicted values.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
```

### Mean Squared Error — MSE

MSE calculates the average squared difference between actual and predicted values.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
```

### Root Mean Squared Error — RMSE

RMSE is the square root of MSE.

```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

### R² Score

R² indicates how well the model explains the variation in the target variable.

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
```

A higher **R² score** generally indicates better model performance.

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Collection
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Missing Value Treatment
   ↓
Exploratory Data Analysis
   ↓
Categorical Encoding
   ↓
Feature Selection
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
KNN Regression
   ↓
Sales Prediction
   ↓
Model Evaluation
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Scikit-learn

### Development Environment

* Jupyter Notebook
* Google Colab / VS Code

---

## 📦 Installation

Install the required Python libraries using:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## ▶️ How to Run the Project

### Step 1 — Clone the repository

```bash
git clone <your-github-repository-link>
```

### Step 2 — Open the project

Open the project using:

* Jupyter Notebook
* Google Colab
* VS Code

### Step 3 — Load the dataset

Place the CSV dataset inside the project folder.

Example:

```python
import pandas as pd

df = pd.read_csv("KNN_reg_outlet_sales.csv")
```

### Step 4 — Run preprocessing

Handle missing values and encode categorical features.

### Step 5 — Scale the features

```python
from sklearn.preprocessing import MinMaxScaler

norm_scaler = MinMaxScaler()
X_nor = norm_scaler.fit_transform(X)
```

### Step 6 — Train the model

```python
knn_reg.fit(X_train_nor, y_train)
```

### Step 7 — Generate predictions

```python
y_pred = knn_reg.predict(X_test_nor)
```

### Step 8 — Evaluate the model

Calculate MAE, MSE, RMSE, and R² score.

---

## 📈 Prediction

After training, the model can predict the expected sales value for new product and outlet information.

Example:

```python
prediction = knn_reg.predict(X_test_nor)

print("Predicted Sales:", prediction)
```

---

## 💡 Key Insights

This project helps demonstrate how Machine Learning can be applied to retail sales prediction.

Important factors that may influence outlet sales include:

* Product MRP
* Product visibility
* Product type
* Product weight
* Outlet type
* Outlet size
* Outlet location
* Outlet establishment year

The relationship between these variables and sales can be investigated through EDA and model analysis.

---

## 🚀 Future Scope

The project can be improved in several ways:

### 1. Try Different Algorithms

Other regression algorithms can be compared with KNN:

* Linear Regression
* Decision Tree Regression
* Random Forest Regression
* Gradient Boosting
* XGBoost
* Support Vector Regression

### 2. Hyperparameter Tuning

The value of `K` can be optimized using techniques such as Grid Search or Cross-Validation.

### 3. Feature Engineering

New useful features can be created from existing columns to improve prediction accuracy.

### 4. Model Deployment

The trained model can be deployed using:

* Flask
* Streamlit
* FastAPI

### 5. Web Application

A user-friendly web application can be created where users enter product and outlet details and receive predicted sales.

---

## 📁 Suggested Project Structure

```text
Outlet-Sales-Prediction/
│
├── Dataset/
│   └── KNN_reg_outlet_sales.csv
│
├── Notebook/
│   └── Outlet_Sales_Prediction.ipynb
│
├── Model/
│   └── knn_regressor.pkl
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

---

## 📋 Requirements

Example `requirements.txt`:

```text
pandas
numpy
matplotlib
scikit-learn
```

---

## ⚠️ Important Notes

KNN Regression requires numerical input and does not handle missing values directly. Therefore:

* Missing values must be handled before model training.
* Categorical variables must be encoded.
* Features should be scaled because KNN uses distance calculations.
* Training and testing data should be processed consistently.

---

## 🎓 Learning Outcomes

After completing this project, the following concepts are demonstrated:

* Python programming
* Pandas data manipulation
* Data preprocessing
* Missing value handling
* Categorical encoding
* Feature scaling
* Exploratory Data Analysis
* Train-test splitting
* KNN Regression
* Model prediction
* Regression evaluation metrics
* Basic Machine Learning workflow

---

## 👨‍💻 Author

**Aware Uday Navnath**

**Roll No.: 06**

**Degree:** B.E. – Artificial Intelligence & Data Science
LinkedIn:https://www.linkedin.com/in/uday-aware-5ba400381
---

## ⭐ Conclusion

The **Outlet Sales Prediction** project demonstrates a complete Machine Learning workflow, starting from data preprocessing and exploratory analysis to model training, prediction, and evaluation.

Using **K-Nearest Neighbors Regression**, the project attempts to predict `Item_Outlet_Sales` from product and outlet characteristics.

This project provides practical experience in handling real-world datasets and applying Machine Learning techniques to solve a business-oriented prediction problem.

---

## 📌 Project Status

**Status:** Completed / Under Development

**Model:** KNN Regression

**Task:** Regression

**Target:** `Item_Outlet_Sales`

**Domain:** Machine Learning / Retail Sales Prediction
