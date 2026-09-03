# 🏎️ AutoPrice AI – Vehicle Price Prediction

## 📌 Project Overview

**AutoPrice AI** is a Machine Learning web application designed to predict the estimated market price of a vehicle based on its specifications.

The application is built using **Python, Machine Learning, and Streamlit**. It provides an interactive dashboard where users can enter vehicle details such as brand, fuel type, transmission, horsepower, engine size, curb weight, mileage, and wheelbase to generate an estimated vehicle price.

The prediction engine uses a **Random Forest Regressor** trained for vehicle price valuation.

---

## 🚀 Features

* 🏎️ Vehicle price prediction
* 🤖 Random Forest Machine Learning model
* 📊 Interactive Streamlit dashboard
* 🚗 Multiple vehicle brands and body styles
* ⛽ Fuel type selection
* ⚙️ Automatic/Manual transmission selection
* 🛞 Drive train selection
* 🐎 Horsepower input
* 🔧 Engine size input
* ⚖️ Curb weight input
* 🛣️ Highway mileage
* 🏙️ City mileage
* 📏 Wheelbase input
* 📥 Downloadable vehicle valuation report in CSV format
* 📊 Dataset analytics
* 👥 Development team section
* 🌑 Modern dark glassmorphism UI

---

## 🧠 Machine Learning Model

The application uses:

**Algorithm:** Random Forest Regressor

**Problem Type:** Regression

**Target Variable:** Vehicle Price

Random Forest combines predictions from multiple decision trees to produce a robust regression result.

The trained model is loaded from:

```text
random_forest_model.pkl
```

The feature information required by the model is loaded from:

```text
features.pkl
```

The application checks that both files are available before running the prediction system.

---

## 🛠️ Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Programming language      |
| Streamlit    | Web application framework |
| Pandas       | Data processing           |
| NumPy        | Numerical operations      |
| Scikit-Learn | Machine Learning          |
| Pickle       | Loading trained model     |
| HTML/CSS     | User interface styling    |

---

## 📂 Project Structure

```text
AutoPrice-AI/
│
├── app.py
├── autos_dataset.csv
├── random_forest_model.pkl
├── features.pkl
├── README.md
└── requirements.txt
```

### Important Files

* `app.py` – Main Streamlit application
* `autos_dataset.csv` – Automobile dataset used for analytics
* `random_forest_model.pkl` – Trained Random Forest model
* `features.pkl` – Model feature information
* `README.md` – Project documentation
* `requirements.txt` – Python dependencies

---

## 💻 Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd AutoPrice-AI
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

Or install the main packages manually:

```bash
pip install streamlit pandas numpy scikit-learn
```

---

## ▶️ Run the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📊 Application Pages

### 🏠 Dashboard

The dashboard provides an overview of the AutoPrice AI system.

It displays:

* Machine Learning model
* Number of active features
* Problem domain
* Lead developer
* Valuation drivers visualization
* Core system capabilities

---

### 💰 Price Prediction

Users can enter vehicle specifications including:

**Vehicle Information**

* Make / Brand
* Body Style
* Fuel Type
* Transmission
* Drive Train
* Engine Position

**Engine & Dimensions**

* Horsepower
* Engine Size
* Curb Weight
* Highway MPG
* City MPG
* Wheelbase

After clicking:

```text
🚀 ESTIMATE VEHICLE PRICE
```

the Random Forest model generates an estimated vehicle price.

The application also provides an option to download the prediction as a CSV report.

---

### 📊 Dataset Analytics

The Dataset Analytics page loads:

```text
autos_dataset.csv
```

and displays:

* Total records
* Total columns
* Target variable
* Missing values
* Dataset preview

---

### 👥 Our Team

The project includes the following contributors:

### 👨‍💻 Uday Aware

**Lead AI & Machine Learning Engineer**

Responsibilities:

* System architecture
* Random Forest model
* Streamlit application
* Core machine learning implementation

### 🎨 Aniket

**UI/UX & Frontend Specialist**

Responsibilities:

* UI design
* Glassmorphism interface
* Layout styling

### 📊 Data Science Team

**Data Engineering & Analytics**

Responsibilities:

* Automobile data cleaning
* Missing-value handling
* Feature engineering
* Dataset analytics

---

## 🔄 How the System Works

```text
Vehicle Dataset
      ↓
Data Cleaning & Preparation
      ↓
Feature Engineering
      ↓
Random Forest Regressor
      ↓
Trained Model (.pkl)
      ↓
Streamlit Application
      ↓
User Enters Vehicle Details
      ↓
Model Prediction
      ↓
Estimated Vehicle Price
      ↓
CSV Valuation Report
```

---

## 📥 Prediction Report

After generating a prediction, the application creates a CSV report containing the model input features and predicted price.

The generated file is:

```text
vehicle_valuation.csv
```

---

## 🎨 User Interface

The application uses a modern **dark glassmorphism design** with:

* Dark background
* Glass-style cards
* Gradient headings
* Interactive buttons
* Dashboard metrics
* Responsive Streamlit columns
* Vehicle-themed interface

---

## 📋 Requirements

Create a `requirements.txt` file containing:

```text
streamlit
pandas
numpy
scikit-learn
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚠️ Important

Make sure these files are present in the project root directory:

```text
random_forest_model.pkl
features.pkl
autos_dataset.csv
```

Without the model and feature files, the prediction system cannot operate.

---

## 🔮 Future Scope

Future improvements could include:

* 🌐 Deploying the application online
* 📈 Adding advanced model performance metrics
* 🧠 Comparing multiple regression algorithms
* 📊 Adding interactive charts
* 🗺️ Adding location-based vehicle pricing
* 💾 Database integration
* 👤 User authentication
* 📱 Mobile-friendly improvements
* 🔄 Automatic model retraining
* 📈 Real-time automobile market data integration

---

## 👨‍💻 Developer

**Uday Aware**

Lead AI & Machine Learning Engineer

### 🔗 LinkedIn

https://www.linkedin.com/in/uday-aware-5ba400381

### 🔗 GitHub

https://github.com/udayaware5602-coder

---

## ⭐ Project Highlights

> **AutoPrice AI** combines Machine Learning and an interactive Streamlit interface to provide fast and convenient vehicle price estimation.

**Built with ❤️ using Python, Machine Learning & Streamlit.**

---

## 📜 License

This project is intended for educational, academic, and demonstration purposes.
