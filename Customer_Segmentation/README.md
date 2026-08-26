# 🎯 Customer Segmentation & Analytics Dashboard

Developed by **Uday Aware**

An end-to-end Machine Learning project that segments customers based on demographic attributes and purchasing behaviors. This project combines **K-Means Clustering**, interactive data visualization, and a **Streamlit Web Application** for real-time customer classification and targeted strategy development.

---

## 📌 Project Overview
Customer segmentation enables businesses to categorize customers into distinct groups to deliver targeted marketing, improve retention, and optimize revenue streams. 

### Key Features:
* **RFM & Demographic Analysis:** Evaluates customer Recency, Frequency, Spending Score, Income, and Age.
* **K-Means Clustering:** Employs optimal cluster identification ($k=4$) using Elbow Method analysis and Silhouette Scores.
* **Interactive Dashboard:** Built with Streamlit and Plotly for visual exploration and dynamic customer classification.
* **Actionable Business Recommendations:** Maps each customer segment to tailored engagement strategies.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (StandardScaler, KMeans)
* **Visualizations:** `plotly`, `seaborn`, `matplotlib`
* **Web Framework:** `streamlit`

---

## 📊 Customer Segment Insights

| Cluster | Segment Name | Primary Profile | Recommended Business Strategy |
| :--- | :--- | :--- | :--- |
| **Cluster 0** | **Loyal Shoppers** | High purchase frequency, older demographic | Target with loyalty programs & subscription perks |
| **Cluster 1** | **Trend Seekers** | High spending score, younger demographic | Promote high-end product recommendations & trendy items |
| **Cluster 2** | **At-Risk Customers** | Low spending score, low purchase frequency | Launch win-back campaigns & limited-time discount codes |
| **Cluster 3** | **High Potential** | High annual income, low current spending score | Position premium luxury offerings & cross-sell upsells |

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/udayaware5602-coder/Customer-Segmentation.git](https://github.com/udayaware5602-coder/Customer-Segmentation.git)
cd Customer-Segmentation
