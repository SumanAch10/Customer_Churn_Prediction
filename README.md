# 📉 Customer Churn Prediction

This project builds and deploys a **Customer Churn Prediction system** using **machine learning** and **Streamlit**.  
The model predicts whether a customer is likely to churn based on their account, service, and billing information.

---
## https://customerchurnprediction-tq9ketjwbrp4somcnrxgpz.streamlit.app/


## 🚀 Project Overview

Customer churn prediction helps businesses:
- identify high-risk customers
- take proactive retention actions
- reduce revenue loss

In this project:
- A **Logistic Regression model** is trained using scikit-learn
- Feature preprocessing is handled using a **Pipeline + ColumnTransformer**
- The trained pipeline is deployed as an **interactive Streamlit web app**

---

## 🧠 Machine Learning Approach

- **Problem type:** Binary classification  
- **Target variable:** `Churn` (Yes / No)
- **Model:** Logistic Regression  
- **Class imbalance handling:** `class_weight="balanced"`
- **Preprocessing:**
  - Numerical features → scaling
  - Categorical features → one-hot encoding
- **Pipeline:** Preprocessing + model combined into a single pipeline

---

## 🖥️ Streamlit Web App

The Streamlit app allows users to:
- enter customer details via form inputs and dropdowns
- get a churn prediction instantly
- view churn probability
- experiment with different customer profiles interactively

---

## 📁 Project Structure

