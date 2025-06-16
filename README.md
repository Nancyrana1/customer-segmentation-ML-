# 🛍️ Customer Segmentation Web App

This is a **Flask-based web application** for customer segmentation using **K-Means Clustering**. Users can upload a CSV file (e.g., Annual Income and Spending Score), and the app will predict customer segments based on your trained machine learning model.

---

## 📌 Features

- 📁 Upload a `.csv` file with customer data
- 🤖 Predict customer segments using a trained KMeans model
- 📊 Visual representation of clusters (optional)
- 💾 Model loaded from a `.pkl` (Pickle) file
- 🌐 Built using Flask and Bootstrap CSS

---

## 🛠️ Tech Stack

- Python 🐍
- Flask 🌶️
- Pandas / NumPy 📊
- scikit-learn 🤖
- HTML / CSS 🎨
- Bootstrap 5 (UI)

---

## 📂 Project Structure

|── app.py # Flask backend
├── customer_segmentation.pkl # Trained KMeans model
├── templates/
│ └── index.html # Frontend HTML
├── static/
│ └── style.css # Custom CSS
├── uploads/ # Directory for uploaded files (optional)
├── README.md # This file

