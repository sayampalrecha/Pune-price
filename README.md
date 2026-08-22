# 🏠 Pune House Price Prediction

## Link to the [dashboard](https://pune-price-prediction.streamlit.app/)

A machine learning project that predicts residential property prices in Pune, India. It includes a full data science notebook for model training and a lightweight **Streamlit web app** that runs inference using only NumPy — no scikit-learn required at runtime.

## 📁 Repository Structure

```
Pune-price/
├── Pune house price prediction.ipynb  # EDA, feature engineering & model training
├── Pune_House_Data.csv                # Raw dataset
├── app.py                             # Streamlit prediction app
├── export_model_assets.py             # Exports sklearn model to NumPy weights
├── model_weights.npz                  # Serialised linear model coefficients
├── model_features.json                # Ordered feature list for inference
├── Pune-House-Price.pkl               # Pickled sklearn model (training artefact)
├── requirements.txt                   # Python dependencies
└── .gitignore
```

---

## ✨ Features

- **Exploratory Data Analysis** — data cleaning, outlier removal, and feature engineering in the Jupyter notebook
- **Linear Regression model** trained with scikit-learn on Pune housing data
- **Lightweight inference** — the Streamlit app reconstructs predictions using only NumPy (no sklearn dependency at runtime)
- **Interactive UI** — select location, BHK, bathrooms, balconies, square footage, area type, and availability to get an instant price estimate in **lakhs (₹)**

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
git clone https://github.com/sayampalrecha/Pune-price.git
cd Pune-price
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```
---

## 🧠 How It Works

### Training (Notebook)

The Jupyter notebook (`Pune house price prediction.ipynb`) covers:

1. Loading and inspecting `Pune_House_Data.csv`
2. Cleaning missing values and removing outliers
3. One-hot encoding of categorical features (location, area type, availability)
4. Training a **Linear Regression** model using scikit-learn
5. Evaluating model performance

### Exporting the Model

After training, run the export script to serialise the model weights into NumPy-compatible files:

```bash
python export_model_assets.py
```

This generates:
- `model_weights.npz` — linear coefficients and intercept
- `model_features.json` — ordered list of feature column names

### Inference (App)

`app.py` loads the `.npz` and `.json` files and performs a pure-NumPy dot product for prediction — no sklearn import needed at runtime.

---

## 🖥️ App Inputs

| Field        | Description                                      |
|--------------|--------------------------------------------------|
| Location     | Neighbourhood in Pune (or "other")              |
| BHK          | Number of bedrooms (1–20)                       |
| Bathrooms    | Number of bathrooms (1–10)                      |
| Balconies    | Number of balconies (1–3)                       |
| Total sqft   | Total area in square feet                        |
| Area type    | Super built-up, Built-up, Carpet, or Plot Area  |
| Availability | Ready To Move / Not Ready                        |

The predicted price is displayed in **Indian Rupees (lakhs)**.

---

## 📦 Dependencies

| Package       | Purpose                        |
|---------------|--------------------------------|
| `numpy`       | Inference & numerical ops      |
| `streamlit`   | Web UI                         |
| `pandas`      | Data processing (notebook)     |
| `scikit-learn`| Model training (notebook only) |

---
