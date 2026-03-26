import json
from pathlib import Path

import numpy as np
import streamlit as st

BASE = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE / "model_weights.npz"
FEATURES_PATH = BASE / "model_features.json"


def load_trained_model():
    """Load linear model exported from sklearn (no sklearn required at runtime)."""
    data = np.load(WEIGHTS_PATH)
    coef = np.asarray(data["coef"], dtype=np.float64)
    intercept = float(np.asarray(data["intercept"]))
    with open(FEATURES_PATH, encoding="utf-8") as f:
        columns = json.load(f)
    if coef.shape[0] != len(columns):
        raise ValueError("coef length does not match feature list")
    return coef, intercept, columns


def predict_price(coef, intercept, columns, location, bhk, bath, balcony, sqft, area_type, availability):
    """Match notebook logic: one-hot location / availability / area_type; reference categories left as 0."""
    cols = np.array(columns)
    loc_index = area_index = avail_index = -1

    if location != "other":
        loc_index = int(np.where(cols == location)[0][0])
    if area_type != "Super built-up  Area":
        area_index = int(np.where(cols == area_type)[0][0])
    if availability != "Not Ready":
        avail_index = int(np.where(cols == availability)[0][0])

    x = np.zeros(len(columns))
    x[0] = float(bath)
    x[1] = float(balcony)
    x[2] = float(bhk)
    x[3] = float(sqft)

    if loc_index >= 0:
        x[loc_index] = 1
    if area_index >= 0:
        x[area_index] = 1
    if avail_index >= 0:
        x[avail_index] = 1

    return float(np.dot(coef, x) + intercept)


st.set_page_config(page_title="Pune House Price", layout="wide")
st.title("Pune House Price Prediction App")

try:
    coef, intercept, columns = load_trained_model()
except Exception as e:
    st.error(f"Could not load model from **{WEIGHTS_PATH}** / **{FEATURES_PATH}**: {e}")
    st.info(
        "If you retrained the notebook, run `python export_model_assets.py` from this folder "
        "(needs a working sklearn once) to regenerate the weights."
    )
    st.stop()

idx_ready = columns.index("Ready To Move")
location_options = ["other"] + columns[4:idx_ready]

area_type_options = [
    "Super built-up  Area",
    "Built-up  Area",
    "Carpet  Area",
    "Plot  Area",
]
availability_options = ["Not Ready", "Ready To Move"]

st.write("Enter the following details:")

left, right = st.columns((2, 2))
with right:
    bhk = st.number_input("BHK (bedrooms)", min_value=1, max_value=20, value=2, step=1)
    st.write("You selected:", bhk)

with left:
    location = st.selectbox(
        "Location",
        location_options,
    )
    st.write("You selected:", location)

with left:
    balcony = st.selectbox("Balconies", ("1", "2", "3"))
    st.write("You selected:", balcony)

with right:
    bath = st.selectbox("Bathrooms", [str(i) for i in range(1, 11)])
    st.write("You selected:", bath)

with left:
    sqft = st.number_input("Total sqft", min_value=1.0, value=1000.0, step=1.0)
    st.write("You selected:", sqft)

with right:
    area_type = st.selectbox("Area type", area_type_options)
    st.write("You selected:", area_type)

with left:
    availability = st.selectbox("Availability", availability_options)
    st.write("You selected:", availability)

if st.button("Predict"):
    price = predict_price(
        coef,
        intercept,
        columns,
        location,
        bhk,
        bath,
        balcony,
        sqft,
        area_type,
        availability,
    )
    st.success(f"Estimated price: **{price:.2f} lakh** (model target is price in lakhs).")
