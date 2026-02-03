import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")

# 1) Load your trained pipeline ONCE (important because Streamlit reruns scripts)
@st.cache_resource
def load_pipeline():
    return joblib.load("churn_pipeline.joblib")

pipe = load_pipeline()
# print(pipe)
# print(pipe.feature_names_in_) //Returns all the features that are used in X_Train

st.title("📉 Customer Churn Prediction")
st.write("Enter customer info → get churn prediction and probability.")

# 2) We need the EXACT feature columns used during training
# If your pipeline was trained on a DataFrame, sklearn stores feature names here:
feature_names = None
if hasattr(pipe, "feature_names_in_"):
    feature_names = list(pipe.feature_names_in_)
    # print(type(feature_names))

# If we can't auto-detect, we need you to paste X_train.columns from notebook
if not feature_names:
    st.warning("I couldn't auto-detect your training feature names.")
    st.info("In your notebook run: X_train.columns.tolist() and paste it into app.py.")
    st.stop()

feature_names = [x.lower() for x in feature_names]
print(feature_names)
st.subheader("Inputs")

# 3) Build an input form dynamically
# We'll create a dict of inputs, then convert to a 1-row DataFrame
inputs = {}
with st.form("churn_form"):
    for col in feature_names:
        # Simple heuristic: treat common numeric columns as numeric, rest as text
        # You'll refine this after first run (I’ll show how below)
        if col.lower() in {"tenure", "monthlycharges", "totalcharges"}:
            inputs[col] = st.number_input(col, value=0.0)
        elif col.lower() in {"gender"}:
            inputs[col] = st.selectbox("Gender",["Male","Female"])
        elif col.lower() in {"partner","dependents","phoneservice","paperlessbilling"}:
            inputs[col] = st.selectbox(col,["Yes","No"])
        elif col.lower() in {"seniorcitizen"}:
            inputs[col] = st.selectbox(col,["1","0"])
        elif col.lower() in {"paymentmethod"}:
            inputs[col] = st.selectbox(col,["Bank Transfer","Credit Card","Electronic check","Mailed check"])

        else:
            inputs[col] = st.text_input(col, value="")

    submitted = st.form_submit_button("Predict")

# 4) Predict only on submit
if submitted:
    X_user = pd.DataFrame([inputs])
    print(X_user)
    # Prediction
    pred = pipe.predict(X_user)[0]

    # Probability (LogisticRegression supports this)
    prob = None
    if hasattr(pipe, "predict_proba"):
        prob = pipe.predict_proba(X_user)[0][1]

    st.divider()
    if int(pred) == 1:
        st.error("Prediction: Likely to churn")
    else:
        st.success("Prediction: Likely to stay")

    if prob is not None:
        st.write(f"Churn probability: **{prob:.2%}**")

    # Debug view (useful early)
    with st.expander("See input row sent to model"):
        st.dataframe(X_user)
