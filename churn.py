import streamlit as st
import pickle
import pandas as pd

# Load scaler, model, and model columns
scaler = pickle.load(open('scalerxg11.sav', 'rb'))
model = pickle.load(open('Random_model111.sav', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

# Custom CSS
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://wallpaperaccess.com/full/2482036.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
.stButton>button {
    border-radius: 10px;
    padding: 10px 24px;
    font-size: 16px;
    background-color: #1f77b4;
    color: white;
}
.stButton>button:hover {
    background-color: #105090;
}
.stMarkdown h3 {
    color: #fff;
}
.result-box {
    background-color: rgba(255,255,255,0.9);
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# Input helper functions
def safe_selectbox(label, options, key):
    return st.selectbox(label, options, key=key)


def safe_float_input(label, key):
    value = st.text_input(label, key=key)
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        st.error(f"Invalid input for {label}. Please enter a valid number.")
        return None


def main():
    st.markdown("<h2 style='text-align:center;'>TELCO CUSTOMER CHURN PREDICTOR</h2>", unsafe_allow_html=True)

    with st.expander("ℹ️ About this App"):
        st.info(
            "This intelligent tool leverages machine learning to identify customers who are likely to discontinue Telco services. "
            "By analyzing patterns in customer behavior, usage, and demographics, it provides real-time predictions to help telecom businesses:")

    col1, col2 = st.columns(2)

    with col1:
        gender = safe_selectbox("Gender", ["Female", "Male"], "gender")
        SeniorCitizen = safe_selectbox("Senior Citizen", ["No", "Yes"], "SeniorCitizen")
        Partner = safe_selectbox("Has Partner?", ["No", "Yes"], "Partner")
        Dependents = safe_selectbox("Has Dependents?", ["No", "Yes"], "Dependents")
        tenure = safe_float_input("Tenure (months)", "tenure")
        PhoneService = safe_selectbox("Phone Service", ["No", "Yes"], "PhoneService")
        MultipleLines = safe_selectbox("Multiple Lines", ["No", "Yes", "No phone service"], "MultipleLines")
        InternetService = safe_selectbox("Internet Service", ["DSL", "Fiber optic", "No"], "InternetService")
        OnlineSecurity = safe_selectbox("Online Security", ["No", "Yes", "No internet service"], "OnlineSecurity")

    with col2:
        OnlineBackup = safe_selectbox("Online Backup", ["No", "Yes", "No internet service"], "OnlineBackup")
        DeviceProtection = safe_selectbox("Device Protection", ["No", "Yes", "No internet service"], "DeviceProtection")
        TechSupport = safe_selectbox("Tech Support", ["No", "Yes", "No internet service"], "TechSupport")
        StreamingTV = safe_selectbox("Streaming TV", ["No", "Yes", "No internet service"], "StreamingTV")
        StreamingMovies = safe_selectbox("Streaming Movies", ["No", "Yes", "No internet service"], "StreamingMovies")
        Contract = safe_selectbox("Contract", ["Month-to-month", "One year", "Two year"], "Contract")
        PaperlessBilling = safe_selectbox("Paperless Billing", ["No", "Yes"], "PaperlessBilling")
        PaymentMethod = safe_selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], "PaymentMethod")
        MonthlyCharges = safe_float_input("Monthly Charges", "MonthlyCharges")
        TotalCharges = safe_float_input("Total Charges", "TotalCharges")

    # Collect input dictionary
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if SeniorCitizen == "Yes" else 0,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    # Check if any input is None (invalid or missing)
    inputs_valid = all(v is not None for v in input_data.values())

    # Disable button if inputs are invalid
    predict_button = st.button("🔮 Predict Churn", disabled=not inputs_valid)

    if predict_button:
        if not inputs_valid:
            st.error("❌ Please fill in all fields with valid inputs before predicting.")
            return

        df = pd.DataFrame([input_data])

        # Binary encode specific columns
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            df[col] = df[col].map({"No": 0, "Yes": 1, "Female": 0, "Male": 1})

        # One-hot encode remaining categorical columns
        df = pd.get_dummies(df)

        # Align to model columns
        df = df.reindex(columns=model_columns, fill_value=0)

        # Debug outputs
        st.write("🧪 Model columns count:", len(model_columns))
        st.write("🧪 Input columns count:", len(df.columns))
        st.write("🧪 Missing columns (should be empty):", set(model_columns) - set(df.columns))
        st.write("🧪 Input preview before scaling:")
        st.dataframe(df)

        try:
            # Scale input
            scaled_input = scaler.transform(df)
            st.write("🧪 Scaled input preview:", scaled_input)

            # Predict churn
            prediction = model.predict(scaled_input)[0]
            st.write("🧪 Prediction result:", prediction)

            if prediction == 0:
                st.markdown(f"""
                <div class="result-box">
                    <h3>🟢 Customer is likely to <u>stay</u>.</h3>
                    <p>Great! Keep focusing on customer engagement.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box">
                    <h3>🔴 Customer is likely to <u>churn</u>.</h3>
                    <p>Consider offering retention incentives or better support.</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")


if __name__ == "__main__":
    main()
