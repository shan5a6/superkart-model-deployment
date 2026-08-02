import os
import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="SuperKart Sales Forecasting",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 SuperKart Sales Forecasting App")

st.write(
    "This application predicts product-store level sales revenue using the SuperKart machine learning backend."
)

st.info(
    "For Docker Compose deployment, the default backend URL is `http://backend:5000`. "
    "For local manual testing, use `http://127.0.0.1:5000`."
)

backend_url = st.sidebar.text_input(
    "Backend API URL",
    value=os.getenv(
        "BACKEND_URL",
        os.getenv("BACKEND_API_URL", "http://127.0.0.1:5000")
    )
)

st.sidebar.markdown(
    '''
    **Backend URL guidance**

    - Docker Compose: `http://backend:5000`
    - Manual local testing: `http://127.0.0.1:5000`
    - Public Codespaces test: use forwarded backend URL for port `5000`
    '''
)

mode = st.radio("Choose Prediction Mode", ["Single Prediction", "Batch Prediction"])

if mode == "Single Prediction":
    st.header("Single Prediction")

    col1, col2 = st.columns(2)

    with col1:
        Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
        Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027, format="%.3f")
        Product_MRP = st.number_input("Product MRP", min_value=0.0, value=117.08)
        Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])

    with col2:
        Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        Store_Type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
        Product_Id_char = st.selectbox("Product Id Prefix", ["FD", "DR", "NC"])
        Store_Age_Years = st.number_input("Store Age in Years", min_value=0, value=16)
        Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

    input_data = {
        "Product_Weight": Product_Weight,
        "Product_Sugar_Content": Product_Sugar_Content,
        "Product_Allocated_Area": Product_Allocated_Area,
        "Product_MRP": Product_MRP,
        "Store_Size": Store_Size,
        "Store_Location_City_Type": Store_Location_City_Type,
        "Store_Type": Store_Type,
        "Product_Id_char": Product_Id_char,
        "Store_Age_Years": Store_Age_Years,
        "Product_Type_Category": Product_Type_Category
    }

    if st.button("Predict Sales"):
        try:
            response = requests.post(
                f"{backend_url}/predict",
                json=input_data,
                timeout=30
            )

            if response.status_code == 200:
                prediction = response.json()["prediction"]
                st.success(f"Predicted Product Store Sales Total: {prediction:,.2f}")
            else:
                st.error(response.json())

        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

else:
    st.header("Batch Prediction")

    uploaded_file = st.file_uploader("Upload Batch CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        st.write("Uploaded Batch Data")
        st.dataframe(batch_df.head())

        if st.button("Run Batch Prediction"):
            try:
                records = batch_df.to_dict(orient="records")

                response = requests.post(
                    f"{backend_url}/batch_predict",
                    json={"records": records},
                    timeout=60
                )

                if response.status_code == 200:
                    predictions = response.json()["predictions"]
                    output_df = pd.DataFrame(predictions)

                    st.success("Batch prediction completed successfully.")
                    st.dataframe(output_df)

                    csv = output_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv,
                        file_name="SuperKart_Batch_Predictions.csv",
                        mime="text/csv"
                    )

                else:
                    st.error(response.json())

            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
