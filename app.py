import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Monthly Units Sold Predictor", page_icon="📊", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

if not os.path.exists("model.pkl"):
    st.error("model.pkl was not found. Place model.pkl in the same folder as app.py.")
    st.stop()

model = load_model()

st.title("📊 Monthly Units Sold Predictor")
st.write("Predict expected **Monthly Units Sold** using the trained Multiple Linear Regression model.")
st.info("Enter the product and market characteristics below and click **Predict Monthly Units Sold**.")

st.subheader("Product Information")
product_category = st.selectbox("Product Category", ["Grocery", "Clothing", "Electronics", "Personal Care", "Home Appliances", "Sports"])
brand_tier = st.selectbox("Brand Tier", ["Economy", "Mid-Range", "Premium"])

st.subheader("Product & Market Variables")
unit_price = st.number_input("Unit Price", min_value=0.01, value=100.0, step=10.0)
discount_percentage = st.number_input("Discount Percentage (decimal)", min_value=0.0, max_value=0.40, value=0.10, step=0.01, help="Enter as a decimal. Example: 0.20 means a 20% discount.")
advertising_spend = st.number_input("Advertising Spend", min_value=500.0, value=5000.0, step=500.0)
store_coverage_count = st.number_input("Store Coverage Count", min_value=1, value=100, step=10)
customer_rating = st.number_input("Customer Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
inventory_availability = st.number_input("Inventory Availability (decimal)", min_value=0.0, max_value=1.0, value=0.90, step=0.01, help="Enter as a decimal. Example: 0.90 means 90% availability.")

st.divider()

if st.button("Predict Monthly Units Sold", type="primary"):
    input_data = pd.DataFrame({
        "Product_Category": [product_category],
        "Brand_Tier": [brand_tier],
        "Unit_Price": [unit_price],
        "Discount_Percentage": [discount_percentage],
        "Advertising_Spend": [advertising_spend],
        "Store_Coverage_Count": [store_coverage_count],
        "Customer_Rating": [customer_rating],
        "Inventory_Availability_Percentage": [inventory_availability]
    })

    try:
        prediction = max(0, float(model.predict(input_data)[0]))
        st.success("Prediction generated successfully!")
        st.metric("Predicted Monthly Units Sold", f"{prediction:,.0f} units")

        st.subheader("Input Summary")
        display_data = input_data.copy()
        display_data["Discount (%)"] = display_data["Discount_Percentage"] * 100
        display_data["Inventory Availability (%)"] = display_data["Inventory_Availability_Percentage"] * 100
        display_data = display_data.drop(columns=["Discount_Percentage", "Inventory_Availability_Percentage"])
        st.dataframe(display_data, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error("Prediction could not be generated.")
        st.exception(e)

st.divider()
st.subheader("About the Model")
st.write("The application uses the final Multiple Linear Regression pipeline. The saved pipeline contains the required categorical encoding and logarithmic transformations.")
st.caption("Product_ID and Competitor_Price are not used by the final prediction model.")
