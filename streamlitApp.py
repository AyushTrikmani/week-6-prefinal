# Import the required libraries: Streamlit, NumPy, and Pillow (PIL).
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import requests
import os

# Set the page configuration of the app, including the page title, icon, and layout.
st.set_page_config(
    page_title="Timelytics",
    page_icon="📦",
    layout="wide"
)

# Display the title and captions for the app.
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - XGBoost, Random Forests, and Support Vector Machines (SVM) - to accurately forecast Order to Delivery (OTD) times. By combining the strengths of these three algorithms, Timelytics provides a robust and reliable prediction of OTD times, helping businesses to optimize their supply chain operations."
)

st.caption(
    "With Timelytics, businesses can identify potential bottlenecks and delays in their supply chain and take proactive measures to address them, reducing lead times and improving delivery times. The model utilizes historical data on order processing times, production lead times, shipping times, and other relevant variables to generate accurate forecasts of OTD times. These forecasts can be used to optimize inventory management, improve customer service, and increase overall efficiency in the supply chain."
)

# Google Drive file ID for the model
GOOGLE_DRIVE_FILE_ID = "1dmuDcdvi1wo92TtxZxvgX00WZpW_zHeN"
LOCAL_MODEL_PATH = "./voting_model.pkl"

# Caching the model for faster loading
@st.cache_resource
def load_model():
    """Download model from Google Drive and load it"""
    
    # Check if model already exists locally
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            # Check if it's a valid pickle file (not a Git LFS pointer)
            file_size = os.path.getsize(LOCAL_MODEL_PATH)
            if file_size > 10000:  # If file is larger than 10KB, likely a real model
                with open(LOCAL_MODEL_PATH, 'rb') as file:
                    model = pickle.load(file)
                st.success(f"✅ Model loaded from local file! Size: {file_size:,} bytes")
                return model
        except:
            pass  # If loading fails, download from Google Drive
    
    # Download model from Google Drive
    st.info("📥 Downloading model from Google Drive...")
    
    try:
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
        
        # Download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save the downloaded file
        with open(LOCAL_MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Load the model
        with open(LOCAL_MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        
        file_size = os.path.getsize(LOCAL_MODEL_PATH)
        st.success(f"✅ Model downloaded and loaded successfully! Size: {file_size:,} bytes")
        return model
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error downloading model: {str(e)}")
        st.error("Please check your internet connection and try again.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load the cached model
with st.spinner("Loading model... This may take a moment for first-time download."):
    voting_model = load_model()

# Stop execution if model loading failed
if voting_model is None:
    st.error("❌ Unable to load the model. Please refresh the page and try again.")
    st.stop()

# Define the function for the wait time predictor using the loaded model. This function takes in the input parameters and returns a predicted wait time in days.
def waitime_predictor(
    purchase_dow,
    purchase_month,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer,
    geolocation_state_seller,
    distance,
):
    if voting_model is None:
        st.error("Model not available for prediction")
        return None
    
    try:
        prediction = voting_model.predict(
            np.array(
                [
                    [
                        purchase_dow,
                        purchase_month,
                        year,
                        product_size_cm3,
                        product_weight_g,
                        geolocation_state_customer,
                        geolocation_state_seller,
                        distance,
                    ]
                ]
            )
        )
        return round(prediction[0])
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Define the input parameters using Streamlit's sidebar. These parameters include the purchased day of the week, month, and year, product size, weight, geolocation state of the customer and seller, and distance.
with st.sidebar:
    # Try to load image, if not available, show a placeholder
    try:
        img = Image.open("./assets/supply_chain_optimisation.jpg")
        st.image(img)
    except:
        st.markdown("### 📦 Supply Chain Optimization")
    
    st.header("Input Parameters")
    purchase_dow = st.number_input(
        "Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3
    )
    purchase_month = st.number_input(
        "Purchased Month", min_value=1, max_value=12, step=1, value=1
    )
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm^3", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input(
        "Geolocation State of the Customer", value=10
    )
    geolocation_state_seller = st.number_input(
        "Geolocation State of the Seller", value=20
    )
    distance = st.number_input("Distance", value=475.35)
    
    # Submit button
    submit = st.button("Predict Wait Time", type="primary")

# Define the submit button for the input parameters.
with st.container():
    # Define the output container for the predicted wait time.
    st.header("Output: Wait Time in Days")

    # When the submit button is clicked, call the wait time predictor function and display the predicted wait time in the output container.
    if submit:
        prediction = waitime_predictor(
            purchase_dow,
            purchase_month,
            year,
            product_size_cm3,
            product_weight_g,
            geolocation_state_customer,
            geolocation_state_seller,
            distance,
        )
        if prediction is not None:
            with st.spinner(text="This may take a moment..."):
                st.success(f"Predicted Wait Time: {prediction} days")
        else:
            st.error("Unable to make prediction. Please check your inputs.")

    # Define a sample dataset for demonstration purposes.
    data = {
        "Purchased Day of the Week": ["0", "3", "1"],
        "Purchased Month": ["6", "3", "1"],
        "Purchased Year": ["2018", "2017", "2018"],
        "Product Size in cm^3": ["37206.0", "63714", "54816"],
        "Product Weight in grams": ["16250.0", "7249", "9600"],
        "Geolocation State Customer": ["25", "25", "25"],
        "Geolocation State Seller": ["20", "7", "20"],
        "Distance": ["247.94", "250.35", "4.915"],
    }

    # Create a DataFrame from the sample dataset.
    df = pd.DataFrame(data)

    # Display the sample dataset in the Streamlit app.
    st.header("Sample Dataset")
    st.write(df)
