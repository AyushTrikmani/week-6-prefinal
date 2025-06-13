# Import the required libraries: Streamlit, NumPy, and Pillow (PIL).
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import requests
import os
import warnings

# Suppress sklearn warnings for compatibility
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Try to import gdown, fallback to requests if not available
try:
    import gdown
    USE_GDOWN = True
except ImportError:
    USE_GDOWN = False

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

# Display current environment info
with st.expander("🔧 Environment Info (Click to expand)"):
    try:
        import sklearn
        st.write(f"**Scikit-learn version:** {sklearn.__version__}")
    except:
        st.write("**Scikit-learn:** Not available")
    
    try:
        import joblib
        st.write(f"**Joblib version:** {joblib.__version__}")
    except:
        st.write("**Joblib:** Not available")
    
    try:
        st.write(f"**NumPy version:** {np.__version__}")
    except:
        st.write("**NumPy:** Not available")

# Caching the model for faster loading
@st.cache_resource
def load_model():
    """Download model from Google Drive and load it with compatibility handling"""
    
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
        except Exception as e:
            st.warning(f"⚠️ Local model loading failed: {str(e)}")
            # Continue to download from Google Drive
    
    # Download model from Google Drive
    st.info("📥 Downloading model from Google Drive...")
    
    download_success = False
    
    if USE_GDOWN:
        # Method 1: Use gdown (more reliable for large files)
        try:
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, LOCAL_MODEL_PATH, quiet=False)
            download_success = True
            st.info("✅ Download completed with gdown")
            
        except Exception as e:
            st.warning(f"⚠️ gdown download failed: {str(e)}")
            # Fall back to requests method
    
    if not download_success:
        # Method 2: Use requests with better error handling
        try:
            # Use session to handle redirects properly
            session = requests.Session()
            
            # First request to get the download URL
            url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
            response = session.get(url, stream=True)
            
            # Check if we need to handle the "large file" confirmation
            if 'confirm=' in response.text or 'download_warning' in response.text:
                # Try to extract confirmation token
                import re
                confirm_token = re.search(r'confirm=([^&"]+)', response.text)
                if confirm_token:
                    confirm_url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}&confirm={confirm_token.group(1)}"
                    response = session.get(confirm_url, stream=True)
                else:
                    # Try alternative method for large files
                    confirm_url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}&confirm=t"
                    response = session.get(confirm_url, stream=True)
            
            # Check if we got a proper response
            if response.status_code != 200:
                st.error(f"❌ Failed to download. Status code: {response.status_code}")
                return None
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type and len(response.content) < 10000:
                st.error("❌ Received HTML instead of file. Please check if your Google Drive link is correct.")
                st.info("Make sure your file is shared as 'Anyone with the link can view'")
                return None
            
            # Save the downloaded file
            total_size = 0
            with open(LOCAL_MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
            
            st.info(f"✅ Downloaded {total_size:,} bytes")
            download_success = True
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error downloading model: {str(e)}")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected download error: {str(e)}")
            return None
    
    if not download_success:
        st.error("❌ All download methods failed")
        return None
    
    # Verify the file is not empty
    file_size = os.path.getsize(LOCAL_MODEL_PATH)
    if file_size < 1000:
        st.error("❌ Downloaded file is too small. Check your Google Drive sharing settings.")
        return None
    
    # Load the model with compatibility handling
    try:
        with open(LOCAL_MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        
        st.success(f"✅ Model loaded successfully! Size: {file_size:,} bytes")
        return model
        
    except ValueError as e:
        if "incompatible dtype" in str(e) or "node array" in str(e):
            st.error("❌ **Scikit-learn Version Compatibility Issue**")
            st.error("Your model was trained with a different version of scikit-learn.")
            st.info("""
            **Solutions:**
            1. Check if your requirements.txt has the correct scikit-learn version
            2. Try retraining your model with the current environment
            3. Use joblib instead of pickle for better compatibility
            """)
        else:
            st.error(f"❌ Model loading error: {str(e)}")
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("This might be due to version incompatibility or file corruption.")
        return None

# Load the cached model
with st.spinner("Loading model... This may take a moment for first-time download."):
    voting_model = load_model()

# Stop execution if model loading failed
if voting_model is None:
    st.error("❌ Unable to load the model. Please check the troubleshooting info above.")
    st.info("**Troubleshooting Steps:**")
    st.info("1. Ensure your Google Drive file is shared properly")
    st.info("2. Check that requirements.txt has the correct scikit-learn version")
    st.info("3. Try refreshing the page")
    st.stop()

# Define the function for the wait time predictor using the loaded model.
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

# Define the input parameters using Streamlit's sidebar.
with st.sidebar:
    # Try to load image, if not available, show a placeholder
    try:
        img = Image.open("./assets/supply_chain_optimisation.jpg")
        st.image(img)
    except:
        st.markdown("### 📦 Supply Chain Optimization")
    
    st.header("Input Parameters")
    purchase_dow = st.number_input(
        "Purchased Day of the Week", min_value=0, max_value=6, step=1, value=3,
        help="0=Monday, 1=Tuesday, ..., 6=Sunday"
    )
    purchase_month = st.number_input(
        "Purchased Month", min_value=1, max_value=12, step=1, value=1
    )
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size in cm³", value=9328)
    product_weight_g = st.number_input("Product Weight in grams", value=1800)
    geolocation_state_customer = st.number_input(
        "Geolocation State of the Customer", value=10
    )
    geolocation_state_seller = st.number_input(
        "Geolocation State of the Seller", value=20
    )
    distance = st.number_input("Distance (km)", value=475.35)
    
    # Submit button
    submit = st.button("Predict Wait Time", type="primary")

# Main content area
with st.container():
    # Define the output container for the predicted wait time.
    st.header("Output: Wait Time in Days")

    # When the submit button is clicked, call the wait time predictor function and display the predicted wait time.
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
            with st.spinner(text="Making prediction..."):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.success(f"🎯 **Predicted Wait Time: {prediction} days**")
                    
                    # Add some context
                    if prediction <= 5:
                        st.info("✅ Fast delivery expected!")
                    elif prediction <= 10:
                        st.info("📦 Standard delivery time")
                    else:
                        st.warning("⏳ Longer delivery time - consider optimizing")
        else:
            st.error("Unable to make prediction. Please check your inputs and try again.")

    # Define a sample dataset for demonstration purposes.
    st.header("Sample Dataset")
    data = {
        "Purchased Day of the Week": ["0", "3", "1"],
        "Purchased Month": ["6", "3", "1"],
        "Purchased Year": ["2018", "2017", "2018"],
        "Product Size in cm³": ["37206.0", "63714", "54816"],
        "Product Weight in grams": ["16250.0", "7249", "9600"],
        "Geolocation State Customer": ["25", "25", "25"],
        "Geolocation State Seller": ["20", "7", "20"],
        "Distance (km)": ["247.94", "250.35", "4.915"],
    }

    # Create a DataFrame from the sample dataset.
    df = pd.DataFrame(data)

    # Display the sample dataset in the Streamlit app.
    st.dataframe(df, use_container_width=True)
