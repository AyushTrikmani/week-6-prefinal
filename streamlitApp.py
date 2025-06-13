# Import the required libraries: Streamlit, NumPy, and Pillow (PIL).
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import requests
import os
import warnings
import sys

# Suppress sklearn warnings for compatibility
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Try to import required packages with fallbacks
missing_packages = []

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    missing_packages.append("joblib")

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    missing_packages.append("scikit-learn")

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

# Check for missing packages and display warnings
if missing_packages:
    st.error(f"❌ Missing required packages: {', '.join(missing_packages)}")
    st.info("Please install missing packages using: `pip install " + " ".join(missing_packages) + "`")
    if not SKLEARN_AVAILABLE:
        st.stop()

# Google Drive file ID for the model
GOOGLE_DRIVE_FILE_ID = "1dmuDcdvi1wo92TtxZxvgX00WZpW_zHeN"
LOCAL_MODEL_PATH = "./voting_model.pkl"

# Display current environment info
with st.expander("🔧 Environment Info (Click to expand)"):
    st.write(f"**Python version:** {sys.version}")
    
    if SKLEARN_AVAILABLE:
        try:
            st.write(f"**Scikit-learn version:** {sklearn.__version__}")
        except:
            st.write("**Scikit-learn:** Available but version unknown")
    else:
        st.write("**Scikit-learn:** ❌ Not available")
    
    if JOBLIB_AVAILABLE:
        try:
            st.write(f"**Joblib version:** {joblib.__version__}")
        except:
            st.write("**Joblib:** Available but version unknown")
    else:
        st.write("**Joblib:** ❌ Not available")
    
    try:
        st.write(f"**NumPy version:** {np.__version__}")
    except:
        st.write("**NumPy:** Available but version unknown")
    
    try:
        st.write(f"**Pandas version:** {pd.__version__}")
    except:
        st.write("**Pandas:** Available but version unknown")
    
    st.write(f"**Gdown available:** {'✅ Yes' if USE_GDOWN else '❌ No'}")

def download_with_requests(file_id, output_path):
    """Download file from Google Drive using requests with better error handling"""
    try:
        session = requests.Session()
        
        # Try direct download first
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = session.get(url, stream=True)
        
        # Handle large file confirmation
        if 'confirm=' in response.text or 'download_warning' in response.text:
            import re
            confirm_match = re.search(r'confirm=([^&"]+)', response.text)
            
            if confirm_match:
                confirm_token = confirm_match.group(1)
                confirm_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                response = session.get(confirm_url, stream=True)
            else:
                # Alternative method for large files
                confirm_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                response = session.get(confirm_url, stream=True)
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}")
        
        # Check if we got HTML instead of the file
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            content_preview = response.content[:500].decode('utf-8', errors='ignore')
            if 'Google Drive' in content_preview or 'download' in content_preview.lower():
                raise Exception("Received HTML page instead of file. Check sharing permissions.")
        
        # Save the file
        total_size = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        return total_size
        
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

def load_model_with_compatibility():
    """Load model with multiple fallback methods for compatibility"""
    
    if not os.path.exists(LOCAL_MODEL_PATH):
        return None, "Model file not found"
    
    file_size = os.path.getsize(LOCAL_MODEL_PATH)
    if file_size < 1000:
        return None, f"Model file too small ({file_size} bytes)"
    
    # Method 1: Try joblib first (recommended for sklearn models)
    if JOBLIB_AVAILABLE:
        try:
            model = joblib.load(LOCAL_MODEL_PATH)
            return model, f"Loaded with joblib ({file_size:,} bytes)"
        except Exception as e:
            st.warning(f"Joblib loading failed: {str(e)}")
    
    # Method 2: Try pickle
    try:
        with open(LOCAL_MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        return model, f"Loaded with pickle ({file_size:,} bytes)"
    except Exception as e:
        return None, f"Pickle loading failed: {str(e)}"

# Caching the model for faster loading
@st.cache_resource
def load_model():
    """Download model from Google Drive and load it with compatibility handling"""
    
    # Check if model already exists locally and try to load it
    if os.path.exists(LOCAL_MODEL_PATH):
        model, status = load_model_with_compatibility()
        if model is not None:
            st.success(f"✅ {status}")
            return model
        else:
            st.warning(f"⚠️ {status}")
    
    # Download model from Google Drive
    st.info("📥 Downloading model from Google Drive...")
    
    download_success = False
    download_size = 0
    
    # Method 1: Use gdown (more reliable for large files)
    if USE_GDOWN:
        try:
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, LOCAL_MODEL_PATH, quiet=False)
            download_size = os.path.getsize(LOCAL_MODEL_PATH)
            download_success = True
            st.info(f"✅ Downloaded {download_size:,} bytes with gdown")
            
        except Exception as e:
            st.warning(f"⚠️ gdown download failed: {str(e)}")
    
    # Method 2: Use requests as fallback
    if not download_success:
        try:
            download_size = download_with_requests(GOOGLE_DRIVE_FILE_ID, LOCAL_MODEL_PATH)
            download_success = True
            st.info(f"✅ Downloaded {download_size:,} bytes with requests")
            
        except Exception as e:
            st.error(f"❌ {str(e)}")
            return None
    
    if not download_success:
        st.error("❌ All download methods failed")
        return None
    
    # Verify download
    if download_size < 1000:
        st.error("❌ Downloaded file is too small. Check Google Drive sharing settings.")
        st.info("Make sure your file is shared as 'Anyone with the link can view'")
        return None
    
    # Load the downloaded model
    model, status = load_model_with_compatibility()
    if model is not None:
        st.success(f"✅ {status}")
        return model
    else:
        st.error(f"❌ {status}")
        st.info("""
        **Possible solutions:**
        1. Update scikit-learn: `pip install scikit-learn>=1.3.0`
        2. Install setuptools: `pip install setuptools`
        3. Retrain your model with the current environment
        4. Use joblib for saving models instead of pickle
        """)
        return None

# Load the cached model
with st.spinner("Loading model... This may take a moment for first-time download."):
    voting_model = load_model()

# Stop execution if model loading failed
if voting_model is None:
    st.error("❌ Unable to load the model. Please check the troubleshooting info above.")
    
    with st.expander("🔧 Troubleshooting Steps"):
        st.markdown("""
        **Common Issues and Solutions:**
        
        1. **Version Compatibility Issues:**
           - Update packages: `pip install scikit-learn>=1.3.0 joblib>=1.2.0 setuptools`
           - Use Python 3.11 instead of 3.13 if possible
        
        2. **Google Drive Access Issues:**
           - Ensure file is shared as "Anyone with the link can view"
           - Check if the file ID is correct
           - Try downloading manually and placing in the same directory
        
        3. **Package Installation Issues:**
           - Install setuptools: `pip install setuptools`
           - Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
        
        4. **Alternative Model Loading:**
           - If you have the model file, place it as `voting_model.pkl` in the app directory
           - Consider retraining the model with current package versions
        """)
    
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
        # Prepare input data
        input_data = np.array([[
            purchase_dow,
            purchase_month,
            year,
            product_size_cm3,
            product_weight_g,
            geolocation_state_customer,
            geolocation_state_seller,
            distance,
        ]])
        
        # Make prediction
        prediction = voting_model.predict(input_data)
        
        # Return rounded prediction
        return round(prediction[0]) if prediction[0] > 0 else 1
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("This might be due to model incompatibility or input data issues.")
        return None

# Define the input parameters using Streamlit's sidebar.
with st.sidebar:
    # Try to load image, if not available, show a placeholder
    try:
        img = Image.open("./assets/supply_chain_optimisation.jpg")
        st.image(img)
    except:
        st.markdown("### 📦 Supply Chain Optimization")
        st.info("💡 Add an image at `./assets/supply_chain_optimisation.jpg` for better visuals")
    
    st.header("Input Parameters")
    
    # Input fields with better validation and help text
    purchase_dow = st.number_input(
        "Purchased Day of the Week", 
        min_value=0, max_value=6, step=1, value=3,
        help="0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday"
    )
    
    purchase_month = st.number_input(
        "Purchased Month", 
        min_value=1, max_value=12, step=1, value=1,
        help="1=January, 2=February, ..., 12=December"
    )
    
    year = st.number_input(
        "Purchased Year", 
        min_value=2016, max_value=2025, step=1, value=2018,
        help="Year when purchase was made"
    )
    
    product_size_cm3 = st.number_input(
        "Product Size in cm³", 
        min_value=1.0, step=100.0, value=9328.0,
        help="Volume of the product in cubic centimeters"
    )
    
    product_weight_g = st.number_input(
        "Product Weight in grams", 
        min_value=1.0, step=100.0, value=1800.0,
        help="Weight of the product in grams"
    )
    
    geolocation_state_customer = st.number_input(
        "Geolocation State of the Customer", 
        min_value=0, step=1, value=10,
        help="Encoded state ID where customer is located"
    )
    
    geolocation_state_seller = st.number_input(
        "Geolocation State of the Seller", 
        min_value=0, step=1, value=20,
        help="Encoded state ID where seller is located"
    )
    
    distance = st.number_input(
        "Distance (km)", 
        min_value=0.1, step=10.0, value=475.35,
        help="Distance between customer and seller in kilometers"
    )
    
    # Submit button
    submit = st.button("🎯 Predict Wait Time", type="primary", use_container_width=True)

# Main content area
with st.container():
    # Define the output container for the predicted wait time.
    st.header("📊 Prediction Results")

    # When the submit button is clicked, call the wait time predictor function and display the predicted wait time.
    if submit:
        with st.spinner("🔮 Making prediction..."):
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
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Display prediction with context
                st.success(f"🎯 **Predicted Wait Time: {prediction} days**")
                
                # Add interpretation
                if prediction <= 5:
                    st.info("✅ **Fast delivery expected!** This order should arrive quickly.")
                elif prediction <= 10:
                    st.info("📦 **Standard delivery time** - Within expected range.")
                elif prediction <= 15:
                    st.warning("⏳ **Longer delivery time** - Consider optimization opportunities.")
                else:
                    st.error("🚨 **Very long delivery time** - Significant delays expected.")
                
                # Show input summary
                with st.expander("📋 Input Summary"):
                    input_df = pd.DataFrame({
                        'Parameter': [
                            'Day of Week', 'Month', 'Year', 'Product Size (cm³)', 
                            'Product Weight (g)', 'Customer State', 'Seller State', 'Distance (km)'
                        ],
                        'Value': [
                            f"{purchase_dow} ({'Mon,Tue,Wed,Thu,Fri,Sat,Sun'.split(',')[purchase_dow]})",
                            f"{purchase_month} ({['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][purchase_month-1]})",
                            year, f"{product_size_cm3:,.0f}", f"{product_weight_g:,.0f}",
                            geolocation_state_customer, geolocation_state_seller, f"{distance:.2f}"
                        ]
                    })
                    st.dataframe(input_df, use_container_width=True, hide_index=True)
        else:
            st.error("❌ Unable to make prediction. Please check your inputs and model status.")

    # Define a sample dataset for demonstration purposes.
    st.header("📋 Sample Dataset")
    st.info("Here are some example inputs you can try:")
    
    data = {
        "Day of Week": ["0 (Mon)", "3 (Thu)", "1 (Tue)"],
        "Month": ["6 (Jun)", "3 (Mar)", "1 (Jan)"],
        "Year": ["2018", "2017", "2018"],
        "Product Size (cm³)": ["37,206", "63,714", "54,816"],
        "Product Weight (g)": ["16,250", "7,249", "9,600"],
        "Customer State": ["25", "25", "25"],
        "Seller State": ["20", "7", "20"],
        "Distance (km)": ["247.94", "250.35", "4.91"],
    }

    # Create a DataFrame from the sample dataset.
    df = pd.DataFrame(data)

    # Display the sample dataset in the Streamlit app.
    st.dataframe(df, use_container_width=True)
    
    # Add footer information
    st.markdown("---")
    st.markdown(
        "*Timelytics v2.0 - Enhanced compatibility and error handling*  \n"
        "💡 **Tip:** If you encounter issues, check the Environment Info section above."
    )
