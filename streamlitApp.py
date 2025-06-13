# Import the required libraries: Streamlit, NumPy, and Pillow (PIL).
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import os

# Set the page configuration of the app, including the page title, icon, and layout.
st.set_page_config(
    page_title="Timelytics",
    page_icon="📦",
    layout="wide"
)

# Function to load images safely
def load_image(image_path):
    """Load image if it exists, otherwise return None"""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            return None
    except Exception as e:
        st.warning(f"Could not load image {image_path}: {str(e)}")
        return None

# Load header image if available
header_image = load_image("./assets/header.jpg") or load_image("./assets/header.png") or load_image("./assets/supply_chain.jpg")

# Display header image if available
if header_image:
    st.image(header_image, use_column_width=True)

# Display the title and captions for the app.
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - XGBoost, Random Forests, and Support Vector Machines (SVM) - to accurately forecast Order to Delivery (OTD) times. By combining the strengths of these three algorithms, Timelytics provides a robust and reliable prediction of OTD times, helping businesses to optimize their supply chain operations."
)

st.caption(
    "With Timelytics, businesses can identify potential bottlenecks and delays in their supply chain and take proactive measures to address them, reducing lead times and improving delivery times. The model utilizes historical data on order processing times, production lead times, shipping times, and other relevant variables to generate accurate forecasts of OTD times. These forecasts can be used to optimize inventory management, improve customer service, and increase overall efficiency in the supply chain."
)

# Caching the model for faster loading
@st.cache_resource
def load_model():
    # Load the trained ensemble model from the saved pickle file.
    modelfile = "./voting_model.pkl"
    
    # Check if file exists
    if not os.path.exists(modelfile):
        st.error(f"❌ Model file '{modelfile}' not found!")
        st.info("Please ensure the voting_model.pkl file is uploaded to your repository.")
        return None
    
    # Check file size
    file_size = os.path.getsize(modelfile)
    st.info(f"📁 Model file size: {file_size} bytes")
    
    # If file is too small, it's likely a Git LFS pointer or corrupted
    if file_size < 1000:  # Less than 1KB is suspicious for a ML model
        st.warning(f"⚠️ Model file seems too small ({file_size} bytes). This might be a Git LFS pointer file or corrupted file.")
        
        # Try to read the first few lines to check if it's a text file
        try:
            with open(modelfile, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith('version') or 'git-lfs' in first_line.lower():
                    st.error("🔍 This appears to be a Git LFS pointer file, not the actual pickle file!")
                    st.info("**Solution:** Upload the actual .pkl file directly to GitHub without using Git LFS.")
                    return None
        except:
            pass  # It's binary, continue with pickle loading
    
    try:
        with open(modelfile, 'rb') as file:
            model = pickle.load(file)
        st.success("✅ Model loaded successfully!")
        st.info(f"📊 Model type: {type(model).__name__}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        
        # Provide specific help based on error type
        if "invalid load key" in str(e):
            st.info("**This error usually means:**")
            st.info("1. The pickle file is corrupted")
            st.info("2. You uploaded a Git LFS pointer instead of the actual file")
            st.info("3. The file was not uploaded in binary mode")
            st.info("**Solution:** Re-upload the original .pkl file directly to GitHub")
        
        return None

# Load CSV data if available
@st.cache_data
def load_csv_data():
    """Load CSV data from data folder if available"""
    csv_files = []
    data_folder = "./data"
    
    if os.path.exists(data_folder):
        for file in os.listdir(data_folder):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(data_folder, file))
    
    return csv_files

# Load the model
voting_model = load_model()

# Load available CSV files
csv_files = load_csv_data()
if csv_files:
    st.info(f"📊 Found {len(csv_files)} CSV file(s) in data folder: {[os.path.basename(f) for f in csv_files]}")

# Only proceed if model is loaded successfully
if voting_model is not None:
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
        # Load sidebar image if available
        sidebar_image = load_image("./assets/logistics.jpg") or load_image("./assets/supply_chain.png")
        if sidebar_image:
            st.image(sidebar_image, use_column_width=True)
        
        st.markdown("### 📦 Supply Chain Optimization")
        st.header("Input Parameters")
        
        # Input parameters with better descriptions
        purchase_dow = st.selectbox(
            "Purchased Day of the Week",
            options=[0, 1, 2, 3, 4, 5, 6],
            index=3,
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x]
        )
        
        purchase_month = st.selectbox(
            "Purchased Month",
            options=list(range(1, 13)),
            index=0,
            format_func=lambda x: ["January", "February", "March", "April", "May", "June",
                                 "July", "August", "September", "October", "November", "December"][x-1]
        )
        
        year = st.number_input("Purchased Year", min_value=2010, max_value=2030, value=2018)
        product_size_cm3 = st.number_input("Product Size (cm³)", min_value=0.0, value=9328.0)
        product_weight_g = st.number_input("Product Weight (grams)", min_value=0.0, value=1800.0)
        geolocation_state_customer = st.number_input(
            "Customer State Code", min_value=0, max_value=50, value=10
        )
        geolocation_state_seller = st.number_input(
            "Seller State Code", min_value=0, max_value=50, value=20
        )
        distance = st.number_input("Distance (km)", min_value=0.0, value=475.35)
        
        # Submit button
        submit = st.button("🔮 Predict Wait Time", type="primary", use_container_width=True)

    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Define the output container for the predicted wait time.
        st.header("🎯 Prediction Results")

        # When the submit button is clicked, call the wait time predictor function
        if submit:
            with st.spinner("🤖 Analyzing supply chain data..."):
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
                    st.success(f"📅 **Predicted Wait Time: {prediction} days**")
                    
                    # Add interpretation
                    if prediction <= 5:
                        st.info("🚀 **Fast delivery expected!**")
                    elif prediction <= 10:
                        st.info("📦 **Standard delivery timeframe**")
                    else:
                        st.warning("⏰ **Longer delivery time - consider optimizing**")
                else:
                    st.error("❌ Prediction failed. Please check your input values.")

    with col2:
        # Display model info
        st.header("🤖 Model Information")
        st.info("**Ensemble Model Components:**")
        st.write("• XGBoost Regressor")
        st.write("• Random Forest")
        st.write("• Support Vector Machine")
        st.write("• Voting Ensemble Strategy")

else:
    st.error("❌ **Cannot proceed without a valid model file.**")
    st.markdown("### 🔧 **Troubleshooting Steps:**")
    st.markdown("""
    1. **Check your repository:** Ensure `voting_model.pkl` exists in the root directory
    2. **File size:** The pickle file should be larger than 1KB (typically several MB)
    3. **Re-upload:** Delete and re-upload the original pickle file from your assignment
    4. **Avoid Git LFS:** Upload directly through GitHub's web interface
    5. **Binary format:** Ensure the file is uploaded in binary mode, not as text
    """)

# Load and display sample dataset
st.header("📊 Sample Dataset")

# Try to load actual data from CSV if available
if csv_files:
    try:
        # Load the first CSV file found
        df_actual = pd.read_csv(csv_files[0])
        st.write("**Actual dataset from your data folder:**")
        st.dataframe(df_actual.head(10), use_container_width=True)
        st.info(f"📈 Dataset shape: {df_actual.shape[0]} rows × {df_actual.shape[1]} columns")
    except Exception as e:
        st.warning(f"Could not load CSV data: {str(e)}")
        # Fall back to sample data
        show_sample_data = True
else:
    show_sample_data = True

# Show sample data if no CSV is available or if CSV loading failed
if 'show_sample_data' not in locals():
    show_sample_data = False

if show_sample_data or not csv_files:
    # Define a sample dataset for demonstration purposes.
    data = {
        "Purchased Day of the Week": ["Monday", "Thursday", "Tuesday"],
        "Purchased Month": ["June", "March", "January"],
        "Purchased Year": ["2018", "2017", "2018"],
        "Product Size in cm³": ["37206.0", "63714", "54816"],
        "Product Weight in grams": ["16250.0", "7249", "9600"],
        "Geolocation State Customer": ["25", "25", "25"],
        "Geolocation State Seller": ["20", "7", "20"],
        "Distance (km)": ["247.94", "250.35", "4.915"],
        "Expected Wait Time": ["7 days", "12 days", "3 days"]
    }

    # Create a DataFrame from the sample dataset.
    df = pd.DataFrame(data)

    # Display the sample dataset in the Streamlit app.
    st.write("**Sample dataset for reference:**")
    st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Timelytics** - Powered by Machine Learning 🤖 | Built with Streamlit 🚀")
