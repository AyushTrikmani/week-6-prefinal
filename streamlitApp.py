# Import the required libraries: Streamlit, NumPy, and Pillow (PIL).
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import os

# Set the page configuration of the app, including the page title, icon, and layout.
st.set_page_config(
    page_title="Timelytics - Supply Chain Optimizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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

# Display header image if available
header_image = load_image("./assets/supply_chain_optimisation.jpg") or load_image("./assets/header.jpg")
if header_image:
    st.image(header_image, use_column_width=True)

# Display the title and captions for the app.
st.markdown('<h1 class="main-header">📦 Timelytics: Optimize your supply chain with advanced forecasting techniques</h1>', unsafe_allow_html=True)

# Introduction section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 🚀 About Timelytics
    
    **Timelytics** is an ensemble model that utilizes three powerful machine learning algorithms:
    - **XGBoost** - Gradient boosting for high performance
    - **Random Forests** - Ensemble learning for stability  
    - **Support Vector Machines (SVM)** - Non-linear pattern recognition
    
    By combining the strengths of these algorithms, Timelytics provides robust and reliable predictions 
    of Order to Delivery (OTD) times, helping businesses optimize their supply chain operations.
    """)

with col2:
    st.markdown("""
    <div class="info-card">
    <h4>🎯 Key Benefits</h4>
    <ul>
    <li>Accurate delivery time forecasting</li>
    <li>Identify supply chain bottlenecks</li>
    <li>Optimize inventory management</li>
    <li>Improve customer satisfaction</li>
    <li>Reduce operational costs</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Model loading with comprehensive error handling
@st.cache_resource
def load_model():
    """Load the trained ensemble model from the saved pickle file."""
    modelfile = "./voting_model.pkl"
    
    # Check if file exists
    if not os.path.exists(modelfile):
        st.error(f"❌ Model file '{modelfile}' not found!")
        st.info("Please ensure the voting_model.pkl file is uploaded to your repository.")
        return None
    
    # Check file size
    file_size = os.path.getsize(modelfile)
    
    # If file is too small, it's likely a Git LFS pointer or corrupted
    if file_size < 1000:  # Less than 1KB is suspicious for a ML model
        st.warning(f"⚠️ Model file seems too small ({file_size} bytes). This might be a Git LFS pointer file.")
        
        # Try to read the first few lines to check if it's a text file
        try:
            with open(modelfile, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if 'git-lfs' in first_line.lower() or first_line.startswith('version'):
                    st.error("🔍 This is a Git LFS pointer file, not the actual pickle file!")
                    st.info("**Solution:** Upload the actual .pkl file directly to GitHub.")
                    return None
        except:
            pass  # It's binary, continue with pickle loading
    
    try:
        with open(modelfile, 'rb') as file:
            model = pickle.load(file)
        
        st.success("✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        
        if "invalid load key" in str(e):
            st.info("""
            **This error usually means:**
            - The pickle file is corrupted
            - You uploaded a Git LFS pointer instead of the actual file
            - The file was not uploaded in binary mode
            
            **Solution:** Re-upload the original .pkl file directly to GitHub
            """)
        
        return None

# Load the model
voting_model = load_model()

# Define the function for the wait time predictor
def waitime_predictor(purchase_dow, purchase_month, year, product_size_cm3, 
                     product_weight_g, geolocation_state_customer, 
                     geolocation_state_seller, distance):
    """
    Predict wait time using the ensemble model
    
    Args:
        All the input parameters for prediction
    
    Returns:
        Predicted wait time in days (rounded to nearest integer)
    """
    try:
        prediction = voting_model.predict(
            np.array([[
                purchase_dow, purchase_month, year, product_size_cm3,
                product_weight_g, geolocation_state_customer,
                geolocation_state_seller, distance
            ]])
        )
        return round(prediction[0])
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Only proceed if model is loaded successfully
if voting_model is not None:
    
    # Sidebar for input parameters
    with st.sidebar:
        # Load sidebar image if available
        sidebar_image = load_image("./assets/supply_chain_optimisation.jpg")
        if sidebar_image:
            st.image(sidebar_image, use_column_width=True)
        
        st.markdown("### 📊 Input Parameters")
        st.markdown("---")
        
        # Day of week selection
        purchase_dow = st.selectbox(
            "📅 Purchased Day of the Week",
            options=[0, 1, 2, 3, 4, 5, 6],
            index=3,
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                 "Friday", "Saturday", "Sunday"][x]
        )
        
        # Month selection
        purchase_month = st.selectbox(
            "📆 Purchased Month",
            options=list(range(1, 13)),
            index=0,
            format_func=lambda x: ["January", "February", "March", "April", "May", "June",
                                 "July", "August", "September", "October", "November", "December"][x-1]
        )
        
        # Year input
        year = st.number_input(
            "🗓️ Purchased Year", 
            min_value=2010, 
            max_value=2030, 
            value=2018,
            help="Year of purchase"
        )
        
        # Product specifications
        st.markdown("#### 📦 Product Specifications")
        product_size_cm3 = st.number_input(
            "📏 Product Size (cm³)", 
            min_value=0.0, 
            value=9328.0,
            help="Volume of the product in cubic centimeters"
        )
        
        product_weight_g = st.number_input(
            "⚖️ Product Weight (grams)", 
            min_value=0.0, 
            value=1800.0,
            help="Weight of the product in grams"
        )
        
        # Geographic information
        st.markdown("#### 🌍 Geographic Information")
        geolocation_state_customer = st.number_input(
            "🏠 Customer State Code", 
            min_value=0, 
            max_value=50, 
            value=10,
            help="Numerical code representing customer's state"
        )
        
        geolocation_state_seller = st.number_input(
            "🏪 Seller State Code", 
            min_value=0, 
            max_value=50, 
            value=20,
            help="Numerical code representing seller's state"
        )
        
        distance = st.number_input(
            "📍 Distance (km)", 
            min_value=0.0, 
            value=475.35,
            help="Distance between customer and seller in kilometers"
        )
        
        st.markdown("---")
        
        # Submit button
        submit = st.button(
            "🔮 Predict Delivery Time", 
            type="primary", 
            use_container_width=True,
            help="Click to get delivery time prediction"
        )

    # Main content area
    st.markdown("## 🎯 Prediction Results")
    
    # Create columns for better layout
    result_col1, result_col2 = st.columns([2, 1])
    
    with result_col1:
        if submit:
            with st.spinner("🤖 Analyzing supply chain data..."):
                prediction = waitime_predictor(
                    purchase_dow, purchase_month, year, product_size_cm3,
                    product_weight_g, geolocation_state_customer,
                    geolocation_state_seller, distance
                )
                
                if prediction is not None:
                    # Display prediction with styling
                    if prediction <= 5:
                        st.success(f"🚀 **Predicted Delivery Time: {prediction} days**")
                        st.info("🎉 **Excellent! Fast delivery expected!**")
                    elif prediction <= 10:
                        st.success(f"📦 **Predicted Delivery Time: {prediction} days**")
                        st.info("✅ **Good! Standard delivery timeframe**")
                    elif prediction <= 15:
                        st.warning(f"⏰ **Predicted Delivery Time: {prediction} days**")
                        st.info("🔄 **Moderate delivery time**")
                    else:
                        st.error(f"🐌 **Predicted Delivery Time: {prediction} days**")
                        st.warning("⚠️ **Extended delivery time - consider supply chain optimization**")
                    
                    # Additional insights
                    st.markdown("### 📈 Delivery Insights")
                    
                    insights = []
                    if distance > 1000:
                        insights.append("🌍 Long distance may contribute to extended delivery time")
                    if product_weight_g > 5000:
                        insights.append("📦 Heavy product may require special handling")
                    if product_size_cm3 > 50000:
                        insights.append("📏 Large product size may affect shipping options")
                    if geolocation_state_customer != geolocation_state_seller:
                        insights.append("🚚 Interstate delivery may add processing time")
                    
                    if insights:
                        for insight in insights:
                            st.write(f"• {insight}")
                    else:
                        st.write("• ✅ Optimal delivery conditions detected")
                        
                else:
                    st.error("❌ Prediction failed. Please check your input values and try again.")
        else:
            st.info("👆 Please set your parameters in the sidebar and click 'Predict Delivery Time' to get started!")
    
    with result_col2:
        # Model information card
        st.markdown("""
        <div class="info-card">
        <h4>🤖 Model Information</h4>
        <p><strong>Ensemble Components:</strong></p>
        <ul>
        <li>🌟 XGBoost Regressor</li>
        <li>🌲 Random Forest</li>
        <li>🎯 Support Vector Machine</li>
        <li>🗳️ Voting Ensemble Strategy</li>
        </ul>
        <p><strong>Prediction Accuracy:</strong> High</p>
        <p><strong>Model Type:</strong> Regression Ensemble</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Error handling when model fails to load
    st.error("❌ **Cannot proceed without a valid model file.**")
    st.markdown("### 🔧 **Troubleshooting Steps:**")
    
    with st.expander("📋 Click here for detailed troubleshooting"):
        st.markdown("""
        1. **Check your repository:** Ensure `voting_model.pkl` exists in the root directory
        2. **File size:** The pickle file should be larger than 1KB (typically several MB)
        3. **Re-upload:** Delete and re-upload the original pickle file from your assignment
        4. **Avoid Git LFS:** Upload directly through GitHub's web interface
        5. **Binary format:** Ensure the file is uploaded in binary mode, not as text
        6. **File integrity:** Make sure the file wasn't corrupted during upload
        
        **Common Error Solutions:**
        - `FileNotFoundError`: File not in correct location
        - `Invalid load key`: File is corrupted or is a Git LFS pointer
        - `Module not found`: Missing dependencies in requirements.txt
        """)

# Sample Dataset Section
st.markdown("---")
st.markdown("## 📊 Sample Dataset")

# Create sample data
sample_data = {
    "Day of Week": ["Monday", "Thursday", "Tuesday", "Friday", "Wednesday"],
    "Month": ["June", "March", "January", "August", "November"],
    "Year": [2018, 2017, 2018, 2019, 2017],
    "Product Size (cm³)": [37206.0, 63714.0, 54816.0, 28945.0, 41256.0],
    "Product Weight (g)": [16250.0, 7249.0, 9600.0, 3450.0, 12800.0],
    "Customer State": [25, 25, 25, 15, 10],
    "Seller State": [20, 7, 20, 18, 22],
    "Distance (km)": [247.94, 250.35, 4.915, 156.78, 892.45],
    "Expected Delivery": ["7-9 days", "10-12 days", "2-4 days", "5-7 days", "12-15 days"]
}

df_sample = pd.DataFrame(sample_data)

# Display sample dataset with styling
st.dataframe(
    df_sample, 
    use_container_width=True,
    hide_index=True
)

st.markdown("""
**💡 Tips for using this dataset:**
- Use the sample values as reference for your predictions
- Notice how distance and product specifications affect delivery times
- Different state combinations can impact shipping duration
- Weekend purchases might have different processing times
""")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>🚀 Timelytics</strong> - Powered by Machine Learning 🤖</p>
        <p>Built with ❤️ using Streamlit | Supply Chain Optimization Made Simple</p>
    </div>
    """, 
    unsafe_allow_html=True
)
