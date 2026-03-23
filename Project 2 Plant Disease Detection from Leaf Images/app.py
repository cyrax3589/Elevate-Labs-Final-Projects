import streamlit as st
import numpy as np
import keras
from keras.preprocessing import image

# 1. Page config
st.set_page_config(page_title="LEAFIX ", page_icon="🌱", layout="wide")

# 2. CSS
st.markdown("""
<style>
    /* Hide Streamlit header, footer, and main menu */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Squeeze out empty space at the top and lock height */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-height: 100vh;
        overflow-y: hidden;
    }

    /* Restrict image height so it never causes scrolling */
    [data-testid="stImage"] img {
        max-height: 40vh;
        object-fit: contain;
        border-radius: 10px;
    }
    
    /* Make headers a bit more compact */
    h1 {
        margin-top: -10px;
        padding-bottom: 5px;
    }

    /* --- NEW: Fixed Footer CSS --- */
    .custom-footer {
        position: fixed;
        bottom: 15px;
        left: 0;
        width: 100%;
        text-align: center;
        color: #9ca3af;
        font-family: sans-serif;
        font-size: 0.9rem;
        z-index: 100;
    }
    
    .custom-footer a {
        color: #9ca3af;
        margin: 0 10px;
        transition: color 0.3s ease;
        text-decoration: none;
    }
    
    .custom-footer a:hover {
        color: #4ade80; /* Lights up green on hover */
    }
    
    .custom-footer svg {
        vertical-align: middle;
        margin-bottom: 3px;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return keras.models.load_model("best_model.keras")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Class names
class_names = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", 
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", 
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# Probable Treatment Dictionary
treatment_suggestions = {
    "Pepper__bell___Bacterial_spot": "Often managed by removing heavily infected plants and applying copper-based sprays. Avoid overhead watering.",
    "Pepper__bell___healthy": "The plant appears healthy. Continue regular watering and maintain good soil drainage.",
    "Potato___Early_blight": "Consider applying a protective fungicide. Practicing crop rotation and removing plant debris may help.",
    "Potato___Late_blight": "Typically aggressive; prompt application of appropriate fungicides is usually recommended.",
    "Potato___healthy": "The plant appears healthy. Keep monitoring for pests and maintain standard care.",
    "Tomato_Bacterial_spot": "Might benefit from copper fungicides. Watering at the base of the plant keeps leaves dry.",
    "Tomato_Early_blight": "Pruning lower leaves for better air circulation and applying a general-purpose fungicide are common steps.",
    "Tomato_Late_blight": "Often requires immediate fungicidal treatment. Ensuring the plants have plenty of airflow helps.",
    "Tomato_Leaf_Mold": "Generally managed by improving ventilation around the plants and avoiding wetting the foliage.",
    "Tomato_Septoria_leaf_spot": "Consider removing severely affected leaves and treating with a fungicidal spray to protect new growth.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Commonly treated with insecticidal soaps or horticultural oils.",
    "Tomato__Target_Spot": "Improving airflow by pruning and using targeted fungicides usually helps manage this issue.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "No direct cure. Management usually involves controlling whitefly populations and removing infected plants.",
    "Tomato__Tomato_mosaic_virus": "Cannot be cured. Remove and destroy infected plants, and thoroughly wash hands and garden tools.",
    "Tomato_healthy": "The plant appears healthy. Maintain standard fertilization and watering schedules."
}

# Header Section
st.markdown("<h2 style='text-align: center;'>🥬 Leaf Disease Detection</h2>", unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

# Prediction and Results Section
if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing..."):
        pred = model.predict(img_array)
        
    pred_index = np.argmax(pred)
    confidence = float(np.max(pred))
    raw_prediction = class_names[pred_index]
    display_prediction = raw_prediction.replace("_", " ").replace("  ", " ").strip()

    # 3 COLUMN LAYOUT
    col1, col2, col3 = st.columns([1.3, 1, 1.3], gap="medium")

    with col1:
        st.image(uploaded_file, use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("Diagnosis")
            
            if "healthy" in display_prediction.lower():
                st.success(f"**{display_prediction}**")
            else:
                st.error(f"**{display_prediction}**")

            st.markdown("---")
            
            st.subheader("Confidence")
            st.progress(confidence)
            st.write(f"**{confidence*100:.2f}%**")

    with col3:
        treatment = treatment_suggestions.get(raw_prediction, "No general suggestions available for this class.")
        
        green_box_html = f"""
        <div style="
            background-color: rgba(25, 135, 84, 0.1); 
            border: 1px solid rgba(25, 135, 84, 0.4); 
            border-radius: 10px; 
            padding: 20px; 
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div>
                <h4 style="color: #4ade80; margin-top: 0px; margin-bottom: 10px;">💡 Probable Treatment</h4>
                <p style="font-size: 0.95rem; line-height: 1.5; color: #e2e8f0;">{treatment}</p>
            </div>
            <div style="margin-top: 15px;">
                <hr style="border-color: rgba(25, 135, 84, 0.3); margin-bottom: 10px;">
                <p style="font-size: 0.75rem; color: #9ca3af; margin: 0;">
                    ⚠️ <i>Disclaimer: General recommendations only. Consult a local agronomist for exact treatments.</i>
                </p>
            </div>
        </div>
        """
        st.markdown(green_box_html, unsafe_allow_html=True)


# --- Footer ---
footer_html = """
<div class="custom-footer">
    <span>Developed by <b>Debabrata Kuiry</b></span> | 
    <a href="https://github.com/cyrax3589" target="_blank" rel="noopener noreferrer" title="GitHub">
        <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
    </a>
    <a href="https://www.linkedin.com/in/debabrata-kuiry/" target="_blank" rel="noopener noreferrer" title="LinkedIn">
        <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.225 0z"/></svg>
    </a>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)