import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import os
from collections import Counter
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Solid Waste Detection | Model Comparison",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- UI Customization ---
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .st-emotion-cache-1y4p8pa { max-width: 100%; }
    /* Hide hamburger menu */
    .st-emotion-cache-1v0mbdj { display: none; } 
    .stMetric {
        border: 1px solid #ccc;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
	.stTabs [data-baseweb="tab"] {
		height: 50px; white-space: pre-wrap; background-color: #F0F2F6;
		border-radius: 4px 4px 0px 0px; gap: 1px; padding: 10px;
    }
	.stTabs [aria-selected="true"] { background-color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)


# --- File Paths Configuration ---
MODEL_PATH_V11 = "best11_20.pt"
MODEL_PATH_V12 = "best12_20.pt"
METRICS_DIR_V11 = "metrics/yolov11"
METRICS_DIR_V12 = "metrics/yolov12"
ARCHITECTURE_IMAGE_PATH = "architecture.png" # Your architecture image

METRIC_FILES_V11 = {
    "Learning Curves": "learning_curve1.png", "Confusion Matrix": "confustion_matrixyolov11.png",
    "Precision-Recall Curve": "PR_curveyolov11.png", "F1-Confidence Curve": "F1_curveyolov11.png",
    "Precision-Confidence Curve": "P_curveyolov11.png", "Recall-Confidence Curve": "R_curveyolov11.png",
}

METRIC_FILES_V12 = {
    "Learning Curves": "learning_curve2.png", "Confusion Matrix": "confustion_matrixyolov12.png",
    "Precision-Recall Curve": "PR_curveyolo12.png", "F1-Confidence Curve": "F1_curveyolov12.png",
    "Precision-Confidence Curve": "P_curveyolov12.png", "Recall-Confidence Curve": "R_curveyolov12.png",
}

# --- Caching and Loading ---
@st.cache_resource
def load_yolo_model(model_path):
    try: return YOLO(model_path)
    except Exception as e: return e

# --- Page Rendering Functions ---

def render_home_page():
    st.title("Solid Waste Detection Model Analysis 🛰️")
    st.markdown("Welcome to the YOLO model comparison dashboard. This tool allows you to analyze and compare two custom-trained models for solid waste detection.")
    st.markdown("---")
    st.subheader("How to use this tool:")
    st.markdown("""
    - **🧪 Test Models:** Upload an image to see a side-by-side detection comparison in real-time.
    - **📊 Model Performance Metrics:** View the training performance graphs (like P-R curves and confusion matrices) for each model.
    - **🏗️ Architecture:** See the underlying neural network architecture used for these models.
    
    Use the navigation menu on the left to switch between sections.
    """)
    st.markdown("---")
    st.subheader("Asset Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("Model Files")
        if os.path.exists(MODEL_PATH_V11): st.success(f"'{MODEL_PATH_V11}' found.")
        else: st.error(f"'{MODEL_PATH_V11}' not found.")
        if os.path.exists(MODEL_PATH_V12): st.success(f"'{MODEL_PATH_V12}' found.")
        else: st.error(f"'{MODEL_PATH_V12}' not found.")
    with col2:
        st.info("Metric Folders")
        if os.path.isdir(METRICS_DIR_V11): st.success(f"Metrics for Model 1 found.")
        else: st.warning(f"'{METRICS_DIR_V11}' not found.")
        if os.path.isdir(METRICS_DIR_V12): st.success(f"Metrics for Model 2 found.")
        else: st.warning(f"'{METRICS_DIR_V12}' not found.")
    with col3:
        st.info("Architecture Image")
        if os.path.exists(ARCHITECTURE_IMAGE_PATH): st.success(f"'{ARCHITECTURE_IMAGE_PATH}' found.")
        else: st.error(f"'{ARCHITECTURE_IMAGE_PATH}' not found.")

def render_test_page():
    st.header("🧪 Test Models")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05, key="test_confidence")
    uploaded_file = st.file_uploader("Choose an image of waste...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        model_v11 = load_yolo_model(MODEL_PATH_V11)
        model_v12 = load_yolo_model(MODEL_PATH_V12)
        if isinstance(model_v11, Exception) or isinstance(model_v12, Exception):
            if isinstance(model_v11, Exception): st.error(f"Failed to load {MODEL_PATH_V11}: {model_v11}")
            if isinstance(model_v12, Exception): st.error(f"Failed to load {MODEL_PATH_V12}: {model_v12}")
            return

        image = Image.open(uploaded_file)
        st.subheader("Original Uploaded Image")
        st.image(image, caption="Original Image", use_container_width=True)
        st.markdown("---")

        with st.spinner("Processing image with both models..."):
            results_v11 = model_v11.predict(source=image, conf=confidence_threshold, save=False)[0]
            results_v12 = model_v12.predict(source=image, conf=confidence_threshold, save=False)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            display_detection_results(results_v11, MODEL_PATH_V11)
        with col2:
            display_detection_results(results_v12, MODEL_PATH_V12)

def render_metrics_page():
    st.header("\U0001F4CA Model Performance Metrics")
    st.info("These are static images generated during model training, showing overall performance.")
    
    tab_titles = list(METRIC_FILES_V11.keys())
    tabs = st.tabs(tab_titles)

    for i, tab_title in enumerate(tab_titles):
        with tabs[i]:
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.subheader(os.path.basename(MODEL_PATH_V11))
                metric_path = os.path.join(METRICS_DIR_V11, METRIC_FILES_V11[tab_title])
                if os.path.exists(metric_path): st.image(metric_path, use_container_width=True)
                else: st.warning(f"Metric image not found: `{metric_path}`")
            with metric_col2:
                st.subheader(os.path.basename(MODEL_PATH_V12))
                metric_path = os.path.join(METRICS_DIR_V12, METRIC_FILES_V12[tab_title])
                if os.path.exists(metric_path): st.image(metric_path, use_container_width=True)
                else: st.warning(f"Metric image not found: `{metric_path}`")

def render_architecture_page():
    st.header("🏗️ Model Architecture")
    st.info("This diagram illustrates the neural network architecture used for the object detection models.")
    if os.path.exists(ARCHITECTURE_IMAGE_PATH):
        st.image(ARCHITECTURE_IMAGE_PATH, caption="Model Architecture Diagram", use_container_width=True)
    else:
        st.error(f"Architecture image not found! Please place a file named '{ARCHITECTURE_IMAGE_PATH}' in the root directory.")

def display_detection_results(results, model_path):
    """Helper function to display detection results for a single model."""
    annotated_image = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)
    st.subheader(f"Results: {os.path.basename(model_path)}")
    st.image(annotated_image, caption=f"Detected by {os.path.basename(model_path)}", use_container_width=True)
    
    total_count = len(results.boxes)
    st.metric(label="Total Objects Detected", value=total_count)
    
    if total_count > 0:
        st.subheader("Detailed Counts")
        class_names = [results.names[int(cls)] for cls in results.boxes.cls]
        df = pd.DataFrame(list(Counter(class_names).items()), columns=['Object Type', 'Count'])
        st.table(df)
    else:
        st.warning("No objects detected by this model.")

# --- Main App Router ---
def main():
    with st.sidebar:
        st.title("🛰️ Navigation")
        page = st.radio(
            "Go to",
            ("🏠 Home", "🧪 Test Models", "📊 Model Performance Metrics", "🏗️ Architecture")
        )
    
    if page == "🏠 Home":
        render_home_page()
    elif page == "🧪 Test Models":
        render_test_page()
    elif page == "📊 Model Performance Metrics":
        render_metrics_page()
    elif page == "🏗️ Architecture":
        render_architecture_page()

if __name__ == "__main__":
    main()
    # To run this app, use the command:
    # streamlit run your_app_name.py --server.fileWatcherType none