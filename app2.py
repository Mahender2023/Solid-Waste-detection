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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- UI Customization ---
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .st-emotion-cache-1y4p8pa { max-width: 100%; }
    .st-emotion-cache-1v0mbdj { display: none; }
    .stMetric {
        border: 1px solid #ccc;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px;
	}
	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: #F0F2F6;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
    }
	.stTabs [aria-selected="true"] {
  		background-color: #FFFFFF;
	}
</style>
""", unsafe_allow_html=True)


# --- Model and Metrics Paths ---
# IMPORTANT: Make sure your model filenames match these
MODEL_PATH_V11 = "best11_20.pt"
MODEL_PATH_V12 = "best12_20.pt"

# IMPORTANT: Make sure your metrics folders match these
METRICS_DIR_V11 = f"metrics/yolov11"
METRICS_DIR_V12 = f"metrics/yolov12"

# Corrected filenames for metrics for each model
METRIC_FILES_V11 = {
    "Learning Curves": "learning_curve1.png",
    "Confusion Matrix": "confustion_matrixyolov11.png",
    "Precision-Recall Curve": "PR_curveyolov11.png",
    "F1-Confidence Curve": "F1_curveyolov11.png",
    "Precision-Confidence Curve": "P_curveyolov11.png",
    "Recall-Confidence Curve": "R_curveyolov11.png",
}

METRIC_FILES_V12 = {
    "Learning Curves": "learning_curve2.png",
    "Confusion Matrix": "confustion_matrixyolov12.png",
    "Precision-Recall Curve": "PR_curveyolo12.png",
    "F1-Confidence Curve": "F1_curveyolov12.png",
    "Precision-Confidence Curve": "P_curveyolov12.png",
    "Recall-Confidence Curve": "R_curveyolov12.png",
}

# --- Caching and Loading Functions ---
@st.cache_resource
def load_yolo_model(model_path):
    """Loads a YOLO model from the specified path."""
    try:
        return YOLO(model_path)
    except Exception as e:
        return e

# --- Main Application Logic ---
def main():
    st.title("YOLO Model Comparison for Solid Waste Detection ♻️")
    st.markdown("Upload an image to compare model detections, then review their training performance metrics below.")

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        st.subheader("Model Status")
        if os.path.exists(MODEL_PATH_V11):
            st.success(f"Model 1 Found: '{MODEL_PATH_V11}'")
        else:
            st.error(f"Model 1 not found at '{MODEL_PATH_V11}'")
        
        if os.path.exists(MODEL_PATH_V12):
            st.success(f"Model 2 Found: '{MODEL_PATH_V12}'")
        else:
            st.error(f"Model 2 not found at '{MODEL_PATH_V12}'")
        
        st.markdown("---")
        st.subheader("Metrics Status")
        if os.path.isdir(METRICS_DIR_V11):
            st.success(f"Metrics for Model 1 Found")
        else:
            st.warning(f"Metrics folder not found for Model 1: '{METRICS_DIR_V11}'")

        if os.path.isdir(METRICS_DIR_V12):
            st.success(f"Metrics for Model 2 Found")
        else:
            st.warning(f"Metrics folder not found for Model 2: '{METRICS_DIR_V12}'")

        st.markdown("---")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

    # --- Image Uploader and Processing ---
    uploaded_file = st.file_uploader("Choose an image of waste...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        model_v11 = load_yolo_model(MODEL_PATH_V11)
        model_v12 = load_yolo_model(MODEL_PATH_V12)

        # Check for loading errors
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
        
        annotated_image_1 = cv2.cvtColor(results_v11.plot(), cv2.COLOR_BGR2RGB)
        annotated_image_2 = cv2.cvtColor(results_v12.plot(), cv2.COLOR_BGR2RGB)

        # --- Display Detection Results Side-by-Side ---
        col1, col2 = st.columns(2)
        with col1:
            display_detection_results(results_v11, annotated_image_1, MODEL_PATH_V11)
        with col2:
            display_detection_results(results_v12, annotated_image_2, MODEL_PATH_V12)
        
        # --- NEW: Display Metrics Section ---
        st.markdown("---")
        st.header("\U0001F4CA Model Performance Metrics")
        st.info("These are static images generated during model training, showing overall performance.")

        # Create tabs for each metric
        tab_titles = list(METRIC_FILES_V11.keys())
        tabs = st.tabs(tab_titles)

        for i, tab_title in enumerate(tab_titles):
            with tabs[i]:
                metric_col1, metric_col2 = st.columns(2)

                with metric_col1:
                    st.subheader(os.path.basename(MODEL_PATH_V11))
                    metric_path = os.path.join(METRICS_DIR_V11, METRIC_FILES_V11[tab_title])
                    if os.path.exists(metric_path):
                        st.image(metric_path, use_container_width=True)
                    else:
                        st.warning(f"Metric image not found: `{metric_path}`")
                
                with metric_col2:
                    st.subheader(os.path.basename(MODEL_PATH_V12))
                    metric_path = os.path.join(METRICS_DIR_V12, METRIC_FILES_V12[tab_title])
                    if os.path.exists(metric_path):
                        st.image(metric_path, use_container_width=True)
                    else:
                        st.warning(f"Metric image not found: `{metric_path}`")

    else:
        st.info("👋 Welcome! Please upload an image file to get started.")

def display_detection_results(results, annotated_image, model_path):
    """Helper function to display detection results for a single model."""
    st.header(f"Results: {os.path.basename(model_path)}")
    st.image(annotated_image, caption=f"Detected by {os.path.basename(model_path)}", use_container_width=True)
    
    st.subheader("Detection Summary")
    total_count = len(results.boxes)
    st.metric(label="Total Objects Detected", value=total_count)
    
    if total_count > 0:
        st.subheader("Detailed Counts")
        class_names = [results.names[int(cls)] for cls in results.boxes.cls]
        detection_counts = Counter(class_names)
        df = pd.DataFrame(list(detection_counts.items()), columns=['Object Type', 'Count'])
        st.table(df)
    else:
        st.warning("No objects detected by this model.")


if __name__ == "__main__":
    main()
    # To run this app, use the command:
    # streamlit run your_app_name.py --server.fileWatcherType none