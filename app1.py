import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import os
from collections import Counter
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Solid Waste Detection | Model Comparison",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- UI Customization ---
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-1y4p8pa { /* Main content area */
        max-width: 100%;
    }
    .st-emotion-cache-1v0mbdj { /* Hamburger menu */
        display: none;
    }
    .stMetric {
        border: 1px solid #ccc;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO model from the specified path.
    Caches the model to avoid reloading on every app interaction.
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        # We will display a more specific error in the main app
        return e

# --- Main Application Logic ---
def main():
    st.title("YOLO Model Comparison for Solid Waste Detection ♻️")
    st.markdown("Upload an image to compare the performance of **YOLOv11** and **YOLOv12** side-by-side.")

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Define model paths - CHANGE THESE TO YOUR ACTUAL FILENAMES
        model_path_v11 = "best11_20.pt"
        model_path_v12 = "best12_20.pt"
        
        # Check for model files and display status
        st.markdown("---")
        st.subheader("Model Status")
        if os.path.exists(model_path_v11):
            st.success(f"Model 1 Found: '{model_path_v11}'")
        else:
            st.error(f"Model 1 not found at '{model_path_v11}'")
        
        if os.path.exists(model_path_v12):
            st.success(f"Model 2 Found: '{model_path_v12}'")
        else:
            st.error(f"Model 2 not found at '{model_path_v12}'")
        st.markdown("---")

        # Confidence Threshold Slider
        confidence_threshold = st.slider(
            "Confidence Threshold", 0.0, 1.0, 0.25, 0.05
        )
        st.markdown("---")
        st.info(
            "**About this project:**\n"
            "This app visually compares two custom-trained YOLO models for detecting solid waste in diverse environments."
        )

    # --- Image Uploader ---
    uploaded_file = st.file_uploader(
        "Choose an image of waste...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Load models
        with st.spinner("Loading models... This may take a moment."):
            model_v11 = load_model(model_path_v11)
            model_v12 = load_model(model_path_v12)

        # Check for loading errors
        if isinstance(model_v11, Exception) or isinstance(model_v12, Exception):
            if isinstance(model_v11, Exception):
                st.error(f"Failed to load {model_path_v11}: {model_v11}")
            if isinstance(model_v12, Exception):
                st.error(f"Failed to load {model_path_v12}: {model_v12}")
            return # Stop execution if models can't be loaded

        # Display the original image
        image = Image.open(uploaded_file)
        st.subheader("Original Uploaded Image")
        # FIX: Replaced use_column_width with use_container_width
        st.image(image, caption="Original Image", use_container_width=True)
        st.markdown("---")

        # Run predictions in parallel
        with st.spinner("Processing image with both models..."):
            results_v11 = model_v11.predict(source=image, conf=confidence_threshold, save=False)
            results_v12 = model_v12.predict(source=image, conf=confidence_threshold, save=False)
        
        result1 = results_v11[0]
        result2 = results_v12[0]

        # Get annotated images
        annotated_image_1 = cv2.cvtColor(result1.plot(), cv2.COLOR_BGR2RGB)
        annotated_image_2 = cv2.cvtColor(result2.plot(), cv2.COLOR_BGR2RGB)

        # Create side-by-side columns
        col1, col2 = st.columns(2)

        # --- Column for YOLOv11 ---
        with col1:
            st.header(f"Results: {os.path.basename(model_path_v11)}")
            # FIX: Replaced use_column_width with use_container_width
            st.image(annotated_image_1, caption=f"Detected by {os.path.basename(model_path_v11)}", use_container_width=True)
            
            st.subheader("Detection Summary")
            total_count_1 = len(result1.boxes)
            st.metric(label="Total Objects Detected", value=total_count_1)
            
            if total_count_1 > 0:
                st.subheader("Detailed Counts")
                class_names_1 = [model_v11.names[int(cls)] for cls in result1.boxes.cls]
                detection_counts_1 = Counter(class_names_1)
                
                df_1 = pd.DataFrame(list(detection_counts_1.items()), columns=['Object Type', 'Count'])
                st.table(df_1)
            else:
                st.warning("No objects detected by this model.")

        # --- Column for YOLOv12 ---
        with col2:
            st.header(f"Results: {os.path.basename(model_path_v12)}")
            # FIX: Replaced use_column_width with use_container_width
            st.image(annotated_image_2, caption=f"Detected by {os.path.basename(model_path_v12)}", use_container_width=True)

            st.subheader("Detection Summary")
            total_count_2 = len(result2.boxes)
            st.metric(label="Total Objects Detected", value=total_count_2)

            if total_count_2 > 0:
                st.subheader("Detailed Counts")
                class_names_2 = [model_v12.names[int(cls)] for cls in result2.boxes.cls]
                detection_counts_2 = Counter(class_names_2)

                df_2 = pd.DataFrame(list(detection_counts_2.items()), columns=['Object Type', 'Count'])
                st.table(df_2)
            else:
                st.warning("No objects detected by this model.")

    else:
        st.info("👋 Welcome! Please upload an image file to get started.")

if __name__ == "__main__":
    main()