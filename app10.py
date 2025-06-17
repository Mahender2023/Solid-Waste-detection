import streamlit as st
import cv2
from ultralytics import YOLO
import cvzone
from collections import Counter
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Live YOLO Object Detection",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("YOLO Live Detection: Model Comparison 🚀")
st.markdown(
    "This application performs live object detection from your chosen camera source. "
    "Configure your models and settings in the sidebar."
)

# --- Model Loading ---
@st.cache_resource
def load_yolo_model(model_path):
    """Loads a YOLO model from the specified path."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_path}: {e}")
        return None

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")

    # NEW: Simplified Video Source Selection
    source_option = st.radio(
        "Select Video Source",
        ("Laptop Webcam", "Phone (as Virtual Webcam)")
    )

    # Let user specify the camera ID
    if source_option == "Laptop Webcam":
        camera_id = st.number_input("Laptop Camera ID", value=0, min_value=0, max_value=5, step=1)
    else: # Phone (as Virtual Webcam)
        camera_id = st.number_input("Phone Camera ID (try 1 or 2)", value=1, min_value=0, max_value=5, step=1)
        st.info("Ensure the DroidCam PC client is running and connected to your phone.")

    video_source = camera_id

    # Model paths and confidence
    model_1_path = st.text_input("Path to Model 1", "best11_20.pt")
    model_2_path = st.text_input("Path to Model 2", "best12_20.pt")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
    
    # Load models
    model1 = load_yolo_model(model_1_path)
    model2 = load_yolo_model(model_2_path)

# --- Main Application Logic ---
run_detection = st.toggle(f'Start Detection from {source_option}')

col1, col2 = st.columns(2)
with col1:
    st.header(f"Model 1: `{model_1_path}`")
    frame_placeholder1 = st.empty()
    summary_placeholder1 = st.empty()
with col2:
    st.header(f"Model 2: `{model_2_path}`")
    frame_placeholder2 = st.empty()
    summary_placeholder2 = st.empty()


if run_detection:
    if model1 is None or model2 is None:
        st.warning("Please provide valid model paths in the sidebar to start detection.")
    else:
        # Use the selected video source ID
        cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW) # CAP_DSHOW can improve compatibility on Windows
        
        if not cap.isOpened():
            st.error(f"Error: Could not open camera with ID '{video_source}'. Try another ID or check connections.")
        else:
            while run_detection:
                success, frame = cap.read()
                if not success:
                    st.error("Failed to capture frame. Camera may be disconnected. Please restart.", icon="⚠️")
                    break

                # The rest of the processing loop is identical
                frame1 = frame.copy()
                frame2 = frame.copy()

                # --- Process Frame with Model 1 ---
                results1 = model1(frame1, stream=True, conf=confidence_threshold, verbose=False)
                object_counts1 = Counter()
                for r in results1:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cvzone.putTextRect(frame1, f"{model1.names.get(int(box.cls[0]))} {box.conf[0]:.2f}", (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(255, 0, 255))
                        object_counts1[model1.names.get(int(box.cls[0]))] += 1
                
                # --- Process Frame with Model 2 ---
                results2 = model2(frame2, stream=True, conf=confidence_threshold, verbose=False)
                object_counts2 = Counter()
                for r in results2:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cvzone.putTextRect(frame2, f"{model2.names.get(int(box.cls[0]))} {box.conf[0]:.2f}", (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 255, 0))
                        object_counts2[model2.names.get(int(box.cls[0]))] += 1
                
                # --- Update UI ---
                frame_placeholder1.image(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), channels="RGB")
                frame_placeholder2.image(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), channels="RGB")
                
                df1 = pd.DataFrame(list(object_counts1.items()), columns=['Object', 'Count']).sort_values(by='Count', ascending=False)
                df2 = pd.DataFrame(list(object_counts2.items()), columns=['Object', 'Count']).sort_values(by='Count', ascending=False)
                
                summary_placeholder1.dataframe(df1, use_container_width=True)
                summary_placeholder2.dataframe(df2, use_container_width=True)

            cap.release()
else:
    st.info("Configure your settings in the sidebar and click 'Start Detection'.")