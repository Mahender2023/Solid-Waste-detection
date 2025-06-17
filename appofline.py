import os
import argparse
from collections import Counter

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# --- Configuration ---
# Define the paths to your two trained models
MODEL_PATH_V11 = "best11_20.pt"
MODEL_PATH_V12 = "best12_20.pt"
CONFIDENCE_THRESHOLD = 0.25

# --- Metrics Paths ---
METRICS_DIR_V11 = "metrics/yolov11"
METRICS_DIR_V12 = "metrics/yolov12"

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

def run_comparison(image_path):
    """
    Runs both YOLO models on a single image and displays the results side-by-side.
    """
    # 1. --- Validate Inputs ---
    if not os.path.exists(image_path):
        print(f"Error: Input image not found at '{image_path}'")
        return

    if not os.path.exists(MODEL_PATH_V11) or not os.path.exists(MODEL_PATH_V12):
        print("Error: One or both model files are missing.")
        print(f"Checked for '{MODEL_PATH_V11}' and '{MODEL_PATH_V12}'")
        return

    # 2. --- Load Models ---
    print("Loading models... This might take a moment.")
    try:
        model_v11 = YOLO(MODEL_PATH_V11)
        model_v12 = YOLO(MODEL_PATH_V12)
    except Exception as e:
        print(f"An error occurred while loading models: {e}")
        return
    print("Models loaded successfully.")

    # 3. --- Run Predictions ---
    print(f"Processing image: '{image_path}'")
    try:
        # Open image with PIL
        source_image = Image.open(image_path)
        
        # Predict with both models
        results_v11 = model_v11.predict(source=source_image, conf=CONFIDENCE_THRESHOLD, save=False)[0]
        results_v12 = model_v12.predict(source=source_image, conf=CONFIDENCE_THRESHOLD, save=False)[0]
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return

    # 4. --- Process and Print Results ---
    print("\n" + "="*40)
    print("DETECTION RESULTS")
    print("="*40)

    # Process Model 1
    total_count_1 = len(results_v11.boxes)
    class_names_1 = [model_v11.names[int(cls)] for cls in results_v11.boxes.cls]
    detection_counts_1 = Counter(class_names_1)
    
    print(f"\n--- Model: {os.path.basename(MODEL_PATH_V11)} ---")
    print(f"Total Detections: {total_count_1}")
    if total_count_1 > 0:
        print("Detailed Counts:")
        for obj, count in detection_counts_1.items():
            print(f"  - {obj}: {count}")

    # Process Model 2
    total_count_2 = len(results_v12.boxes)
    class_names_2 = [model_v12.names[int(cls)] for cls in results_v12.boxes.cls]
    detection_counts_2 = Counter(class_names_2)

    print(f"\n--- Model: {os.path.basename(MODEL_PATH_V12)} ---")
    print(f"Total Detections: {total_count_2}")
    if total_count_2 > 0:
        print("Detailed Counts:")
        for obj, count in detection_counts_2.items():
            print(f"  - {obj}: {count}")
    
    print("\n" + "="*40)
    print("Displaying annotated images. Close the image window to exit.")

    # 5. --- Display Images Side-by-Side ---
    # Get annotated images from YOLO results (these are in BGR format)
    annotated_image_1_bgr = results_v11.plot()
    annotated_image_2_bgr = results_v12.plot()

    # Convert BGR to RGB for Matplotlib display
    annotated_image_1_rgb = cv2.cvtColor(annotated_image_1_bgr, cv2.COLOR_BGR2RGB)
    annotated_image_2_rgb = cv2.cvtColor(annotated_image_2_bgr, cv2.COLOR_BGR2RGB)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('YOLO Model Comparison', fontsize=20)

    # Display first image
    axes[0].imshow(annotated_image_1_rgb)
    axes[0].set_title(f"{os.path.basename(MODEL_PATH_V11)} Results ({total_count_1} detections)", fontsize=14)
    axes[0].axis('off')  # Hide the axes

    # Display second image
    axes[1].imshow(annotated_image_2_rgb)
    axes[1].set_title(f"{os.path.basename(MODEL_PATH_V12)} Results ({total_count_2} detections)", fontsize=14)
    axes[1].axis('off')  # Hide the axes

    plt.tight_layout()  # Adjust layout to prevent titles overlapping
    plt.show()

    # 6. --- Display Metrics Images ---
    print("\nDisplaying training metrics for both models...")
    for metric_name in METRIC_FILES_V11.keys():
        print(f"\n=== {metric_name} ===")
        metric_path_v11 = os.path.join(METRICS_DIR_V11, METRIC_FILES_V11[metric_name])
        metric_path_v12 = os.path.join(METRICS_DIR_V12, METRIC_FILES_V12[metric_name])
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(metric_name, fontsize=16)
        # Model 1
        if os.path.exists(metric_path_v11):
            img1 = Image.open(metric_path_v11)
            axes[0].imshow(img1)
            axes[0].set_title(f"{os.path.basename(MODEL_PATH_V11)}")
        else:
            axes[0].text(0.5, 0.5, 'Not found', ha='center', va='center', fontsize=12)
            axes[0].set_title(f"{os.path.basename(MODEL_PATH_V11)}")
        axes[0].axis('off')
        # Model 2
        if os.path.exists(metric_path_v12):
            img2 = Image.open(metric_path_v12)
            axes[1].imshow(img2)
            axes[1].set_title(f"{os.path.basename(MODEL_PATH_V12)}")
        else:
            axes[1].text(0.5, 0.5, 'Not found', ha='center', va='center', fontsize=12)
            axes[1].set_title(f"{os.path.basename(MODEL_PATH_V12)}")
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Set up argument parser to accept an image path from the command line
    parser = argparse.ArgumentParser(description="Compare two YOLO models on a single image offline.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    
    args = parser.parse_args()
    
    run_comparison(args.image_path)