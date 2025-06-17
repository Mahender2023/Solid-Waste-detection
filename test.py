from ultralytics import YOLO
import cv2

# Load the custom model
model = YOLO("best11_20.pt")

# Image path
image_path = "1.jpg"

# Run prediction on the image and save the results
results = model.predict(source=image_path, show=False, save=True, conf=0.05, line_width=5) #Set show to false

# Access the Results object to print to console.
for result in results:
    boxes = result.boxes  # Boxes object for bbox coordinates
    probs = result.probs  # Class probabilities for classification tasks
    names = model.names  #Class names

    for box in boxes:
        confidence = box.conf[0].item()  # Confidence score
        class_id = int(box.cls[0].item())  # Class ID
        class_name = names[class_id]  # Class name
        print(f"Detected: {class_name} (Confidence: {confidence:.2f})")