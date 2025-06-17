from ultralytics import YOLO

# Load the custom model
model = YOLO("best12_20.pt")

# Run prediction on the image and display/save the results
model.predict(source="mahi.jpg", show=True, save=True, conf=0.05, line_width=3 )
