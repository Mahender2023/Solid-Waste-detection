from ultralytics import YOLO
import cv2
import cvzone
import math

model = YOLO("best12_20.pt")  


cap = cv2.VideoCapture(1)   
cap.set(3, 640)  
cap.set(4, 480) 

if not cap.isOpened():
    raise IOError("Cannot open webcam") 

while True:
    success, img = cap.read()  

    if not success:
        print("Failed to capture frame. Exiting...")
        break

    results = model(img, stream=True)  

    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates in xyxy format
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers

            # Calculate width and height of the box
            w, h = x2 - x1, y2 - y1

            # Draw a rectangle around the detected object using cvzone (optional)
            cvzone.cornerRect(img, (x1, y1, w, h))


            # Get the confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100  # Confidence score (0 to 1)
            cls = int(box.cls[0])

            # Get class name (assuming you have a 'names' dictionary in your model)
            class_name = model.names[cls] if hasattr(model, 'names') else str(cls)

            # Display confidence and class name on the image using cvzone
            cvzone.putTextRect(img, f"{class_name} {conf:.2f}", (max(0, x1), max(35, y1)),
                               scale=1, thickness=3) #display class name

    # Display the resulting image with detections
    cv2.imshow("Webcam Feed", img)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()