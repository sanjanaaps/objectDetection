import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
from io import BytesIO
import numpy as np

st.title("YOLO Object Detection")
try:
    model = YOLO(r'D:\data science project AiSpry\train5-20241109T102935Z-001\train5\weights\best.pt')
except Exception as e:
    model = None
    st.error(f"Error loading YOLO model weights: {e}")

uploaded_file = st.file_uploader("Choose an image...")

#Load the image
if uploaded_file is not None:
   
    # conver to opencv compatible format
    image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Convert to YOLO compatible format
    uploaded_file.seek(0)
    image_bytes = BytesIO(uploaded_file.read())
    pil_image = Image.open(image_bytes)
    
    # Run inference
    results = model.predict(pil_image)

    # Draw bounding boxes, class labels, and confidence scores
    for result in results:
        for box in result.boxes:  # Each box represents a detected object
            # Extract bounding box coordinates, confidence, and class label
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            label = model.names[class_id]  # Get class label name
            # Draw bounding box on the image
            color = (105, 155, 155)  # Green color for bounding box (unreadable)
            color1 = (2, 5, 28) # black color 
            thickness = 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color1, thickness)
            # Put label and confidence score on top of the bounding box
            label_text = f"{label} {confidence:.2f}"
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 2)       
    
    # coloring the image
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
    # displaying the image
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption='Uploaded Image.')  
    with col2:
        st.image(image, channels="RGB", caption='Detected Objects')
