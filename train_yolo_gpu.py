import torch
from ultralytics import YOLO

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# Initialize the YOLOv11 model with pre-trained weights
model = YOLO("yolo11x.pt")

# Train the model on the custom dataset
results = model.train(
    data="data.yaml", 
    epochs=3, 
    batch=8,
    imgsz=640,
  
      )

