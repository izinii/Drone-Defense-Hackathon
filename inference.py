from ultralytics import YOLO

"""
# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)

# Run inference on an image
results = model("images/img2.jpg", show=True, save=True) # return a Result object
"""

# Display model information
#print("\nModel info:", model.info(), "\n\n")

model = YOLO("yolo11n_visdrone.pt")
results = model("images/", save=True)