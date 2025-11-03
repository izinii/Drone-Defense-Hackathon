from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

# Perform object detection on an image
results = model("image_test.jpg")
print("\n\nAAAAAAAAAAAAA")
print(results[0])

print("\n\nBBBBBBBBBBBBB")
results[0].show()