
from pathlib import Path
from ultralytics import YOLO
import urllib.request

# Project output folder
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Download a sample image for testing
image_path = Path("bus.jpg")
image_url = "https://ultralytics.com/images/bus.jpg"

if not image_path.exists():
    print("Downloading sample image...")
    urllib.request.urlretrieve(image_url, image_path)

print("Loading YOLO26n pretrained model...")
model = YOLO("yolo26n.pt")

print("Running object detection on CPU...")
results = model.predict(
    source=str(image_path),
    device="cpu",
    imgsz=416,
    conf=0.25,
    save=False,
    verbose=False
)

# Save the annotated image
result = results[0]
annotated_image = result.plot()

output_path = OUTPUT_DIR / "test_detection.jpg"

import cv2
cv2.imwrite(str(output_path), annotated_image)

# Display detected objects
print("\nDetection Results:")
if result.boxes is not None and len(result.boxes) > 0:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = result.names[class_id]
        print(f"{label}: {confidence:.2%}")
else:
    print("No objects detected.")

print(f"\nProcessed image saved to: {output_path}")
print("YOLO model test completed successfully!")
