
from pathlib import Path
from urllib.request import Request, urlopen
from ultralytics import YOLO
import cv2

test_dir = Path("test_images")
output_dir = Path("outputs/multiple_tests")

test_dir.mkdir(exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

indian_image_url = (
    "https://images.pexels.com/photos/35755240/"
    "pexels-photo-35755240.jpeg?auto=compress&cs=tinysrgb&w=1280"
)

indian_image_path = test_dir / "indian_traffic.jpg"

# Download Indian image if missing
if not indian_image_path.exists():
    try:
        print("Downloading Indian traffic image...")

        request = Request(
            indian_image_url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urlopen(request, timeout=30) as response:
            indian_image_path.write_bytes(response.read())

        print("Indian traffic image downloaded!")

    except Exception as error:
        print(f"Download failed: {error}")

        if indian_image_path.exists():
            indian_image_path.unlink()

# Collect images ONLY from test_images
image_paths = sorted(
    path for path in test_dir.iterdir()
    if path.is_file()
    and path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    and path.stat().st_size > 1000
)

print("\nImages to test:")
for path in image_paths:
    print(f" - {path.name}")

print("\nLoading YOLO26n model...")
model = YOLO("yolo26n.pt")

for image_path in image_paths:

    print(f"\n{'=' * 45}")
    print(f"IMAGE: {image_path.name}")
    print("=" * 45)

    results = model.predict(
        source=str(image_path),
        device="cpu",
        imgsz=416,
        conf=0.25,
        save=False,
        verbose=False
    )

    result = results[0]
    annotated = result.plot()

    save_path = output_dir / f"detected_{image_path.stem}.jpg"
    cv2.imwrite(str(save_path), annotated)

    counts = {}

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = result.names[class_id]

            counts.setdefault(label, []).append(confidence)

    if counts:
        total = sum(len(scores) for scores in counts.values())
        print(f"Total objects: {total}")

        for label, scores in counts.items():
            print(f"\n{label}: {len(scores)}")

            for score in scores:
                print(f"   Confidence: {score:.2%}")
    else:
        print("No objects detected.")

    print(f"\nSaved: {save_path}")

print("\nAll image tests completed!")
