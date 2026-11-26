
from pathlib import Path
from uuid import uuid4

import cv2
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

print("Loading VisionTrack AI model...")
model = YOLO("yolo26n.pt")
print("Model loaded successfully!")


@app.route("/", methods=["GET", "POST"])
def index():
    detections = []
    result_image = None
    result_video = None
    total_objects = 0
    uploaded_filename = None
    error = None
    tracked_ids = set()

    if request.method == "POST":

        image_file = request.files.get("image")
        video_file = request.files.get("video")

        if image_file and image_file.filename:
            extension = Path(image_file.filename).suffix.lower()

            if extension not in IMAGE_EXTENSIONS:
                error = "Unsupported image format."

            else:
                unique_id = uuid4().hex
                input_path = UPLOAD_DIR / f"{unique_id}{extension}"
                output_path = RESULT_DIR / f"{unique_id}.jpg"

                image_file.save(input_path)

                try:
                    results = model.predict(
                        source=str(input_path),
                        device="cpu",
                        imgsz=416,
                        conf=0.25,
                        save=False,
                        verbose=False
                    )

                    result = results[0]
                    annotated = result.plot()
                    cv2.imwrite(str(output_path), annotated)

                    counts = {}

                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            label = result.names[class_id]

                            counts.setdefault(label, []).append(confidence)

                    for label, scores in counts.items():
                        detections.append({
                            "name": label,
                            "count": len(scores),
                            "confidence": round(
                                sum(scores) / len(scores) * 100, 2
                            )
                        })

                    total_objects = sum(item["count"] for item in detections)
                    result_image = f"/static/results/{unique_id}.jpg"
                    uploaded_filename = image_file.filename

                except Exception as exc:
                    error = f"Image detection failed: {exc}"

        elif video_file and video_file.filename:
            extension = Path(video_file.filename).suffix.lower()

            if extension not in VIDEO_EXTENSIONS:
                error = "Unsupported video format. Use MP4, AVI, MOV or MKV."

            else:
                unique_id = uuid4().hex
                input_path = UPLOAD_DIR / f"{unique_id}{extension}"
                output_path = RESULT_DIR / f"{unique_id}.mp4"

                video_file.save(input_path)

                writer = None
                counts = {}
                track_ids = set()
                frame_count = 0

                try:
                    cap = cv2.VideoCapture(str(input_path))

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not fps or fps <= 0:
                        fps = 25

                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    if width <= 0 or height <= 0:
                        raise ValueError("Could not read video dimensions.")

                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height)
                    )

                    if not writer.isOpened():
                        raise ValueError("Could not create output video.")

                    while cap.isOpened():
                        success, frame = cap.read()

                        if not success:
                            break

                        results = model.track(
                            source=frame,
                            device="cpu",
                            imgsz=416,
                            conf=0.25,
                            tracker="bytetrack.yaml",
                            persist=True,
                            verbose=False
                        )

                        result = results[0]
                        annotated = result.plot()
                        writer.write(annotated)
                        frame_count += 1

                        if result.boxes is not None:
                            for box in result.boxes:
                                class_id = int(box.cls[0])
                                label = result.names[class_id]
                                counts[label] = counts.get(label, 0) + 1

                                if box.id is not None:
                                    track_ids.add(int(box.id[0]))

                    cap.release()

                    if writer:
                        writer.release()
                        writer = None

                    total_objects = sum(counts.values())

                    for label, count in counts.items():
                        detections.append({
                            "name": label,
                            "count": count,
                            "confidence": 0
                        })

                    tracked_ids = track_ids
                    result_video = f"/static/results/{unique_id}.mp4"
                    uploaded_filename = video_file.filename

                    if frame_count == 0:
                        error = "No frames could be processed."

                except Exception as exc:
                    error = f"Video tracking failed: {exc}"

                finally:
                    if writer:
                        writer.release()

    return render_template(
        "index.html",
        detections=detections,
        result_image=result_image,
        result_video=result_video,
        total_objects=total_objects,
        uploaded_filename=uploaded_filename,
        error=error,
        tracked_ids=tracked_ids
    )


if __name__ == "__main__":
    app.run(debug=True)
