import os
import shutil
import uuid
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image

# === Initialize FastAPI ===
app = FastAPI(title="AI Detection API")

# === Enable CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Directory setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# === Mount static folder (for serving result images) ===
app.mount("/static", StaticFiles(directory="static"), name="static")

# === Load YOLO models ===
soil_model_path = os.path.join(BASE_DIR, "weights", "soil_best.pt")
vegetation_model_path = os.path.join(BASE_DIR, "weights", "vegetation_best.pt")

print("Loading YOLOv11 (Soil) model...")
soil_model = YOLO(soil_model_path)

print("Loading YOLOv8 (Vegetation) model...")
vegetation_model = YOLO(vegetation_model_path)


# === Helper: Resize uploaded images ===
def resize_image(image_path, size=(640, 640)):
    """Resize uploaded image to match YOLO training image dimensions."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(size)
        img.save(image_path)
        print(f"✅ Resized {os.path.basename(image_path)} to {size}")
    except Exception as e:
        print(f"❌ Error resizing image {os.path.basename(image_path)}: {e}")


# === Helper: Run YOLO detection ===
def run_detection(model, file: UploadFile, result_subdir: str):
    """Run YOLO detection and return detected type, confidence, and result image path."""

    # ✅ Save uploaded file with unique name to avoid caching issues
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    upload_path = os.path.join(UPLOAD_DIR, unique_filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ✅ Resize to match YOLO input dimensions
    resize_image(upload_path)

    # Run YOLO detection
    results = model.predict(
        source=upload_path,
        conf=0.25,
        save=True,
        project=RESULT_DIR,
        name=result_subdir,
        exist_ok=True
    )

    detections = results[0].boxes
    names = model.names

    summary = []  # for detection summary panel

    if detections is None or len(detections) == 0:
        detected_type = "No detection"
    else:
        class_ids = detections.cls.tolist()
        confidences = detections.conf.tolist()
        classes = [names[int(cls_id)] for cls_id in class_ids]

        # === Create a detection summary ===
        from collections import Counter
        class_count = Counter(classes)

        for cls, conf in zip(classes, confidences):
            summary.append({
                "class": cls,
                "confidence": round(conf * 100, 2)
            })

        detected_type = ", ".join(list(set(classes)))

    # Get latest result image
    result_dir = os.path.join(RESULT_DIR, result_subdir)
    output_files = [f for f in os.listdir(result_dir) if f.endswith((".jpg", ".png"))]
    output_files.sort(key=lambda f: os.path.getmtime(os.path.join(result_dir, f)))

    if not output_files:
        result_image = None
    else:
        output_image = output_files[-1]
        result_image = f"/static/results/{result_subdir}/{output_image}"

    # === Prepare summary output ===
    detection_summary = {}
    if summary:
        from collections import Counter
        class_counts = Counter([item["class"] for item in summary])
        detection_summary = {
            "detected_classes": list(class_counts.keys()),
            "class_counts": dict(class_counts),
            "detailed": summary
        }
    else:
        detection_summary = {"detected_classes": [], "class_counts": {}, "detailed": []}

    return detected_type, result_image, detection_summary


# === Routes ===
@app.get("/")
def home():
    return {"message": "Backend is running successfully!"}


# --- Soil Detection ---
@app.post("/predict/soil")
async def predict_soil(file: UploadFile = File(...)):
    soil_type, result_image, summary = run_detection(soil_model, file, "soil_result")
    return JSONResponse({
        "soil_type": soil_type,
        "result_image": result_image,
        "summary": summary
    })


# --- Vegetation Detection ---
@app.post("/predict/vegetation")
async def predict_vegetation(file: UploadFile = File(...)):
    vegetation_type, result_image, summary = run_detection(vegetation_model, file, "vegetation_result")
    return JSONResponse({
        "vegetation_type": vegetation_type,
        "result_image": result_image,
        "summary": summary
    })


# === Run app ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
