from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import cv2
import numpy as np
from PIL import Image
import io
import base64


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/ui", include_in_schema=False)
def custom_ui(request: Request):
    return templates.TemplateResponse("ui.html", {"request": request})



# Load the pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# COCO classes used by YOLOv8
COCO_CLASSES = model.model.names

# Map categories to class IDs
CATEGORY_MAP = {
    "human": [0],  # person
    "vehicle": [2, 3, 5, 7],  # car, motorcycle, bus, truck
    "animal": [15, 16, 17, 18, 19, 20, 21, 22, 23],  # cat, dog, horse, etc.
}

@app.post("/detect/")
async def detect(category: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = model(image_np)
    filtered_boxes = []

    allowed_ids = CATEGORY_MAP.get(category, [])

    for result in results:
        new_boxes = []
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in allowed_ids:
                new_boxes.append(box)
        result.boxes = new_boxes
        annotated = result.plot()

    _, img_encoded = cv2.imencode(".jpg", annotated)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

@app.post("/detect/")
async def detect(category: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = model(image_np)
    filtered_boxes = []

    allowed_ids = CATEGORY_MAP.get(category, [])

    for result in results:
        new_boxes = []
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in allowed_ids:
                new_boxes.append(box)
        result.boxes = new_boxes
        annotated = result.plot()

    _, img_encoded = cv2.imencode(".jpg", annotated)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

@app.post("/filter/{filter_type}")
async def apply_filter(filter_type: str, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if filter_type == "gaussian":
        filtered = cv2.GaussianBlur(img_np, (11, 11), 0)
    elif filter_type == "median":
        filtered = cv2.medianBlur(img_np, 9)
    elif filter_type == "bilateral":
        filtered = cv2.bilateralFilter(img_np, 9, 75, 75)
    elif filter_type == "edges":
        edges = cv2.Canny(img_np, 100, 200)
        filtered = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    else:
        return {"error": "Unsupported filter type"}

    _, img_encoded = cv2.imencode(".jpg", filtered)
    img_bytes = img_encoded.tobytes()
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")


