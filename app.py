from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import cv2
from ultralytics import YOLO
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()
model = YOLO("yolov8n.pt")  # Using the small YOLO model
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("camera.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/detect")
def detect():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return {"message": "Detection stopped"}

    
from pydantic import BaseModel
import base64
import numpy as np

class ImageData(BaseModel):
    image: str

@app.post("/detect_image")
async def detect_image(data: ImageData):
    # Decode base64 image
    image_data = data.image.split(",")[1]
    decoded = base64.b64decode(image_data)
    nparr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect objects
    results = model(img, stream=False)
    labels = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            labels.append(model.names[cls])

    return {"detected": list(set(labels))}
