from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import cv2
import os
import shutil


app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


model = YOLO("yolov8n.pt")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("camera/detect", response_class=HTMLResponse)
async def detect(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    temp_file = "temp.jpg"

    with open(temp_file, "wb") as f:
        f.write(contents)

    results = model(temp_file)

    
    output_path = os.path.join("static", "result.jpg")
    results[0].save(filename=output_path)

    os.remove(temp_file)  # Clean up

    return templates.TemplateResponse("detect.html", {"request": request, "result_path": "/static/result.jpg"})

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
