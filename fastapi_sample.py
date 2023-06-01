import shutil
import tempfile
import requests
from backend import YOLO_inf, bboxes, draw_bbox, delete_folder
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

url = "http://mai-clinic.neijena.com"
# Set up CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:44311",
    "http://localhost:6688",
    "http://mai-clinic.neijena.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app = FastAPI()

async def main():
    r = requests.get(url)
    print(r.status_code)
    return r.status_code

@app.get("/temp/annotated_image.png")
async def get_image():
    file_path = fr"E:\XR-AI\temp\annotated_image.png"
    return FileResponse(file_path)


@app.post("/button-action")
async def button_action(file: UploadFile = File(...)):
    try:
        # Check if the uploaded file is in PNG format
        #if file.content_type != "image/png":
        #    return {"error_found": "Only PNG files are allowed."}

        # Save the uploaded file to a temporary PNG file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Process the image using the file path
        print("Processing image...")
        delete_folder(r"E:\XR-AI\temp")
        result_csv, processed_image_path = YOLO_inf(tmp_path)  # Use file.filename instead of tmp_file.name
        print("Collecting result...")
        boxes, scores, labels = bboxes(result_csv, 1024, 1024)
        print("Drawing final bounding boxes...")
        output = draw_bbox(processed_image_path, boxes, scores, labels)

        # Return the processed image path as a JSON response
        return "temp/annotated_image.png"

    except Exception as e:
        # Handle any errors and return an appropriate response
        return {"error_found": str(e)}
