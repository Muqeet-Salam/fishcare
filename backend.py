from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from typing import cast
from torch import Tensor
import torchvision.models.resnet

app = FastAPI(
    title="Image Processing API",
    description="FastAPI application for image processing with PyTorch")
origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    filename: str
    predictions: list
    confidence_scores: list


class ErrorResponse(BaseModel):
    error: str
    detail: str

MODEL_PATH = "New folder/fish_classifier_model.pth"
CLASS_NAMES = [
    'ArticCharFish', 'Bangus', 'BassFish', 'Big Head Carp',
    'Black Spotted Barb', 'Catfish', 'Climbing Perch', 'Fourfinger Threadfin',
    'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 'Gourami',
    'Grass Carp', 'Green Spotted Puffer', 'HaddockFish', 'Indian Carp',
    'Indo-Pacific Tarpon', 'Jaguar Gapote', 'Janitor Fish', 'Knifefish',
    'Long-Snouted Pipefish', 'Mosquito Fish', 'Mudfish', 'Mullet', 'Pangasius',
    'Perch', 'RedSnapper', 'Salmon', 'Scat Fish', 'Silver Barb', 'Silver Carp',
    'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia'
]
NUM_CLASSES = len(CLASS_NAMES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None


def load_model():
    global model
    if model is None:
        torch.serialization.add_safe_globals(
            [torchvision.models.resnet.ResNet])
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.get("/")
async def root():
    return {"message": "Image Processing API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/upload-image/", response_model=PredictionResponse)
async def upload_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            return JSONResponse(status_code=400,
                                content={
                                    "error": "Invalid file type",
                                    "detail": "Please upload an image file"
                                })

        # Read image file
        contents = await file.read()
        image_bytes = BytesIO(contents)

        # Open image with PIL
        image = Image.open(image_bytes)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms
        input_tensor = cast(Tensor, transform(image))
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        # Load model if not loaded
        model = load_model()

        # Perform inference
        with torch.no_grad():
            outputs = model(input_batch)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            # Get top 5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, 5)

            predictions = []
            confidence_scores = []

            for i in range(5):
                pred_index = int(top5_indices[i].item())  # Explicit cast
                if pred_index < len(CLASS_NAMES):
                    predictions.append(CLASS_NAMES[pred_index])
                else:
                    predictions.append(f"Unknown class {pred_index}")
                confidence_scores.append(float(top5_prob[i].item()))

        return PredictionResponse(filename=file.filename or "unknown.jpg",
                                  predictions=predictions,
                                  confidence_scores=confidence_scores)

    except Exception as e:
        import traceback
        traceback.print_exc()  # <-- log to console
        return JSONResponse(status_code=500,
                            content={
                                "error": "Processing failed",
                                "detail": str(e),
                                "trace": traceback.format_exc()
                            })


@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    try:
        # Read and process image
        contents = await file.read()
        image_bytes = BytesIO(contents)
        image = Image.open(image_bytes)

        # Basic image information
        info = {
            "filename": file.filename,
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "content_type": file.content_type
        }

        return JSONResponse(content=info)

    except Exception as e:
        return JSONResponse(status_code=500,
                            content={
                                "error": "Image processing failed",
                                "detail": str(e)
                            })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
