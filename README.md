# Fish Care – AI-Powered Fish Species and Disease Detection

Fish Care is an AI-powered fish analysis application designed to identify fish species and detect potential fish diseases from images. It combines two computer vision models into a unified application to assist users with fish identification and health assessment.

## Features

- **Fish Species Identification:** Uses a YOLOv8-based computer vision model to identify fish species from uploaded images.
- **Fish Disease Detection:** Uses a Roboflow-trained object detection model to detect potential fish diseases.
- **Unified Backend:** Integrates both models through a FastAPI backend.
- **Image Processing:** Performs image preprocessing before passing inputs to the respective models.
- **API-Based Inference:** Exposes endpoints for integrating model predictions with the application.

## Model Details

The application uses two computer vision models for different prediction tasks.

### 1. Fish Species Identification

- **Model:** YOLOv8
- **Task:** Fish species identification
- **Purpose:** Detects and identifies fish species from input images.

### 2. Fish Disease Detection

- **Platform:** Roboflow
- **Model:** Roboflow 3.0 Object Detection (Accurate)
- **Model Version:** `fish-disease-object-detect-bpxnf/1`
- **Task:** Fish disease object detection

**Validation Metrics:**

| Metric | Score |
|---|---:|
| mAP@50 | 95.6% |
| Precision | 98.4% |
| Recall | 91.6% |
| F1-score | 94.9% |

These metrics represent the Roboflow model's reported validation performance.

## Architecture

The application follows a modular architecture in which the frontend communicates with a FastAPI backend, which handles image preprocessing and model inference.

```text
                 Input Fish Image
                        |
                        v
                 FastAPI Backend
                        |
                        v
                 Image Preprocessing
                        |
               +--------+--------+
               |                 |
               v                 v
          YOLOv8 Model      Roboflow Model
               |                 |
               v                 v
        Species Detection   Disease Detection
               |                 |
               +--------+--------+
                        |
                        v
                 Prediction Results
```

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| Species Detection | YOLOv8 |
| Disease Detection | Roboflow |
| Deep Learning | PyTorch, Torchvision |
| Image Processing | Python Image Processing Libraries |
| API Integration | REST APIs |

## Getting Started

### Prerequisites

- Python 3.9 or later
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Muqeet-Salam/fishcare.git
cd fishcare
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux/macOS**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Configure Model Access

Configure the required model weights and Roboflow API credentials according to the backend configuration. Keep API keys and other secrets in environment variables rather than committing them to Git.

### 5. Run the Backend

Start the FastAPI application using the appropriate application module:

```bash
uvicorn main:app --reload
```

Replace `main:app` with the actual module and FastAPI application instance used in the repository if they differ.

Once running, the interactive API documentation is generally available at:

```text
http://127.0.0.1:8000/docs
```

## Project Highlights

- Integrated two computer vision models for complementary fish analysis tasks.
- Developed backend APIs to handle image preprocessing and model inference.
- Achieved **95.6% mAP@50** on the Roboflow disease detection model's validation set.
- Built as a collaborative project involving a six-member development team.

## Team

Developed collaboratively by a six-member team. Frontend repository link: https://github.com/Shariqzain/FishCare
