import os
import io
import pickle
import requests
import numpy as np
import pandas as pd
from PIL import Image
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Tensorflow / Keras
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

from dotenv import load_dotenv

# 1. Load Environment Variables from .env file
load_dotenv()

# Retrieve keys safely
HF_OWNER = os.getenv("HF_DATASET_OWNER")
HF_REPO = os.getenv("HF_DATASET_REPO")
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")

# Critical Check: Stop if keys are missing
if not all([HF_OWNER, HF_REPO, CLOUDINARY_CLOUD_NAME]):
    raise ValueError(
        "❌ MISSING CONFIGURATION! \n"
        "Please check your .env file. It must contain: HF_DATASET_OWNER, HF_DATASET_REPO, and CLOUDINARY_CLOUD_NAME"
    )

# 2. URLs (Dynamic based on env vars)
# CHECK THESE FILENAMES: Ensure they match your Hugging Face repo EXACTLY.
EMBEDDINGS_URL = f"https://huggingface.co/datasets/{HF_OWNER}/{HF_REPO}/resolve/main/embeddings.pkl"
FILENAMES_URL = f"https://huggingface.co/datasets/{HF_OWNER}/{HF_REPO}/resolve/main/filenames.pkl"
CSV_URL = f"https://huggingface.co/datasets/{HF_OWNER}/{HF_REPO}/resolve/main/styles.csv"
CLOUDINARY_URL = f"https://res.cloudinary.com/{CLOUDINARY_CLOUD_NAME}/image/upload/"

# Global State Dictionary to hold models
ml_models = {}

# 3. Helper: Load Data safely
def load_pickle_from_hf(url):
    print(f"Loading: {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return pickle.load(io.BytesIO(response.content))
    except Exception as e:
        print(f"❌ Error loading {url}: {e}")
        raise e

def load_csv_from_hf(url):
    print(f"Loading: {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text), on_bad_lines='skip')
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        raise e

# 4. Lifespan Manager (Startup/Shutdown logic)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("🚀 Starting up: Downloading models and data...")
    
    try:
        # Load Data
        feature_list = np.array(load_pickle_from_hf(EMBEDDINGS_URL))
        filenames = load_pickle_from_hf(FILENAMES_URL)
        styles_df = load_csv_from_hf(CSV_URL)
        
        # Load ResNet50
        print("Loading ResNet50 model...")
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        model = Sequential([base_model, GlobalMaxPooling2D()])
        
        # Train KNN once at startup
        print("Fitting NearestNeighbors...")
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)

        # Store in global state
        ml_models["model"] = model
        ml_models["feature_list"] = feature_list
        ml_models["filenames"] = filenames
        ml_models["styles_df"] = styles_df
        ml_models["neighbors"] = neighbors
        
        print("✅ System Ready!")
    except Exception as e:
        print(f"🛑 CRITICAL STARTUP ERROR: {e}")
        raise e # Stop server if startup fails
        
    yield
    
    # --- SHUTDOWN ---
    ml_models.clear()
    print("🛑 Shutting down.")

# 5. Initialize App
app = FastAPI(title="Fashion Recommendation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6. Feature Extraction Logic
def feature_extraction(img: Image.Image, model):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized = result / norm(result)
    return normalized

def get_metadata(image_filename, df):
    try:
        # Try to extract ID from filename (e.g., "1163.jpg" -> 1163)
        image_id = int(os.path.splitext(image_filename)[0])
        row = df[df['id'] == image_id]
        if not row.empty:
            return {
                "gender": row["gender"].values[0],
                "articleType": row["articleType"].values[0],
                "baseColour": row["baseColour"].values[0],
                "usage": row["usage"].values[0] if "usage" in row.columns else None
            }
    except ValueError:
        pass 
    except Exception as e:
        print(f"Metadata error: {e}")
    return {}

# 7. Endpoint
@app.post("/recommend")
async def recommend_fashion(file: UploadFile = File(...)):
    try:
        # Read Image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Access global models
        model = ml_models["model"]
        neighbors = ml_models["neighbors"]
        feature_list = ml_models["feature_list"]
        filenames = ml_models["filenames"]
        styles_df = ml_models["styles_df"]

        # Extract Features
        features = feature_extraction(img, model)
        
        # Inference
        distances, indices = neighbors.kneighbors([features])
        
        # Get top 5 recommendations
        result_indices = indices[0][0:5] 

        results = []
        for idx in result_indices:
            filename = os.path.basename(filenames[idx])
            image_url = CLOUDINARY_URL + filename
            metadata = get_metadata(filename, styles_df)
            
            results.append({
                "filename": filename,
                "image_url": image_url,
                "metadata": metadata,
                "score": float(distances[0][list(result_indices).index(idx)]) 
            })

        return {"recommendations": results}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)