from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import io

app = FastAPI()

# Load model and encoder
model = tf.keras.models.load_model("best_sleeve_model.keras")
with open("sleeve_length_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Define image preprocessing function
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # Change this if your model expects a different size
    image_array = np.array(image) / 255.0  # Normalize if your model was trained that way
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.get("/")
def root():
    return {"message": "Sleeve Length Image Prediction API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        # Make prediction
        prediction = model.predict(input_tensor)
        class_index = np.argmax(prediction, axis=1)
        label = encoder.inverse_transform(class_index)

        return JSONResponse(content={"sleeve_length": label[0]})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
