from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Cataract Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
IMG_SIZE = (300, 300)

def build_model():
    base = EfficientNetB3(input_shape=(300, 300, 3), include_top=False, weights=None)
    inputs = tf.keras.Input(shape=(300, 300, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    return Model(inputs, outputs)

model = build_model()
model.load_weights('cataract_weights.weights.h5')

@app.get("/")
def health_check():
    return {"status": "ok", "model": "cataract-detection-v2"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "diagnosis": CLASSES[class_idx],
        "confidence": round(confidence * 100, 2),
        "cataract_detected": CLASSES[class_idx] == "cataract",
        "cataract_probability": round(float(preds[0]) * 100, 2),
        "all_probabilities": {
            cls: round(float(prob) * 100, 2)
            for cls, prob in zip(CLASSES, preds)
        },
        "disclaimer": "This is a screening tool only. Consult an ophthalmologist for diagnosis."
    }