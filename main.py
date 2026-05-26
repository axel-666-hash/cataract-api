from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gdown

# Download model weights if not present
WEIGHTS_PATH = 'cataract_weights.weights.h5'
if not os.path.exists(WEIGHTS_PATH):
    print("Downloading model weights...")
    gdown.download(
        'https://drive.google.com/uc?id=1Crg5jWhnc6isjRs4kIKQf6rhPNEFX52q',
        WEIGHTS_PATH,
        quiet=False
    )
    print("✅ Weights downloaded!")