import os
import tensorflow as tf
import keras
import gdown

# --- CONFIGURATION ---
INPUT_MODEL_PATH = "models/yolov8_model_manuel_kayit.keras"
OUTPUT_MODEL_PATH = "yolov8_high_acc.tflite"
GDRIVE_FILE_ID = '1JZ0OmNuOIK8l4xxo5KoThcNpykMzsCtq'

# 1. DOWNLOAD MODEL (If not exists)
if not os.path.exists(INPUT_MODEL_PATH):
    print("🌐 Downloading original .keras model...")
    
    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
    
    # Create directory if it doesn't exist
    if os.path.dirname(INPUT_MODEL_PATH):
        os.makedirs(os.path.dirname(INPUT_MODEL_PATH), exist_ok=True)
        
    gdown.download(url, INPUT_MODEL_PATH, quiet=False)
else:
    print("✅ Model file already exists. Skipping download.")

# 2. LOAD KERAS MODEL
print(f"⏳ Loading Keras model from: {INPUT_MODEL_PATH}")
try:
    model = keras.models.load_model(INPUT_MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# 3. CONVERT TO TFLITE (FLOAT32 - HIGH PRECISION)
print("🔧 Starting conversion... (This may take a few minutes)")

# Define a concrete function to fix the input shape.
# This ensures the model accepts exactly (1, 640, 640, 3).
@tf.function
def inference_func(images):
    return model.predict_step(images)

concrete_func = inference_func.get_concrete_function(
    tf.TensorSpec([1, 640, 640, 3], tf.float32)
)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# --- CRITICAL CONFIGURATION ---
# Optimizations (Quantization) are intentionally DISABLED.
# We keep the model in Float32 format to maintain maximum accuracy.

# Enable TensorFlow ops to ensure compatibility
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS,
  tf.lite.OpsSet.SELECT_TF_OPS
]

try:
    tflite_model = converter.convert()
    
    # 4. SAVE TFLITE MODEL
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)

    print(f"\n🎉 SUCCESS! Model saved to: '{OUTPUT_MODEL_PATH}'")
    
    size_mb = os.path.getsize(OUTPUT_MODEL_PATH) / (1024 * 1024)
    print(f"📁 File Size: {size_mb:.2f} MB")
    
except Exception as e:
    print(f"\n❌ Conversion Error: {e}")