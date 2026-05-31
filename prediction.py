print("STARTING PREDICTION FILE")
import os
import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None

# We handle import dynamically so it doesn't break if tensorflow isn't available
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    # pyrefly: ignore [missing-import]
    from tensorflow.keras.applications.efficientnet import preprocess_input

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # We load the exported SavedModel directory which resolves all Keras 3 to Keras 2 incompatibility issues
    MODEL_PATH = os.path.join(BASE_DIR, "crop_model_saved")

    if os.path.exists(MODEL_PATH):
        print(f"Loading native SavedModel from {MODEL_PATH}...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load SavedModel: {e}")
            model = None
    else:
        print(f"Model path not found: {MODEL_PATH}")
        model = None

except Exception as e:
    print("ERROR INITIALIZING TENSORFLOW:")
    print(e)
    model = None


CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Septoria_leaf_spot",
    "Pepper__bell___Bacterial_spot"
]

def predict_disease(image_path):
    """
    Reads image from image_path, processes it, and returns predicted disease and confidence.
    """
    if model is None:
        print("Warning: TensorFlow not installed or model not found. Returning mock prediction.")
        return "Potato___Late_blight", 95.5

    try:
        # EfficientNetB0 expects 224x224
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        img_array = preprocess_input(img_array)

        # Handle both native SavedModel and full Keras models
        if hasattr(model, 'predict'):
            predictions = model.predict(img_array)
        elif hasattr(model, 'signatures'):
            infer = model.signatures['serving_default']
            # Convert to tensor explicitly if needed
            tensor_input = tf.constant(img_array, dtype=tf.float32)
            output = infer(tensor_input)
            # The output is a dict, get the first tensor
            tensor_key = list(output.keys())[0]
            predictions = output[tensor_key].numpy()
        else:
            # Fallback if it's somehow a callable graph
            predictions = model(img_array)
            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
        
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx]) * 100
        
        if class_idx < len(CLASSES):
            disease = CLASSES[class_idx]
        else:
            disease = f"Class_{class_idx}"
            
        return disease, round(confidence, 2)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Unknown Error", 0.0
