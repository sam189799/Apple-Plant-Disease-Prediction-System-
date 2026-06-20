"""import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load model
model = load_model("apple_disease_model.h5")

# Classes
classes = [
    "Apple Scab",
    "Black Rot",
    "Cedar Apple Rust",
    "Healthy"
]

# Enter image path
image_path = input("Enter image path: ")

# Check if file exists
if not os.path.exists(image_path):
    print("Image not found!")
    exit()

# Load image
img = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

# Convert to array
img_array = tf.keras.utils.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Normalize
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

# Get class
result = classes[np.argmax(prediction)]

print("\nPrediction:", result)"""





import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("apple_disease_model.h5")

# Class names (must match train_data.class_names)
classes = [
    "Apple_scab",
    "Black_rot",
    "healthy"
]

# Enter image path
image_path = input("Enter image path: ")

# Load image
img = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)

# Convert image to array
img_array = tf.keras.utils.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Normalize
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

# Show scores
print("\nPrediction Scores:")
for i in range(len(classes)):
    print(f"{classes[i]} : {prediction[0][i]*100:.2f}%")

# Final result
index = np.argmax(prediction)

print("\nFinal Prediction:", classes[index])
print("Confidence:", f"{prediction[0][index]*100:.2f}%")