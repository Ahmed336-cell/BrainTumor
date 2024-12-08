import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model("BrainTumors.h5")

# Define class labels
class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']  # Update based on your dataset


def predict_tumor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize the image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return class_labels[predicted_class]


# Streamlit UI
st.title("Brain Tumor Detection")
st.write("Upload an MRI image to detect if a brain tumor is present.")

uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_file = Image.open(uploaded_file)
    st.image(image_file, caption="Uploaded MRI Image", use_column_width=True)

    # Save the uploaded image temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict and display the result
    result = predict_tumor("temp_image.jpg")
    st.write(f"**Prediction:** {result}")
