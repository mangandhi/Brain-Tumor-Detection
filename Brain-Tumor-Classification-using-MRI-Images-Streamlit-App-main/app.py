import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from io import BytesIO
import gdown
import os

# Download the model file from Google Drive
def download_model():
    model_url = "https://drive.google.com/file/d/1lDBpPogpcNfdPWl3iw2GyX2JHPuWMppj/view?usp=sharing"  # Replace FILE_ID with your actual file ID
    output_path = "brain_tumor_model.h5"
    
    if not os.path.exists(output_path):
        st.write("Downloading model from Google Drive...")
        gdown.download(model_url, output_path, quiet=False)
        st.write("Model downloaded successfully!")
    
    return tf.keras.models.load_model(output_path)

# Load the model
model = download_model()

# Define the class names based on your dataset
class_names = ['no_tumor', 'meningioma', 'glioma', 'pituitary_tumor']  # Adjust this list as needed

# Function to adjust predictions
def adjusted_prediction(predicted_class):
    if predicted_class == 'glioma':
        return 'no tumor'
    elif predicted_class == 'no_tumor':
        return 'glioma'
    return predicted_class

def predict_image_class(uploaded_file):
    img = image.load_img(BytesIO(uploaded_file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Rescale the image
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    predicted_probability = np.max(prediction)  # Get the maximum probability
    adjusted_class_name = adjusted_prediction(predicted_class_name)
    return adjusted_class_name, predicted_probability

# Tumor information
tumor_info = {
    'no tumor': "No tumor detected. The MRI scan shows no signs of abnormal growth.",
    'meningioma': "Meningioma is a tumor that develops from the protective layers covering the brain and spinal cord. It's usually benign but can cause symptoms depending on its location.",
    'glioma': "Glioma is a type of tumor that originates in the glial cells of the brain. They can be aggressive and vary in their prognosis based on type and location.",
    'pituitary tumor': "Pituitary tumors are abnormal growths that develop in the pituitary gland. They can affect hormone levels and cause various symptoms, including vision problems."
}

# Streamlit app
st.title("Brain Tumor Classification")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a mode", ["Home", "Tumor Information"])

if app_mode == "Home":
    st.write("Upload an MRI image to predict the presence of a brain tumor.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")

        # Predict and display the result directly using the uploaded_file
        adjusted_class, predicted_probability = predict_image_class(uploaded_file)
        st.write(f"Predicted Class: *{adjusted_class}*")
        st.write(f"Confidence: *{predicted_probability * 100:.2f}%*")  # Display confidence

        # Additional feedback
        if predicted_probability < 0.6:  # You can adjust this threshold
            st.warning("The prediction confidence is low. Please consider reviewing the image quality or try another image.")

elif app_mode == "Tumor Information":
    st.header("Information About Brain Tumors")
    for tumor_type, info in tumor_info.items():
        st.subheader(tumor_type.capitalize())
        st.write(info)
