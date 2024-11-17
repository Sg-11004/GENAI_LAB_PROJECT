import streamlit as st
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

model_path = 'fl.hdf5'
model = load_model(model_path)
class_names=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

img_width, img_height = 224, 224

def preprocess_image(image):
    img = image.resize((img_width, img_height))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

def main():
    st.title('Image Classification')
    st.sidebar.subheader('UPLOAD YOUR TEST IMAGE')
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        prediction = st.button("Predict")
        if prediction:
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)[0]
            predicted_class = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class]
            st.write(f"Predicted class: {predicted_class_name}")
            st.write("Prediction probabilities:")
            for i, prob in enumerate(prediction):
                st.write(f"{class_names[i]}: {prob * 100:.2f}%")
if __name__ == "__main__":
    main()
