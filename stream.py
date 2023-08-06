import streamlit as st
from PIL import Image
import tempfile
from roboflow import Roboflow
import os

def main():
    st.title("Mushroom Classification App")
    st.write("Upload an image of a mushroom to classify its species.")

    # Create an instance of Roboflow
    rf = Roboflow(api_key="caFNXOrnEdmKjr8A0dhG")
    project = rf.workspace().project("myshroomclassifier")
    model = project.version(1).model

    # Image uploader
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Read the uploaded image as PIL Image
        image = Image.open(uploaded_image)

        # Save the PIL image as a temporary file
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_image.name)

        # Perform inference on the uploaded image
        output = model.predict(temp_image.name)
        prediction = output.json()
        
        # Display the prediction result
        st.subheader("Prediction:")
        st.write(prediction)

        # Save the prediction image locally
        prediction_image_path = "prediction.jpg"
        output.save(prediction_image_path)

        # Display the prediction image
        st.subheader("Prediction Image:")
        st.image(prediction_image_path, caption="Prediction Image", use_column_width=True)

        # Close and delete the temporary file
        temp_image.close()
        os.remove(temp_image.name)
        os.remove(prediction_image_path)

if __name__ == "__main__":
    main()

