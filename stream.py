import streamlit as st
from streamlit.proto.Image_pb2 import Image as ImageProto
from PIL import Image
import tempfile
from roboflow import Roboflow
import os
import uuid
from transformers import pipeline
from gradio_client import Client

@st.cache_resource
def load_image_classification_model():
    pipe = pipeline("image-classification", model="elucidator8918/VIT-MUSH")
    return pipe

@st.cache_resource
def robo():
    rf = Roboflow(api_key="caFNXOrnEdmKjr8A0dhG")
    project = rf.workspace().project("myshroomclassifier")
    model = project.version(1).model
    return model

@st.cache_resource
def zephyr():
    client = Client("https://library-samples-zephyr-7b-alpha.hf.space/--replicas/wdbkk/")
    return client
    
def mushroom_classification():
    st.title("Mushroom Classification")
    st.write("Upload an image of a mushroom to classify its species.")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        image = Image.open(uploaded_image)
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_image.name)
        model = robo()
        output = model.predict(temp_image.name)
        prediction = output.json()
        st.subheader("Mushroom Classification Prediction:")
        st.write(prediction)
        prediction_image_path = f"{uuid.uuid4()}.jpg"
        output.save(prediction_image_path)
        st.subheader("Prediction Image:")
        st.image(prediction_image_path, caption="Prediction Image", use_column_width=True)
        temp_image.close()
        os.remove(temp_image.name)
        os.remove(prediction_image_path)

def image_detection_with_chatbot():
    st.title("Mushroom Image Detection with Chatbot Interaction")
    st.write("Upload an image for mushroom image detection.")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        pipe = load_image_classification_model()
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        image = Image.open(uploaded_image)
        detected_objects = pipe(image)
        st.subheader("Detected Objects:")
        st.write(detected_objects)
        chatbot_message = f"Tell me more about the mushroom {detected_objects[0]['label']}"
        system_prompt = "You are a helpful mushroom specialist virtual assistant that answers user's questions with easy to understand words."
        client,pipe = zephyr()
        result = client.predict(
            chatbot_message,
            system_prompt,
            1024,
            0.7,
            0.95,
            50,
            1,
            api_name="/chat"
        )
        st.subheader("Zephrx Response:")
        st.write(result)
        st.subheader("FactCC Checks")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Mushroom Classification", "Image Detection"])

    if page == "Mushroom Classification":
        mushroom_classification()
    elif page == "Image Detection":
        image_detection_with_chatbot()

if __name__ == "__main__":
    main()
