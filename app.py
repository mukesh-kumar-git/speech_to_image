import streamlit as st
import requests
from PIL import Image
import io
import speech_recognition as sr
import tempfile


st.set_page_config(page_title="Speech / Text to Image", layout="centered")
st.title("🎤 Speech / Text to Image Generator")

# Hugging Face token from Streamlit Secrets
HF_TOKEN = st.secrets["HF_TOKEN"]

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def generate_image(prompt):
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={"inputs": prompt},
        timeout=120
    )
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))

def speech_to_text(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        audio_path = f.name

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)


prompt = st.text_input("✍️ Enter text prompt")

st.markdown("### 🎤 Or speak your prompt")
audio = st.audio_input("Record voice")

if audio:
    try:
        prompt = speech_to_text(audio.getbuffer())
        st.success(f"Recognized: {prompt}")
    except Exception as e:
        st.error("Speech recognition failed")

if st.button("🎨 Generate Image"):
    if not prompt:
        st.warning("Please enter or speak a prompt")
    else:
        with st.spinner("Generating image..."):
            try:
                img = generate_image(prompt)
                st.image(img, use_container_width=True)
                st.success("Done")
            except Exception as e:
                st.error("Image generation failed")
