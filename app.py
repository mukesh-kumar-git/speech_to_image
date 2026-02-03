import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Text to Image", layout="centered")
st.title("🖼️ Text to Image Generator")

HF_TOKEN = st.secrets["HF_TOKEN"]
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def generate_image(prompt):
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={"inputs": prompt},
        timeout=120
    )
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))

prompt = st.text_input("Enter prompt")

if st.button("Generate"):
    if not prompt:
        st.warning("Enter a prompt")
    else:
        with st.spinner("Generating image..."):
            img = generate_image(prompt)
            st.image(img, use_container_width=True)
