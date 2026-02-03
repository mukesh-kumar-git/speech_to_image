import streamlit as st
import requests
from PIL import Image
import io
import speech_recognition as sr
import tempfile


st.set_page_config(page_title="Speech / Text to Image", layout="centered")
st.title("🎤 Speech / Text to Image Generator")


HF_TOKEN = st.secrets.get("HF_TOKEN")

if not HF_TOKEN:
    st.error(
        "HF_TOKEN missing.\n\n"
        "Go to Manage App → Settings → Secrets and add:\n\n"
        'HF_TOKEN = "hf_xxxxxxxxxxxxxxxxx"'
    )
    st.stop()


API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-2-1"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


def generate_image(prompt):
    response = requests.post(
        "https://api-inference.huggingface.co/v1/images/generations",
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
            "prompt": prompt,
            "size": "1024x1024"
        },
        timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError(f"{response.status_code}: {response.text}")

    data = response.json()
    image_base64 = data["data"][0]["b64_json"]

    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes))


def speech_to_text(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        path = f.name

    r = sr.Recognizer()
    with sr.AudioFile(path) as source:
        audio_data = r.record(source)
        return r.recognize_google(audio_data)


prompt = st.text_input("✍️ Enter text prompt")

st.markdown("### 🎤 Or speak your prompt")
audio = st.audio_input("Record voice")

if audio:
    try:
        prompt = speech_to_text(audio.getbuffer())
        st.success(f"Recognized: {prompt}")
    except Exception:
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
                st.code(str(e))
