import streamlit as st
import requests
from PIL import Image
import io
import speech_recognition as sr
import tempfile

st.set_page_config(page_title="Speech / Text to Image", layout="centered")
st.title("🎤 Speech / Text to Image Generator")

API_KEY = st.secrets.get("STABILITY_API_KEY")

if not API_KEY:
    st.error(
        "Missing Stability AI API key.\n\n"
        "Go to Manage App → Settings → Secrets and add:\n\n"
        'STABILITY_API_KEY = "sk-xxxxxxxxxxxxxxxx"'
    )
    st.stop()

def generate_image(prompt):
    response = requests.post(
        "https://api.stability.ai/v2beta/stable-image/generate/sd3",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "image/*"
        },
        files={
            "prompt": (None, prompt),
            "output_format": (None, "png")
        },
        timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError(f"{response.status_code}: {response.text}")

    return Image.open(io.BytesIO(response.content))


def speech_to_text(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        audio_path = f.name

    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
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
