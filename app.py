import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import speech_recognition as sr
from PIL import Image
import tempfile

st.set_page_config(page_title="Speech/Text to Image", layout="centered")

st.title("🎤 Speech / Text to Image Generator")
st.markdown("Generate images using your voice or text prompt.")

@st.cache_resource
def load_pipeline():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    return pipe

pipe = load_pipeline()

def generate_image(prompt):
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        image = pipe(prompt, guidance_scale=8.5).images[0]
    return image

prompt_text = st.text_input("✍️ Enter prompt")

st.markdown("### 🎤 Or speak your prompt")
audio = st.audio_input("Record your voice")

if audio:
    st.audio(audio)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio.getbuffer())
        audio_path = f.name

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            st.success(f"Recognized: {text}")
            prompt_text = text
        except:
            st.error("Could not recognize speech")

if st.button("🎨 Generate Image"):
    if prompt_text:
        with st.spinner("Generating image..."):
            img = generate_image(prompt_text)
            st.image(img, caption="Generated Image", use_container_width=True)
            img.save("generated_image.png")
            st.success("Done!")
    else:
        st.warning("Please enter or speak a prompt")

#python -m streamlit run SI_streamlit2.py
#zhkr zeja mijk mavk
#madhuchillar01@gmail.com