import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import openai
import requests
from io import BytesIO
import base64
import numpy as np
from utils import (
    load_baseline_model,
    generate_baseline_caption,
    enhance_with_openai,
    load_image,
    preprocess_image
)

# --- Configuration ---
st.set_page_config(page_title="Seeing & Speaking", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
@st.cache_resource
def get_models():
    return load_baseline_model()

encoder, decoder, vocab = get_models()

# --- Helper Functions ---
def encode_image_to_base64(image):
    """Convert PIL image to base64 for OpenAI API"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_gpt4_vision_caption(base64_image):
    """Get standalone caption from GPT-4 Vision"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image accurately and concisely."},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"GPT-4 Vision Error: {str(e)}")
        return None

# --- UI Layout ---
st.title("Vision to Text: Baseline 🆚 OpenAI")
st.caption("Compare:Baseline CNN-RNN model | GPT-3.5 Enhanced | GPT-4 Vision")

# Sidebar
with st.sidebar:
    st.header("Settings")
    openai_enabled = st.toggle("Enable OpenAI", True)
    st.divider()
    st.markdown("""
    **Model Details**  
    - Baseline: ResNet50 + Attention LSTM  
    - OpenAI: GPT-3.5 Turbo  
    [View Baseline Model](https://huggingface.co/weakyy/image-captioning-baseline-model)
    """)
    st.write("Model Status:", 
        f"Encoder: {'Loaded' if encoder else 'Failed'}",
        f"Decoder: {'Loaded' if decoder else 'Failed'}",
        f"Vocab Size: {len(vocab['word2idx']) if vocab else 0}"
    )

# Main Content
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
example_images = {
    "Beach": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=600",
    "Dog": "https://images.unsplash.com/photo-1561037404-61cd46aa615b?w=600",
    "Food": "https://images.unsplash.com/photo-1565958011703-72f8583c2708?w=600"
}

# Process image
if not uploaded_file:
    selected = st.selectbox("Or try an example:", list(example_images.keys()))
    image = load_image(example_images[selected])
else:
    image = load_image(uploaded_file)

if image:
    st.image(image, caption="Input Image", use_container_width=True)
    image_tensor = preprocess_image(image, device)
    base64_image = encode_image_to_base64(image)

    col1, col2, col3 = st.columns(3)

    # 1. Baseline Model
    with col1:
        st.subheader("🧠 Baseline CNN+RNN Model")
        with st.spinner("Generating baseline CNN-RNN caption..."):
            baseline_result = generate_baseline_caption(
                image_tensor=image_tensor,
                encoder=encoder,
                decoder=decoder,
                vocab=vocab,
                beam_size=3
            )
            st.success(baseline_result["caption"])
            st.caption(f"Confidence: {baseline_result['confidence']:.0%}")

        st.write("Rate this caption:")
        if st.button("👍", key="like_baseline"):
            st.toast("Thanks for your feedback!")
        st.button("👎", key="dislike_baseline")

    # 2. GPT-3.5 Enhanced Baseline Model
    with col2:
        st.subheader("🔍 GPT-3.5 Enhanced")
        if openai_enabled and 'openai_key' in st.secrets:
            with st.spinner("Enhancing caption with GPT-3.5..."):
                enhanced = enhance_with_openai(baseline_result["caption"])
                if enhanced:
                    st.success(enhanced)
                else:
                    st.error("Enhancement failed")
        else:
            st.warning("Enable OpenAI in sidebar")

        st.write("Rate this enhancement:")
        if st.button("👍", key="like_openai_enhancement"):
            st.toast("Thanks for your feedback!")
        st.button("👎", key="dislike_openai_enhancement")

    # 3. GPT-4 Vision (Standalone)
    with col3:
        st.subheader("✨ GPT-4 Vision")
        if 'openai_key' in st.secrets:
            with st.spinner("Analyzing image..."):
                vision_caption = get_gpt4_vision_caption(base64_image)
                if vision_caption:
                    st.success(vision_caption)
                else:
                    st.error("Vision analysis failed")
        else:
            st.warning("Add OpenAI key to enable")

# Footer
st.divider()
st.caption("""
⚡ **Tip**: Baseline CNN+RNN uses the trained CNN-RNN, GPT-3.5 refines that output, 
while GPT-4 Vision generates captions directly from pixels
""")
st.markdown("""
[GitHub Repo](https://github.com/your-repo) | 
[Model Card](https://huggingface.co/weakyy/image-captioning-baseline-model)
""")
