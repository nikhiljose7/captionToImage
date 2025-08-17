import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Set the page configuration for a better layout
st.set_page_config(page_title="AI Image Generator", layout="wide")

# --- Model Loading Function ---
# Use the @st.cache_resource decorator to load and cache the model.
# This ensures the model is loaded only once.
@st.cache_resource
def load_model():
    """Loads and caches the Stable Diffusion model."""
    model_id = "runwayml/stable-diffusion-v1-5"
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            # Inform the user if running on CPU, which is slower.
            st.warning("GPU not available. The model will run on the CPU, which may be very slow.")
        return pipe
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# --- UI Layout ---
st.title("🎨 AI Image Generation App")
st.write("Create stunning images from text descriptions using the Stable Diffusion model.")

# Load the model with a spinner to show progress.
with st.spinner("Loading the AI model... This might take a few minutes on first run."):
    pipe = load_model()

if pipe:
    st.success("Model loaded successfully!")

    # --- User Input Section ---
    st.header("1. Describe Your Image")
    prompt = st.text_area(
        "**Enter your prompt:**",
        value="A hyperrealistic, cinematic shot of an astronaut riding a horse on Mars, beautiful lighting, 8k",
        height=120,
        help="Be as descriptive as possible for the best results."
    )

    negative_prompt = st.text_area(
        "**Enter things to avoid (negative prompt):**",
        value="ugly, blurry, low quality, deformed, watermark, text, signature",
        height=120,
        help="Specify what you don't want to see in the image."
    )

    # --- Image Generation Button ---
    st.header("2. Generate Your Image")
    if st.button("Generate Image", type="primary"):
        if not prompt:
            st.warning("Please enter a prompt to generate an image.")
        else:
            with st.spinner("Generating... This might take a moment."):
                try:
                    # Generate the image using the pipeline
                    image = pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
                    
                    # Display the generated image
                    st.image(image, caption="Generated Image", use_column_width=True)
                    st.success("Image generated!")

                except Exception as e:
                    st.error(f"An error occurred during image generation: {e}")
else:
    st.error("Model could not be loaded. The application cannot proceed.")


# --- Footer ---
st.markdown("---")
st.markdown("Powered by [Streamlit](https://streamlit.io) and [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/index).")