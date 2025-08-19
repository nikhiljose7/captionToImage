# backend.py
# --- Step 1: Install necessary libraries ---
!pip install fastapi uvicorn pyngrok diffusers transformers accelerate torch Pillow

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import base64
from io import BytesIO
from PIL import Image
import uvicorn
from pyngrok import ngrok, conf
import threading
import os

# --- Optional: Add your ngrok authtoken if you have one ---
# Replace "YOUR_NGROK_AUTHTOKEN" with your actual token from https://dashboard.ngrok.com/get-started/your-authtoken
conf.get_default().auth_token = "enter your ngork auth token"

print("Starting backend server...")

# ---- Step 2: Load the Stable Diffusion model AND your LoRA ----
print("Loading Stable Diffusion model...")
# Define the base model and your LoRA model from the Hugging Face Hub
base_model_id = "runwayml/stable-diffusion-v1-5"
lora_model_id = "nikhiljose7/lora-sdv1-5-minnal-murali"

# Use float16 for faster inference and less memory usage.
pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)

print(f"Loading LoRA weights from: {lora_model_id}")
# Load the LoRA weights into the base model
pipe.load_lora_weights(lora_model_id)

# Move the model to the GPU if available.
if torch.cuda.is_available():
    print("Moving model to GPU...")
    pipe = pipe.to("cuda")
else:
    print("CUDA not available. Using CPU (this will be very slow).")
    pipe = pipe.to("cpu")

# Set up the FastAPI app ----
app = FastAPI()

# Define the request body structure.
# The key must be "caption".
class Prompt(BaseModel):
    caption: str

@app.get("/")
def read_root():
    return {"status": "ok"}

# Define the image generation endpoint.
@app.post("/generate")
def generate_image(data: Prompt):
    """
    Generates an image from a text caption.
    """
    # --- MODIFIED: Add the trigger word to the prompt ---
    # This is crucial to activate your LoRA model's concept.
    trigger_word = "Minnal Murali"
    prompt_with_trigger = f"{trigger_word}, {data.caption}"

    print(f"Received caption: {data.caption}")
    print(f"Using full prompt: {prompt_with_trigger}")

    try:
        # Generate the image using the modified prompt.
        image = pipe(prompt_with_trigger).images[0]

        # Convert the PIL image to a base64 string to send via JSON.
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print("Image generated successfully.")
        return {"image_base64": img_str}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}, 500


# Run the server using Uvicorn and Ngrok ----
# We run Uvicorn in a separate thread so that we can start ngrok in the main thread.
def run_app():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Open a public tunnel to the local server
public_url = ngrok.connect(8000)
print("------------------------------------------------")
print(f"✅ Public API URL: {public_url}")
print("------------------------------------------------")
print("Copy this URL and paste it into the API_URL variable in your frontend.py script.")


# Start the FastAPI server in the background.
thread = threading.Thread(target=run_app, daemon=True)
thread.start()

# Keep the main thread alive to keep ngrok running.
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Shutting down server and ngrok tunnel.")
    ngrok.disconnect(public_url)
