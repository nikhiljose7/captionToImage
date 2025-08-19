# Caption to Image: A Stable Diffusion Image Generator

This project is a web application that generates images from text prompts using a fine-tuned Stable Diffusion model. It features a user-friendly interface built with Streamlit and a powerful backend powered by FastAPI and Diffusers. The model has been specially trained to generate images of the superhero "Minnal Murali."

## Features

* **Web Interface**: An interactive and easy-to-use web interface created with Streamlit.
* **Text-to-Image Generation**: Leverages the power of Stable Diffusion to turn your text descriptions into images.
* **Custom-Trained Model**: Utilizes a LoRA (Low-Rank Adaptation) model fine-tuned on images of "Minnal Murali," allowing for the creation of unique and specific images.
* **API-Based Backend**: A robust backend built with FastAPI that handles the image generation requests.
* **Ngrok Integration**: The backend uses ngrok to create a public URL, making it easy to connect the frontend to the backend, even when running locally.

## Project Structure

The repository is organized as follows:

* `app.py`: The main Streamlit application file for the frontend user interface.
* `backend.py`: The FastAPI backend server for image generation.
* `image_training_caption.ipynb`: A Jupyter notebook detailing the process of fine-tuning the Stable Diffusion model using LoRA.
* `requirements.txt`: A file listing the Python dependencies for the frontend application.

## Getting Started

To get this project up and running on your local machine, follow these steps:

### Prerequisites

* Python 3.7+
* pip (Python package installer)

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nikhiljose7/captiontoimage.git](https://github.com/nikhiljose7/captiontoimage.git)
    cd captiontoimage
    ```

2.  **Set up the Backend:**
    * Navigate to the repository directory and run the backend script. This will install all the necessary backend dependencies and start the FastAPI server.
        ```bash
        python backend.py
        ```
    * When the backend is running, it will generate a public ngrok URL. Copy this URL.

3.  **Set up the Frontend:**
    * Install the frontend dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    * Run the Streamlit application:
        ```bash
        streamlit run app.py
        ```
    * Open your web browser and navigate to the local URL provided by Streamlit.
    * In the sidebar of the web application, paste the ngrok URL you copied from the backend into the "Enter Backend API URL" field.

## Usage

1.  Once the frontend and backend are running and connected, you can start generating images.
2.  Enter a text prompt in the text area. For best results, include the trigger word "**Minnal Murali**" in your prompt.
3.  Click the "**Generate Image**" button.
4.  The application will send the prompt to the backend, and the generated image will be displayed on the page.

### Example Prompts:

* `Minnal Murali standing heroically with lightning in the background`
* `A dramatic scene of Minnal Murali running through a village during a storm`
* `Minnal Murali in his superhero costume, striking a powerful pose in front of a temple`

## Model Training

The `image_training_caption.ipynb` notebook provides a step-by-step guide on how the Stable Diffusion model was fine-tuned using LoRA. It covers the entire process, from installing dependencies and preparing the dataset to running the training script and testing the final model.

## Dependencies

### Frontend (`requirements.txt`)

* streamlit
* requests
* Pillow

### Backend (`backend.py`)

* fastapi
* uvicorn
* pyngrok
* diffusers
* transformers
* accelerate
* torch
* Pillow
