# Image Describer

A small Gradio app that uploads an image and uses a multimodal vision-language model to generate a text description.

## Features

- Upload an image from your computer
- Generate a natural-language description with a Hugging Face vision-language model
- Adjust the number of tokens used during generation
- Run locally with a CPU or CUDA-enabled GPU

## How It Works

1. The user uploads an image in the browser
2. The app builds a chat-style input containing the image and a prompt
3. The processor formats the image and text for the model
4. The model generates a response describing the image
5. The generated text is decoded and displayed back to the user

## Requirements

- Python 3.9+
- Windows, macOS, or Linux
- Optional: CUDA-enabled GPU for faster inference

## Installation

1. Clone or download the repository.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the app:
```bash
python src/main.py
```

**On startup:**
- A Gradio interface will pop up in your default browser. 
- Upload an image and the app will generate a description.

## Configuration

Edit `src/config.py` to customize:
- **MultiModal model**: `MODEL_ID`

## License

MIT License - Feel free to use this project