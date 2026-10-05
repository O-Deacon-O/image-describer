import torch

from config import MODEL_ID
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

class ImageDescriber:
    """Class for generating descriptions of images."""
    def __init__(self) -> None:
        """Initialize the ImageDescriber."""
        # === Set the Device and Data Type ===
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32  # Use float16 on CUDA to reduce GPU memory usage; use float32 on CPU for broader compatibility.

        # === Load the Model and Processor ===
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype).to(self.device)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

        self.model.eval()

    def describe_image(self, image: Image.Image, max_new_tokens: int) -> str:
        """Generate a description for the given image."""
        # === Prepare the Conversation for the Chat Template ===
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": (
                            "Describe only what is clearly visible in this image. "
                            "Do not guess identities, objects, or events. "
                            "If something is unclear, say so."
                        )
                    }
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.device)

        # === Generate the Description Using the Model ===
        with torch.inference_mode():  # Disable gradient tracking for inference
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # === Extract the Generated Answer by Removing the Prompt Tokens ===
        prompt_length = inputs["input_ids"].shape[1]
        answer_ids = generated_ids[:, prompt_length:]

        # === Decode the Answer IDs to Get the Final Description ===
        return self.processor.batch_decode(answer_ids, skip_special_tokens=True)[0]