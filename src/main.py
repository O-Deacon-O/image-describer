import gradio as gr

from image_describer import ImageDescriber

def main() -> None:
    image_describer = ImageDescriber()

    # === Create and Launch the Gradio Interface ===
    app = gr.Interface(
        fn=image_describer.describe_image,
        inputs=[
            gr.Image(type="pil", label="Upload an image"),
            gr.Slider(minimum=100, maximum=300, value=200, step=1, label="Max tokens:")
        ],
        outputs=gr.Textbox(label="Generated description:"),
        title="Image Descsriber",
        description="Upload an image and the model will describe it."
    )
    app.launch(inbrowser=True)
    
if __name__ == "__main__":
    main()