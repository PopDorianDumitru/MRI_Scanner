import gradio as gr
from PIL import Image
from model import open_model, classify_image
import numpy as np
from gen_images import generate_images_ui
import argparse


def main(model_path):
    # Load generator and discriminator once
    generator, discriminator = open_model(model_path)

    severity_levels = ["Normal", "Mild", "Moderate", "Severe", "Very Severe"]

    label_to_index = {
        "Normal": 0,
        "Mild": 1,
        "Moderate": 2,
        "Severe": 3,
        "Very Severe": 4
    }

    def generate_mri(severity: str, num_images: int):
        seeds = np.random.randint(0, 1e6, size=num_images).tolist()
        class_idx = label_to_index[severity]
        return generate_images_ui(model_path, seeds, class_idx)

    def classify_mri(image: Image.Image) -> str:
        return classify_image(discriminator, image)

    classification_interface = gr.Interface(
        fn=classify_mri,
        inputs=gr.Image(type="pil", label="Upload MRI Scan"),
        outputs=gr.Label(label="Predicted Dementia Severity"),
        title="MRI Dementia Severity Classifier"
    )

    generation_interface = gr.Interface(
        fn=generate_mri,
        inputs=[
            gr.Dropdown(choices=severity_levels, label="Select Dementia Severity"),
            gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Number of Images to Generate")
        ],
        outputs=gr.Gallery(label="Generated MRI Scans", type="pil"),
        title="Generate MRI Based on Severity"
    )

    gr.TabbedInterface(
        interface_list=[classification_interface, generation_interface],
        tab_names=["Classify MRI", "Generate MRI"]
    ).launch(share=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R3GAN MRI UI")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the .pkl model file")
    args = parser.parse_args()
    main(args.model_path)
