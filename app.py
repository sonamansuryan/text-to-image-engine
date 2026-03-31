import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

def generate(prompt, negative_prompt, steps, guidance, width, height):
    image = pipe(
        prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        width=int(width),
        height=int(height),
    ).images[0]
    return image

css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500&family=Inter:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #f5f0eb !important;
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 15px 40px !important;
}

/* Header */
.header-wrap {
    margin-bottom: 25px;
    padding-bottom: 18px;
    border-bottom: 1.5px solid #d9cfc4;
}

.header-wrap h1 {
    font-family: 'Playfair Display', serif;
    font-size: 34px;
    color: #2c2416;
    margin: 0 16px 0 0;
    display: inline;
}

#left-panel {
    background: #fdfaf7 !important;
    border: 1.5px solid #e0d6cc !important;
    border-radius: 14px !important;
    padding: 26px !important;
    box-shadow: 0 2px 12px rgba(80,50,20,0.05) !important;
}

#right-panel {
    background: #fdfaf7 !important;
    border: 1.5px solid #e0d6cc !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    display: flex !important;
    align-items: stretch !important;
}

#right-panel .image-container, 
#right-panel img,
#right-panel .preview {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    padding: 0 !important;
    margin: 0 !important;
}

textarea, input[type="text"] {
    background: #f7f2ed !important;
    border: 1.5px solid #ddd5c9 !important;
    border-radius: 10px !important;
    font-style: italic !important;
}

textarea::placeholder {
    font-style: italic !important;
    color: #c4b8aa !important;
}

#generate-btn {
    background: #c07a3a !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    padding: 15px !important;
    cursor: pointer;
    box-shadow: 0 3px 14px rgba(192,122,58,0.28) !important;
}

.divider { border-top: 1.5px solid #ede5dc !important; margin: 18px 0 !important; }
footer { display: none !important; }

.download-div, .share-div, .fullscreen-button, .icon-buttons {
    display: none !important;
}
"""

with gr.Blocks(title="Lumina — Text to Image") as app:

    gr.HTML("""
        <div class="header-wrap">
            <h1>Lumina</h1>
            <span style="font-size: 11px; color: #a09282; text-transform: uppercase; letter-spacing: 0.17em;">
                Fine-tuned Stable Diffusion &nbsp;·&nbsp; LoRA &nbsp;·&nbsp; DiffusionDB
            </span>
        </div>
    """)

    with gr.Row(equal_height=True):

        with gr.Column(scale=5, elem_id="left-panel"):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="a fantasy castle on a mountain...",
                lines=4
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="blurry, low quality...",
                lines=2
            )

            gr.HTML('<hr class="divider">')

            with gr.Row():
                steps = gr.Slider(minimum=10, maximum=50, value=30, label="Steps")
                guidance = gr.Slider(minimum=1, maximum=15, value=7.5, label="Guidance Scale")

            with gr.Row():
                width = gr.Dropdown(choices=["512", "640", "768"], value="512", label="Width")
                height = gr.Dropdown(choices=["512", "640", "768"], value="512", label="Height")

            gr.HTML('<hr class="divider">')

            generate_btn = gr.Button("Generate Image", elem_id="generate-btn")

        with gr.Column(scale=6, elem_id="right-panel"):
            output = gr.Image(show_label=False, interactive=False)

    generate_btn.click(
        fn=generate,
        inputs=[prompt, negative_prompt, steps, guidance, width, height],
        outputs=output,
    )

app.launch(css=css)