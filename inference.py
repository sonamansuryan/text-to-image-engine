"""
inference.py — Generate images using trained LoRA weights
Usage: python inference.py --prompt "your prompt here" --lora_path ./outputs
"""

import argparse
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import os
import warnings
import logging
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with LoRA")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--lora_path", type=str, default="./outputs", help="Path to LoRA weights")
    parser.add_argument("--output_path", type=str, default="./outputs/generated.png")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    )

    # Load LoRA weights
    print(f"Loading LoRA from {args.lora_path}")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora_path)
    pipe = pipe.to("cuda")

    # Generate
    generator = torch.Generator("cuda").manual_seed(args.seed)
    print(f"Generating: '{args.prompt}'")

    images = pipe(
        prompt=args.prompt,
        num_images_per_prompt=args.num_images,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    for i, image in enumerate(images):
        path = args.output_path.replace(".png", f"_{i}.png") if args.num_images > 1 else args.output_path
        image.save(path)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()