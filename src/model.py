import torch
import logging
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


def load_models(config: dict) -> dict:
    """
    Load all Stable Diffusion components.
    Returns dict with all model parts.
    """
    model_name = config["model"]["pretrained_model_name"]
    revision = config["model"]["revision"]

    logger.info(f"Loading models from {model_name}")

    tokenizer = CLIPTokenizer.from_pretrained(
        model_name, subfolder="tokenizer", revision=revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_name, subfolder="text_encoder", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(
        model_name, subfolder="vae", revision=revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_name, subfolder="unet", revision=revision
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_name, subfolder="scheduler"
    )

    logger.info("All models loaded successfully")
    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "noise_scheduler": noise_scheduler,
    }


def setup_lora(unet: UNet2DConditionModel, config: dict) -> UNet2DConditionModel:
    """
    Apply LoRA adapters to UNet.
    """
    lora_cfg = config["lora"]

    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    logger.info("LoRA adapters applied to UNet")
    return unet


def freeze_models(text_encoder: CLIPTextModel, vae: AutoencoderKL):
    """
    Freeze text encoder and VAE — only UNet LoRA trains.
    """
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    logger.info("VAE and text encoder frozen")


def save_lora_weights(unet: UNet2DConditionModel, output_dir: str, step: int = None):
    """
    Save LoRA weights to disk.
    """
    import os
    save_path = output_dir
    if step is not None:
        save_path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(save_path, exist_ok=True)
    unet.save_pretrained(save_path)
    logger.info(f"LoRA weights saved to {save_path}")