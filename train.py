"""
train.py — LoRA Fine-tuning Entry Point
Usage: python train.py --config configs/config.yaml
"""

import argparse
import logging

from src.utils import load_config, set_seed, setup_logging, get_gpu_memory_info
from src.model import load_models, setup_lora, freeze_models
from src.dataset import DiffusionDBDataset
from src.trainer import LoRATrainer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Stable Diffusion")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    setup_logging(config["training"]["logging_dir"])
    logger.info("=" * 60)
    logger.info("LoRA Fine-tuning — Stable Diffusion 1.5")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")

    # Seed
    set_seed(config["training"]["seed"])
    logger.info(f"Seed set to {config['training']['seed']}")

    # GPU info
    logger.info(get_gpu_memory_info())

    # Load models
    models = load_models(config)

    # Freeze VAE & text encoder
    freeze_models(models["text_encoder"], models["vae"])

    # Apply LoRA to UNet
    models["unet"] = setup_lora(models["unet"], config)

    # Dataset
    logger.info("Loading dataset...")
    dataset = DiffusionDBDataset(
        dataset_path=config["data"]["dataset_path"],
        tokenizer=models["tokenizer"],
        config=config,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    # Train
    trainer = LoRATrainer(config=config, models=models, dataset=dataset)
    trainer.train()


if __name__ == "__main__":
    main()