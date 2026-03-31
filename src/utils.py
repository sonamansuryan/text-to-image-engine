import os
import random
import logging
import numpy as np
import torch
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir: str, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "train.log")),
        ],
    )


def get_gpu_memory_info() -> str:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        return f"GPU Memory: {allocated:.1f}GB allocated / {reserved:.1f}GB reserved / {total:.1f}GB total"
    return "CUDA not available"


def count_parameters(model) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return f"Trainable: {trainable:,} / Total: {total:,} ({100 * trainable / total:.2f}%)"