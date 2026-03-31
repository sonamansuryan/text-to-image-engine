"""
run_all.py — Train + Evaluate pipeline
Usage: python run_all.py
       python run_all.py --config config.yaml
       python run_all.py --skip_train   # միայն evaluate
       python run_all.py --skip_eval    # միայն train
"""

import argparse
import logging
import os
import sys
import time
import torch

# ─────────────────────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Train + Evaluate pipeline")
    # Default: config.yaml նույն ֆոլդերում, ինչ run_all.py-ն
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--skip_train", action="store_true", help="Թրեյնինգը skip արա, անցիր evaluate-ի")
    parser.add_argument("--skip_eval",  action="store_true", help="Evaluate-ը skip արա")
    parser.add_argument("--eval_seed",  type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _banner(text: str, char: str = "═", width: int = 60):
    print(f"\n{char * width}")
    pad = (width - len(text) - 2) // 2
    print(f"{char}{' ' * pad}{text}{' ' * (width - pad - len(text) - 2)}{char}")
    print(f"{char * width}\n")


def _post_train_cooldown(seconds: int = 180):
    """Թրեյնինգ→evaluate-ի արանքում GPU-ն հանգստացնում է 3 րոպե։"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n⏳  Waiting {seconds // 60} min before evaluation (GPU cooldown)…")
    from tqdm import tqdm
    for _ in tqdm(range(seconds), desc="  Cooldown", unit="s",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s"):
        time.sleep(1)
    print("  ✅  Ready for evaluation\n")


# ─────────────────────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────────────────────

def run_train(config: dict):
    _banner("PHASE 1 — TRAINING")

    from src.utils import set_seed, setup_logging, get_gpu_memory_info
    from src.model import load_models, setup_lora, freeze_models
    from src.dataset import DiffusionDBDataset
    from src.trainer import LoRATrainer

    setup_logging(config["training"]["logging_dir"])
    logger = logging.getLogger(__name__)

    set_seed(config["training"]["seed"])
    logger.info(f"Seed: {config['training']['seed']}")
    logger.info(get_gpu_memory_info())

    # Models
    models = load_models(config)
    setup_lora(models["unet"], config)
    freeze_models(models["text_encoder"], models["vae"])

    # Dataset
    dataset = DiffusionDBDataset(
        config["data"]["dataset_path"],
        models["tokenizer"],
        config,
    )
    logger.info(f"Dataset size: {len(dataset)} examples")

    # Train
    trainer = LoRATrainer(config, models, dataset)
    train_start = time.time()
    trainer.train()
    train_time = (time.time() - train_start) / 60

    _banner(f"Training done in {train_time:.1f} min ✅")
    logger.info(f"Training complete — {train_time:.1f} min")

    del models, dataset, trainer
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────
#  EVALUATE
# ─────────────────────────────────────────────────────────────

def run_evaluate(lora_path: str, out_dir: str, seed: int = 42):
    _banner("PHASE 2 — EVALUATION")

    # evaluate.py-ն import անում ենք ուղղակիորեն
    import importlib.util, types

    eval_path = os.path.join(os.path.dirname(__file__), "evaluate.py")
    if not os.path.exists(eval_path):
        print(f"❌  evaluate.py չի գտնվել: {eval_path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("evaluate", eval_path)
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)

    os.makedirs(out_dir, exist_ok=True)

    # Load
    base_pipe, lora_pipe = eval_mod.load_pipelines(lora_path)

    # Generate
    print("\n[Generate] Base SD 1.5 images …")
    base_images = eval_mod.generate_images(base_pipe, eval_mod.EVAL_PROMPTS, seed)
    print("\n[Generate] LoRA fine-tuned images …")
    lora_images = eval_mod.generate_images(lora_pipe, eval_mod.EVAL_PROMPTS, seed)

    del base_pipe, lora_pipe
    torch.cuda.empty_cache()

    # CLIP scores
    import numpy as np
    print("\n[CLIP] Computing Base scores …")
    base_scores = eval_mod.compute_clip_scores(base_images, eval_mod.EVAL_PROMPTS)
    print("[CLIP] Computing LoRA scores …")
    lora_scores = eval_mod.compute_clip_scores(lora_images, eval_mod.EVAL_PROMPTS)

    # Print table
    print("\n" + "=" * 52)
    print(f"{'Prompt':<35} {'Base':>6} {'LoRA':>6} {'Diff':>6}")
    print("-" * 52)
    for i, prompt in enumerate(eval_mod.EVAL_PROMPTS):
        diff = lora_scores[i] - base_scores[i]
        sign = "+" if diff >= 0 else ""
        print(f"{prompt[:35]:<35} {base_scores[i]:>6.1f} "
              f"{lora_scores[i]:>6.1f} {sign}{diff:>5.1f}")
    print("-" * 52)
    avg_diff = np.mean(lora_scores) - np.mean(base_scores)
    print(f"{'AVERAGE':<35} {np.mean(base_scores):>6.1f} "
          f"{np.mean(lora_scores):>6.1f} {avg_diff:>+6.1f}")
    print("=" * 52)

    # Plots
    print("\n[Visual] Generating evaluation charts ...")
    eval_mod.plot_clip_scores(base_scores, lora_scores, eval_mod.EVAL_PROMPTS, out_dir)
    eval_mod.plot_radar(base_scores, lora_scores, out_dir)
    eval_mod.plot_before_after(base_images, lora_images, eval_mod.EVAL_PROMPTS,
                               base_scores, lora_scores, out_dir)

    _banner("Evaluation done")
    print(f"  Results saved to: {out_dir}")
    print(f"  eval_01_clip_scores.png   CLIP bar chart")
    print(f"  eval_02_radar.png         Radar profile")
    print(f"  eval_03_before_after.png  Side-by-side grid")
    print(f"  eval_03_dashboard.png    — full dashboard\n")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Config load
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    lora_path = config["training"]["output_dir"]
    eval_out  = os.path.join(lora_path, "eval")

    _banner("LoRA Fine-tuning Pipeline", char="★", width=60)
    print(f"  Config  : {args.config}")
    print(f"  Epochs  : {config['training']['num_train_epochs']}")
    print(f"  Cooldown: {config['training'].get('epoch_cooldown_seconds', 300) // 60} min between epochs")
    print(f"  Output  : {lora_path}\n")

    pipeline_start = time.time()

    # ── Phase 1: Train ──────────────────────────────────────
    if not args.skip_train:
        run_train(config)
        if not args.skip_eval:
            # 3 min cooling before eval
            _post_train_cooldown(seconds=180)
    else:
        print("⏩  Training skipped (--skip_train)\n")

    # ── Phase 2: Evaluate ───────────────────────────────────
    if not args.skip_eval:
        run_evaluate(lora_path, eval_out, seed=args.eval_seed)
    else:
        print("⏩  Evaluation skipped (--skip_eval)\n")

    total_time = (time.time() - pipeline_start) / 60
    _banner(f"All done in {total_time:.1f} min 🎉", char="★")


if __name__ == "__main__":
    main()