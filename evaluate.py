"""
evaluate.py — LoRA Model Evaluation (simplified)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metrics
  ① CLIP Score     — prompt-image alignment
  ② LAION Aesthetic — visual quality score 1-10

Charts
  eval_01_clip_scores.png     CLIP bar chart
  eval_02_radar.png           CLIP radar profile
  eval_03_before_after.png    Side-by-side grid (CLIP + AES badges)
  eval_04_delta.png           CLIP per-prompt delta
  eval_05_aesthetic.png       Aesthetic scores
  eval_06_dashboard.png       Summary dashboard

Usage:
  python evaluate.py
  python evaluate.py --lora_path ./outputs --out_dir ./outputs/eval_v3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)

import os, argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDIMScheduler
from peft import PeftModel
import open_clip
from huggingface_hub import hf_hub_download


# ═══════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════

EVAL_PROMPTS = [
    "a little fox reading a glowing book under a mushroom in a rainy forest, "
    "soft watercolor style, warm lantern light, cozy, magical, detailed illustration",

    "a lone samurai standing in a bamboo forest in heavy rain, "
    "dramatic side lighting, cinematic composition, detailed, moody atmosphere, film grain",

    "a young girl tending a rooftop garden above a misty city at dawn, "
    "soft morning light, Ghibli style, peaceful, detailed, warm pastel colors",

    "a small cat in a sailor outfit steering a wooden boat on a calm ocean, "
    "soft watercolor style, warm sunset colors, whimsical, cozy, detailed illustration",

    "smart robot gardener tending flowers, Studio Ghibli style, soft sunlight, watercolor texture, 4K",

    "a bear in a knitted sweater painting a self-portrait in an autumn forest studio, "
    "fallen leaves everywhere, soft diffused light, cozy impressionist style, heartwarming",
]

PROMPT_LABELS = [
    "Fox & Book",
    "Alone Samurai",
    "Rooftop Garden",
    "Cat Sailor",
    "Robot Gardener",
    "Bear Artist",
]


# ═══════════════════════════════════════════════════════════════
# PALETTE
# ═══════════════════════════════════════════════════════════════

BG         = "#FFFFFF"
CARD       = "#FFFFFF"
BORDER     = "#F0D9C8"
STRIPE     = "#FDF8F3"
BASE_COLOR = "#FFB703"
LORA_COLOR = "#D62828"
POSITIVE   = "#FB8500"
NEGATIVE   = "#6A040F"
TEXT_DARK  = "#1A0A00"
TEXT_MID   = "#6B3A2A"
TEXT_LIGHT = "#C07A5A"
GRID_COLOR = "#FAE8D8"


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    CARD,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT_MID,
        "axes.titlecolor":   TEXT_DARK,
        "xtick.color":       TEXT_LIGHT,
        "ytick.color":       TEXT_LIGHT,
        "text.color":        TEXT_DARK,
        "grid.color":        GRID_COLOR,
        "grid.linewidth":    0.8,
        "font.family":       "sans-serif",
        "figure.dpi":        140,
    })


def _style_ax(ax, xgrid=False):
    ax.set_facecolor(CARD)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.yaxis.grid(True, color=GRID_COLOR, linestyle="-", zorder=0)
    if xgrid:
        ax.xaxis.grid(True, color=GRID_COLOR, linestyle="-", zorder=0)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — LOAD PIPELINES
# ═══════════════════════════════════════════════════════════════

def load_pipelines(lora_path: str):
    print("[Setup] Loading Base SD 1.5 ...")
    base_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    base_pipe.scheduler = DDIMScheduler.from_config(base_pipe.scheduler.config)
    base_pipe.enable_attention_slicing()

    print("[Setup] Loading LoRA pipeline ...")
    lora_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    lora_pipe.scheduler = DDIMScheduler.from_config(lora_pipe.scheduler.config)
    lora_pipe.unet = PeftModel.from_pretrained(lora_pipe.unet, lora_path)
    lora_pipe.enable_attention_slicing()

    print("[Setup] Both pipelines ready.")
    return base_pipe, lora_pipe


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — GENERATE IMAGES
# ═══════════════════════════════════════════════════════════════

def generate_images(pipe, prompts: list, seed: int = 42) -> list:
    images = []
    for prompt in tqdm(prompts, desc="Generating"):
        gen = torch.Generator("cuda").manual_seed(seed)
        with torch.autocast("cuda"):
            result = pipe(prompt, num_inference_steps=35, guidance_scale=8.0,
                          height=512, width=512, generator=gen)
        images.append(result.images[0])
    return images


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — CLIP SCORE
# ═══════════════════════════════════════════════════════════════

def compute_clip_scores(images: list, prompts: list, clip_model, clip_prep, clip_tok) -> list:
    scores = []
    with torch.no_grad():
        for img, prompt in zip(images, prompts):
            img_t = clip_prep(img).unsqueeze(0).to("cuda")
            img_f = clip_model.encode_image(img_t)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_t = clip_tok([prompt]).to("cuda")
            txt_f = clip_model.encode_text(txt_t)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            scores.append((img_f @ txt_f.T).item() * 100)
    return scores


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — AESTHETIC SCORE
# ═══════════════════════════════════════════════════════════════

class _AestheticMLP(nn.Module):
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),  # 0
            nn.ReLU(),                     # 1
            nn.Linear(1024, 128),          # 2
            nn.ReLU(),                     # 3
            nn.Linear(128, 64),            # 4
            nn.ReLU(),                     # 5
            nn.Linear(64, 16),             # 6
            nn.Linear(16, 1),              # 7
        )

    def forward(self, x):
        return self.layers(x)


def compute_aesthetic_scores(images: list, clip_model, clip_prep, mlp) -> list:
    scores = []
    with torch.no_grad():
        for img in tqdm(images, desc="Aesthetic"):
            img_t = clip_prep(img).unsqueeze(0).to("cuda")
            emb   = clip_model.encode_image(img_t)
            emb   = emb / emb.norm(dim=-1, keepdim=True)
            scores.append(mlp(emb.float()).item())
    return scores


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════

def _paired_bar(ax, base_vals, lora_vals, ylim_pad=7):
    x = np.arange(len(base_vals))
    w = 0.36
    bars_b = ax.bar(x - w/2, base_vals, w, color=BASE_COLOR,
                    alpha=0.88, edgecolor=CARD, linewidth=1.1, zorder=3)
    bars_l = ax.bar(x + w/2, lora_vals, w, color=LORA_COLOR,
                    alpha=0.88, edgecolor=CARD, linewidth=1.1, zorder=3)
    for bar, val, c in [*zip(bars_b, base_vals, [BASE_COLOR]*len(base_vals)),
                         *zip(bars_l, lora_vals, [LORA_COLOR]*len(lora_vals))]:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(max(base_vals), max(lora_vals)) * 0.012,
                f"{val:.2f}", ha="center", fontsize=7.5,
                color=c, fontweight="700", zorder=4)
    ax.axhline(np.mean(base_vals), color=BASE_COLOR, linestyle="--",
               alpha=0.4, linewidth=1.5, zorder=2)
    ax.axhline(np.mean(lora_vals), color=LORA_COLOR, linestyle="--",
               alpha=0.4, linewidth=1.5, zorder=2)
    ax.set_ylim(0, max(max(base_vals), max(lora_vals)) + ylim_pad)
    ax.set_xticks(np.arange(len(base_vals)))
    ax.set_xticklabels(PROMPT_LABELS, fontsize=8.5)


def _legend(ax, base_label, lora_label):
    ax.legend(handles=[
        mpatches.Patch(color=BASE_COLOR, alpha=0.88, label=base_label),
        mpatches.Patch(color=LORA_COLOR, alpha=0.88, label=lora_label),
    ], fontsize=9, loc="upper right", framealpha=0.95, edgecolor=BORDER)


def plot_clip_scores(base_scores, lora_scores, out_dir):
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG); _style_ax(ax)
    _paired_bar(ax, base_scores, lora_scores, ylim_pad=7)
    diff = np.mean(lora_scores) - np.mean(base_scores)
    sign = "+" if diff >= 0 else ""
    _legend(ax, f"Base  avg: {np.mean(base_scores):.1f}",
            f"LoRA  avg: {np.mean(lora_scores):.1f}  ({sign}{diff:.1f})")
    ax.set_title("CLIP Score  —  Base SD 1.5  vs  LoRA Fine-tuned",
                 fontsize=13, fontweight="bold", pad=16)
    ax.set_ylabel("CLIP Score", fontsize=10, labelpad=8)
    plt.tight_layout(pad=2)
    out = os.path.join(out_dir, "eval_01_clip_scores.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  Saved → {out}")


def plot_radar(base_scores, lora_scores, out_dir):
    _apply_style()
    N      = len(PROMPT_LABELS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    b_vals = base_scores + [base_scores[0]]
    l_vals = lora_scores + [lora_scores[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
    ax.plot(angles, b_vals, color=BASE_COLOR, linewidth=2.4)
    ax.fill(angles, b_vals, color=BASE_COLOR, alpha=0.10)
    ax.plot(angles, l_vals, color=LORA_COLOR, linewidth=2.4)
    ax.fill(angles, l_vals, color=LORA_COLOR, alpha=0.10)
    ax.set_ylim(min(min(base_scores), min(lora_scores)) - 4,
                max(max(base_scores), max(lora_scores)) + 4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PROMPT_LABELS, fontsize=9.5, color=TEXT_DARK)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color(BORDER)
    ax.grid(color=GRID_COLOR, linewidth=0.9)
    ax.set_title("CLIP Score Radar", fontsize=12, fontweight="bold", pad=22)
    ax.legend(["Base SD 1.5", "LoRA Fine-tuned"],
              loc="upper right", bbox_to_anchor=(1.35, 1.14),
              fontsize=9.5, framealpha=0.95, edgecolor=BORDER)
    plt.tight_layout()
    out = os.path.join(out_dir, "eval_02_radar.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  Saved → {out}")


def plot_before_after(base_images, lora_images, prompts,
                      base_scores, lora_scores,
                      base_aes, lora_aes, out_dir):
    _apply_style()
    n = len(prompts)
    HEADER = 0.13
    height_ratios = []
    for _ in range(n):
        height_ratios += [HEADER, 1.0]

    fig = plt.figure(figsize=(14, 26))
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.993,
             "Model Evaluation  —  Base SD 1.5  ↔  LoRA Fine-tuned",
             ha="center", va="top",
             fontsize=15, fontweight="bold", color=TEXT_DARK)

    LEFT, RIGHT = 0.06, 0.97
    MID = (LEFT + RIGHT) / 2
    fig.text(LEFT + (MID - LEFT)/2, 0.974, "BASE  SD 1.5",
             ha="center", va="top", fontsize=11, fontweight="bold", color=BASE_COLOR)
    fig.text(MID + (RIGHT - MID)/2, 0.974, "LoRA  FINE-TUNED",
             ha="center", va="top", fontsize=11, fontweight="bold", color=LORA_COLOR)

    gs = gridspec.GridSpec(n * 2, 2, figure=fig,
                           height_ratios=height_ratios,
                           hspace=0.0, wspace=0.030,
                           top=0.958, bottom=0.008,
                           left=LEFT, right=RIGHT)

    for i in range(n):
        clip_diff = lora_scores[i] - base_scores[i]
        aes_diff  = lora_aes[i]    - base_aes[i]
        dc        = POSITIVE if clip_diff >= 0 else NEGATIVE
        c_sign    = "+" if clip_diff >= 0 else ""
        a_sign    = "+" if aes_diff  >= 0 else ""
        bg_stripe = STRIPE if i % 2 == 0 else CARD

        ax_h = fig.add_subplot(gs[i * 2, :])
        ax_h.set_facecolor(bg_stripe); ax_h.axis("off")
        ax_h.text(0.012, 0.50, f"#{i+1}", ha="left", va="center",
                  fontsize=9, fontweight="bold", color=TEXT_LIGHT,
                  transform=ax_h.transAxes)
        ax_h.text(0.040, 0.50, PROMPT_LABELS[i], ha="left", va="center",
                  fontsize=10, fontweight="bold", color=TEXT_DARK,
                  transform=ax_h.transAxes)
        short = prompts[i].split(",")[0].strip()
        ax_h.text(0.24, 0.50, f"\u201c{short}\u201d", ha="left", va="center",
                  fontsize=8.5, style="italic", color=TEXT_MID,
                  transform=ax_h.transAxes)
        ax_h.text(0.988, 0.50,
                  f"CLIP \u0394 {c_sign}{clip_diff:.1f}   AES \u0394 {a_sign}{aes_diff:.2f}",
                  ha="right", va="center",
                  fontsize=9, fontweight="bold", color=dc,
                  transform=ax_h.transAxes)

        ax_b = fig.add_subplot(gs[i * 2 + 1, 0])
        ax_b.imshow(np.array(base_images[i])); ax_b.axis("off")
        ax_b.text(0.025, 0.040,
                  f"  CLIP {base_scores[i]:.1f}  |  AES {base_aes[i]:.2f}  ",
                  transform=ax_b.transAxes,
                  fontsize=8.5, fontweight="bold", color="white", va="bottom",
                  bbox=dict(facecolor=BASE_COLOR, alpha=0.85,
                            boxstyle="round,pad=0.28", edgecolor="none"), zorder=10)
        for sp in ax_b.spines.values():
            sp.set_visible(True); sp.set_linewidth(1.8); sp.set_edgecolor(BASE_COLOR)

        ax_l = fig.add_subplot(gs[i * 2 + 1, 1])
        ax_l.imshow(np.array(lora_images[i])); ax_l.axis("off")
        ax_l.text(0.025, 0.040,
                  f"  CLIP {lora_scores[i]:.1f} ({c_sign}{clip_diff:.1f})"
                  f"  |  AES {lora_aes[i]:.2f}  ",
                  transform=ax_l.transAxes,
                  fontsize=8.5, fontweight="bold", color="white", va="bottom",
                  bbox=dict(facecolor=dc, alpha=0.85,
                            boxstyle="round,pad=0.28", edgecolor="none"), zorder=10)
        for sp in ax_l.spines.values():
            sp.set_visible(True); sp.set_linewidth(1.8); sp.set_edgecolor(LORA_COLOR)

    out = os.path.join(out_dir, "eval_03_before_after.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  Saved → {out}")


def plot_delta(base_scores, lora_scores, out_dir):
    _apply_style()
    deltas    = [l - b for l, b in zip(lora_scores, base_scores)]
    avg_delta = float(np.mean(deltas))
    colors    = [POSITIVE if d >= 0 else NEGATIVE for d in deltas]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(BG); _style_ax(ax, xgrid=True)
    ax.yaxis.grid(False)

    y = np.arange(len(PROMPT_LABELS))
    ax.barh(y, deltas, color=colors, alpha=0.88,
            height=0.52, edgecolor=CARD, linewidth=0.8, zorder=3)
    for i, (d, c) in enumerate(zip(deltas, colors)):
        s = "+" if d >= 0 else ""
        ax.text(d + (0.05 if d >= 0 else -0.05), i, f"{s}{d:.2f}",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=9, fontweight="bold", color=c)
    ax.axvline(0, color=BORDER, linewidth=1.8, zorder=2)
    avg_c = POSITIVE if avg_delta >= 0 else NEGATIVE
    ax.axvline(avg_delta, color=avg_c, linewidth=1.8,
               linestyle="--", alpha=0.75, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(PROMPT_LABELS, fontsize=10.5)
    ax.set_xlabel("CLIP Δ  (LoRA − Base)", fontsize=10, labelpad=9)
    ax.set_title("Per-Prompt CLIP Delta", fontsize=13, fontweight="bold", pad=16)
    ax.invert_yaxis()
    plt.tight_layout(pad=2)
    out = os.path.join(out_dir, "eval_04_delta.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  Saved → {out}")


def plot_aesthetic(base_aes, lora_aes, out_dir):
    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG); _style_ax(ax)
    _paired_bar(ax, base_aes, lora_aes, ylim_pad=1.5)
    diff = np.mean(lora_aes) - np.mean(base_aes)
    sign = "+" if diff >= 0 else ""
    _legend(ax, f"Base  avg: {np.mean(base_aes):.2f}",
            f"LoRA  avg: {np.mean(lora_aes):.2f}  ({sign}{diff:.2f})")
    ax.set_title("LAION Aesthetic Score  —  Base SD 1.5  vs  LoRA Fine-tuned",
                 fontsize=13, fontweight="bold", pad=16)
    ax.set_ylabel("Aesthetic Score (1-10)", fontsize=10, labelpad=8)
    plt.tight_layout(pad=2)
    out = os.path.join(out_dir, "eval_05_aesthetic.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  Saved → {out}")


def plot_dashboard(base_clip, lora_clip, base_aes, lora_aes, out_dir):
    _apply_style()
    wins = sum(l > b for l, b in zip(lora_clip, base_clip))

    panels = [
        ("CLIP Score",      np.mean(base_clip), np.mean(lora_clip), True,  "",           ".1f"),
        ("Aesthetic Score", np.mean(base_aes),  np.mean(lora_aes),  True,  "1-10 scale", ".2f"),
        ("CLIP Win Rate",   50.0,
         wins / len(lora_clip) * 100,
         True, f"{wins}/{len(lora_clip)} prompts", ".0f"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Summary  —  Base SD 1.5  vs  LoRA Fine-tuned",
                 fontsize=14, fontweight="bold", y=1.02, color=TEXT_DARK)

    fig.legend(handles=[
        mpatches.Patch(color=BASE_COLOR, alpha=0.88, label="Base SD 1.5"),
        mpatches.Patch(color=LORA_COLOR, alpha=0.88, label="LoRA Fine-tuned"),
    ], loc="upper right", bbox_to_anchor=(0.99, 1.0),
    fontsize=9.5, framealpha=0.95, edgecolor=BORDER)

    for ax, (title, base_val, lora_val, higher_better, note, fmt) in zip(axes, panels):
        _style_ax(ax)
        bars = ax.bar(["Base", "LoRA"], [base_val, lora_val],
                      color=[BASE_COLOR, LORA_COLOR],
                      alpha=0.88, edgecolor=CARD, linewidth=1.1,
                      width=0.45, zorder=3)
        top = max(base_val, lora_val) * 1.28 if max(base_val, lora_val) > 0 else 10
        ax.set_ylim(0, top)
        for bar, v in zip(bars, [base_val, lora_val]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + top * 0.02,
                    f"{v:{fmt}}", ha="center", fontsize=10.5, fontweight="bold",
                    color=bar.get_facecolor(), zorder=4)
        diff  = lora_val - base_val
        sign  = "+" if diff >= 0 else ""
        good  = (diff >= 0) == higher_better
        d_col = POSITIVE if good else NEGATIVE
        ax.text(0.97, 0.97, f"Δ {sign}{diff:{fmt}}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, fontweight="bold", color=d_col)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        if note:
            ax.set_xlabel(note, fontsize=8, color=TEXT_LIGHT, labelpad=4)
        ax.set_xticklabels(["Base", "LoRA"], fontsize=10)

    plt.tight_layout()
    out = os.path.join(out_dir, "eval_06_dashboard.png")
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  Saved → {out}")


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — MAIN
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str, default="./outputs")
    parser.add_argument("--out_dir",   type=str, default="./outputs/eval_v3")
    parser.add_argument("--seed",      type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 55)
    print("  Model Evaluation  —  Base SD 1.5  vs  LoRA")
    print("  Metrics: CLIP · Aesthetic")
    print("=" * 55)

    # ── Generate images ───────────────────────────────────────
    base_pipe, lora_pipe = load_pipelines(args.lora_path)

    print("\n[Generate] Base images ...")
    base_images = generate_images(base_pipe, EVAL_PROMPTS, args.seed)
    print("[Generate] LoRA images ...")
    lora_images = generate_images(lora_pipe, EVAL_PROMPTS, args.seed)

    del base_pipe, lora_pipe
    torch.cuda.empty_cache()

    # ── CLIP (load once, use for both CLIP score + Aesthetic) ─
    print("\n[CLIP] Loading ViT-B/32 for CLIP scores ...")
    clip_b32, clip_b32_prep, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    clip_b32_tok = open_clip.get_tokenizer("ViT-B-32")
    clip_b32 = clip_b32.to("cuda").eval()

    print("[CLIP] Base ...")
    base_clip = compute_clip_scores(base_images, EVAL_PROMPTS,
                                    clip_b32, clip_b32_prep, clip_b32_tok)
    print("[CLIP] LoRA ...")
    lora_clip = compute_clip_scores(lora_images, EVAL_PROMPTS,
                                    clip_b32, clip_b32_prep, clip_b32_tok)

    del clip_b32
    torch.cuda.empty_cache()

    # ── Aesthetic (ViT-L/14 + MLP, load once) ────────────────
    print("\n[Aesthetic] Loading CLIP ViT-L/14 ...")
    clip_l14, clip_l14_prep, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai")
    clip_l14 = clip_l14.to("cuda").eval()

    print("[Aesthetic] Downloading MLP weights ...")
    weights_path = hf_hub_download(
        repo_id="camenduru/improved-aesthetic-predictor",
        filename="sac+logos+ava1-l14-linearMSE.pth",
    )
    mlp = _AestheticMLP(input_size=768).to("cuda").eval()
    mlp.load_state_dict(torch.load(weights_path, map_location="cuda"))

    print("[Aesthetic] Base ...")
    base_aes = compute_aesthetic_scores(base_images, clip_l14, clip_l14_prep, mlp)
    print("[Aesthetic] LoRA ...")
    lora_aes = compute_aesthetic_scores(lora_images, clip_l14, clip_l14_prep, mlp)

    del clip_l14, mlp
    torch.cuda.empty_cache()

    # ── Console summary ───────────────────────────────────────
    W = 55
    print("\n" + "=" * W)
    print(f"  {'Metric':<22} {'Base':>8} {'LoRA':>8} {'Δ':>8}")
    print("  " + "─" * (W - 2))

    def _row(label, bvals, lvals, fmt):
        bv, lv = np.mean(bvals), np.mean(lvals)
        d = lv - bv; s = "+" if d >= 0 else ""
        print(f"  {label:<22} {bv:>8{fmt}} {lv:>8{fmt}} {s}{d:>8{fmt}}")

    _row("CLIP Score",       base_clip, lora_clip, ".1f")
    _row("Aesthetic (1-10)", base_aes,  lora_aes,  ".2f")
    wins = sum(l > b for l, b in zip(lora_clip, base_clip))
    print(f"  {'CLIP Win Rate':<22} {'':>8} {wins}/{len(lora_clip):>3} prompts LoRA wins")
    print("=" * W)

    # ── Charts ────────────────────────────────────────────────
    print("\n[Visual] Generating charts ...")
    plot_clip_scores(base_clip, lora_clip, args.out_dir)
    plot_radar(base_clip, lora_clip, args.out_dir)
    plot_before_after(base_images, lora_images, EVAL_PROMPTS,
                      base_clip, lora_clip, base_aes, lora_aes, args.out_dir)
    plot_delta(base_clip, lora_clip, args.out_dir)
    plot_aesthetic(base_aes, lora_aes, args.out_dir)
    plot_dashboard(base_clip, lora_clip, base_aes, lora_aes, args.out_dir)

    print("\nEvaluation complete!")
    print(f"  Results → {args.out_dir}/")
    for name, desc in [
        ("eval_01_clip_scores.png", "CLIP bar chart"),
        ("eval_02_radar.png",       "CLIP radar profile"),
        ("eval_03_before_after.png","Side-by-side grid (CLIP + AES badges)"),
        ("eval_04_delta.png",       "CLIP per-prompt delta"),
        ("eval_05_aesthetic.png",   "Aesthetic scores"),
        ("eval_06_dashboard.png",   "Summary dashboard"),
    ]:
        print(f"  {name:<36} {desc}")


if __name__ == "__main__":
    main()