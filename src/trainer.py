import os
import time
import torch
import logging
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from accelerate import Accelerator

from src.model import save_lora_weights

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  GPU Temperature helpers
# ─────────────────────────────────────────────────────────────

def _get_gpu_temp() -> int | None:
    """Returns GPU temperature in °C, or None if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def cooling_pause(cooldown_seconds: int, temp_limit: int = 83):
    """
    Epoch-ի վերջում GPU-ն հանգստացնում է։
    - Սպասում է cooldown_seconds-ը
    - Եթե GPU temp > temp_limit, սպասում է մինչև temp_limit-10-ից ցածր
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    temp = _get_gpu_temp()
    temp_str = f" (GPU: {temp}°C)" if temp is not None else ""
    logger.info(f"⏸  Epoch ավարտ{temp_str} — սկսում է {cooldown_seconds // 60} րոպե cooling pause…")
    print(f"\n{'─'*55}")
    print(f"  ⏸  Cooling pause — {cooldown_seconds // 60} րոպե{temp_str}")
    print(f"{'─'*55}")

    # Base cooldown — progress bar-ով
    for remaining in tqdm(range(cooldown_seconds, 0, -1),
                          desc="  Cooling", unit="s",
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s"):
        time.sleep(1)

    # Եթե GPU-ն դեռ տաք է, շարունակել սպասել
    if temp is not None and temp > temp_limit:
        target = temp_limit - 10
        print(f"  🌡  GPU {temp}°C > {temp_limit}°C limit, սպասում ենք մինչև {target}°C…")
        while True:
            time.sleep(15)
            current = _get_gpu_temp()
            if current is None or current <= target:
                break
            print(f"      GPU temp: {current}°C, target: {target}°C…")

    final_temp = _get_gpu_temp()
    temp_str = f" (GPU: {final_temp}°C)" if final_temp is not None else ""
    print(f"  ✅  Cooling ավարտ{temp_str} — շարունակում ենք թրեյնինգը\n")
    logger.info(f"Cooling pause ավարտ{temp_str}")


# ─────────────────────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────────────────────

class LoRATrainer:
    """
    Main trainer class for LoRA fine-tuning of Stable Diffusion.
    Ավելացված՝ epoch-ների արանքում cooling pause + GPU temp monitoring.
    """

    def __init__(self, config: dict, models: dict, dataset):
        self.config = config
        self.models = models
        self.dataset = dataset
        self.train_cfg = config["training"]

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.train_cfg["gradient_accumulation_steps"],
            mixed_precision=self.train_cfg["mixed_precision"],
            log_with="tensorboard",
            project_dir=self.train_cfg["logging_dir"],
        )

        self.writer = SummaryWriter(log_dir=self.train_cfg["logging_dir"])
        self.global_step = 0

        # Cooling config
        self.cooldown_seconds = self.train_cfg.get("epoch_cooldown_seconds", 300)
        self.gpu_temp_limit = self.train_cfg.get("gpu_temp_limit", 83)

    def _build_optimizer(self):
        opt_cfg = self.config["optimizer"]
        unet = self.models["unet"]
        trainable_params = [p for p in unet.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.train_cfg["learning_rate"],
            betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
            weight_decay=opt_cfg["weight_decay"],
            eps=opt_cfg["epsilon"],
        )
        return optimizer

    def _build_dataloader(self):
        from src.dataset import collate_fn
        return DataLoader(
            self.dataset,
            batch_size=self.train_cfg["train_batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

    def _get_lr_scheduler(self, optimizer, num_training_steps: int):
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_steps = self.train_cfg["lr_warmup_steps"]
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=num_training_steps - warmup_steps)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    def _log_gpu_stats(self):
        """Loggerում GPU stats-ը։"""
        temp = _get_gpu_temp()
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            temp_str = f", Temp: {temp}°C" if temp is not None else ""
            logger.info(f"GPU — VRAM: {allocated:.1f}GB alloc / {reserved:.1f}GB reserved{temp_str}")

    def train(self):
        unet = self.models["unet"]
        vae = self.models["vae"]
        text_encoder = self.models["text_encoder"]
        noise_scheduler = self.models["noise_scheduler"]

        if self.train_cfg["gradient_checkpointing"]:
            unet.enable_gradient_checkpointing()

        optimizer = self._build_optimizer()
        dataloader = self._build_dataloader()

        num_epochs = self.train_cfg["num_train_epochs"]
        num_training_steps = num_epochs * len(dataloader)
        lr_scheduler = self._get_lr_scheduler(optimizer, num_training_steps)

        unet, optimizer, dataloader, lr_scheduler = self.accelerator.prepare(
            unet, optimizer, dataloader, lr_scheduler
        )

        # Resume from checkpoint if exists
        resume_step = 0
        checkpoint_dir = self.train_cfg["output_dir"]
        if os.path.isdir(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
                resume_step = int(latest.split("-")[1])
                self.global_step = resume_step
                from peft import set_peft_model_state_dict
                import safetensors.torch
                ckpt_path = os.path.join(checkpoint_dir, latest, "adapter_model.safetensors")
                state_dict = safetensors.torch.load_file(ckpt_path)
                set_peft_model_state_dict(unet, state_dict)
                logger.info(f"Resumed from checkpoint: {latest} (step {resume_step})")

        vae.to(self.accelerator.device, dtype=torch.float32)
        text_encoder.to(self.accelerator.device)

        logger.info(f"Starting training — {num_epochs} epochs total")
        logger.info(f"Total steps: {num_training_steps}")
        logger.info(f"Epoch cooldown: {self.cooldown_seconds}s, GPU temp limit: {self.gpu_temp_limit}°C")
        self._log_gpu_stats()

        for epoch in range(num_epochs):
            epoch_start = time.time()
            unet.train()
            epoch_loss = 0.0

            print(f"\n{'═'*55}")
            print(f"  🚀  Epoch {epoch + 1}/{num_epochs}")
            temp = _get_gpu_temp()
            if temp:
                print(f"  🌡  GPU temp before epoch: {temp}°C")
            print(f"{'═'*55}")

            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )

            for step, batch in enumerate(progress_bar):
                if step < resume_step % len(dataloader):
                    continue
                with self.accelerator.accumulate(unet):
                    with torch.no_grad():
                        latents = vae.encode(
                            batch["pixel_values"].to(device=self.accelerator.device, dtype=torch.float32)
                        ).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (bsz,), device=latents.device
                    ).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    loss = torch.nn.functional.mse_loss(
                        model_pred.float(), noise.float(), reduction="mean"
                    )

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            unet.parameters(), self.train_cfg["max_grad_norm"]
                        )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.detach().item()
                self.global_step += 1

                if self.global_step % self.train_cfg["logging_steps"] == 0:
                    avg_loss = epoch_loss / (step + 1)
                    self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                    self.writer.add_scalar(
                        "train/lr", lr_scheduler.get_last_lr()[0], self.global_step
                    )
                    # GPU temp logging
                    temp = _get_gpu_temp()
                    if temp is not None:
                        self.writer.add_scalar("system/gpu_temp", temp, self.global_step)
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "gpu": f"{temp}°C" if temp else "N/A"
                    })

                if self.global_step % self.train_cfg["save_steps"] == 0:
                    if self.accelerator.is_main_process:
                        save_lora_weights(
                            self.accelerator.unwrap_model(unet),
                            self.train_cfg["output_dir"],
                            step=self.global_step,
                        )

            # ── Epoch ավարտ ────────────────────────────────────
            epoch_time = (time.time() - epoch_start) / 60
            avg_epoch_loss = epoch_loss / len(dataloader)
            temp = _get_gpu_temp()
            temp_str = f", GPU: {temp}°C" if temp else ""
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} — "
                f"avg loss: {avg_epoch_loss:.4f}, "
                f"time: {epoch_time:.1f}min{temp_str}"
            )

            # Cooling pause — վերջին epoch-ից հետո ՉԻ անում
            is_last_epoch = (epoch == num_epochs - 1)
            if not is_last_epoch and self.accelerator.is_main_process:
                cooling_pause(self.cooldown_seconds, self.gpu_temp_limit)

        # ── Final save ─────────────────────────────────────────
        if self.accelerator.is_main_process:
            save_lora_weights(
                self.accelerator.unwrap_model(unet),
                self.train_cfg["output_dir"],
            )
            logger.info("✅ Training complete!")
            self._log_gpu_stats()

        self.writer.close()