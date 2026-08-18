import os
import glob
import json
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Dict, List

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ── Colab path setup (matches 01_bridge_real_data_colab.py exactly) ────────
_DRIVE_ROOT = "/content/drive/MyDrive"
if os.path.isdir(_DRIVE_ROOT):
    _DEFAULT_BRIDGE_OUT_DIR = f"{_DRIVE_ROOT}/bridge_real_data"
else:
    _DEFAULT_BRIDGE_OUT_DIR = "/content/bridge_real_data"
DEFAULT_DATA_DIR = f"{_DEFAULT_BRIDGE_OUT_DIR}/all_frames"

OPENVLA_REPO_DIR = "/content/openvla"


def _load_action_tokenizer_class(openvla_repo_dir: str):
    action_tokenizer_path = os.path.join(
        openvla_repo_dir, "prismatic", "vla", "action_tokenizer.py"
    )
    if not os.path.isfile(action_tokenizer_path):
        raise RuntimeError(
            f"Could not find {action_tokenizer_path}. Clone the OpenVLA "
            f"repo first:\n"
            f"  !git clone https://github.com/openvla/openvla {openvla_repo_dir}\n"
            f"Training cannot proceed without the real action tokenizer."
        )
    spec = importlib.util.spec_from_file_location(
        "action_tokenizer", action_tokenizer_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ActionTokenizer


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — EFE Head
# ─────────────────────────────────────────────────────────────────────────────

class EFEHead(nn.Module):
    def __init__(self, hidden_dim: int = 4096, latent_dim: int = 128):
        super().__init__()
        self.mu_head      = nn.Linear(hidden_dim, latent_dim)
        self.log_var_head = nn.Linear(hidden_dim, latent_dim)
        nn.init.normal_(self.log_var_head.weight, std=0.01)
        nn.init.zeros_(self.log_var_head.bias)
        self.log_tau = nn.Parameter(torch.tensor(0.4))   # tau_init ≈ 1.5

        self.mu_clamp      = (-10.0, 10.0)

    def forward(self, h: torch.Tensor):
        """h: (B, hidden_dim) — mean-pooled last hidden state."""
        mu      = self.mu_head(h).clamp(*self.mu_clamp)
        log_var = self.log_var_head(h).clamp(*self.log_var_clamp)
        z       = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        return z, mu, log_var

    def efe_loss(self, mu, log_var):
        return -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        )

    def uncertainty(self, log_var) -> torch.Tensor:
        return torch.exp(log_var).mean(dim=-1)

    def sigma2_ceiling(self) -> float:
        return float(np.exp(self.log_var_clamp[1]))

    def soft_ceiling_penalty(self, s: torch.Tensor, s_max: float,
                              beta: float, s_min: float = -1.5) -> torch.Tensor:

        beta * relu(s - s_max)^2 + beta * relu(s_min - s)^2
        return (beta * F.relu(s - s_max).pow(2)
                + beta * F.relu(s_min - s).pow(2))


def occlusion_ranking_loss(s: torch.Tensor, categories: List[str],
                            margin: float = 0.2) -> torch.Tensor:
    occ_idx  = [i for i, c in enumerate(categories) if c == "occluded"]
    open_idx = [i for i, c in enumerate(categories) if c == "open"]
    if not occ_idx or not open_idx:
        return torch.zeros((), device=s.device, dtype=s.dtype)

    s_occ  = s[occ_idx].mean()
    s_open = s[open_idx].mean()
    return F.relu(margin - (s_occ - s_open))


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — Load real OpenVLA-7B with LoRA
# ─────────────────────────────────────────────────────────────────────────────

def load_efe_openvla(device: str = "cuda:0", use_4bit: bool = False,
                      gradient_checkpointing: bool = True):
    import transformers
    _PINNED_TRANSFORMERS_VERSION = "4.44.2"
    if transformers.__version__ != _PINNED_TRANSFORMERS_VERSION:
        print(f"NOTE: running transformers=={transformers.__version__}, "
              f"but this file pins {_PINNED_TRANSFORMERS_VERSION}. "
              f"Proceeding, but re-pin and restart the kernel if you hit "
              f"loading/inference errors.")

    print("Loading OpenVLA-7B processor...")
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True
    )

    if use_4bit:
        print("Loading OpenVLA-7B in 4-bit (this needs ~8-10GB VRAM)...")
        print("WARNING: use_4bit=True has a known unresolved dispatch_model "
              "incompatibility — see this function's docstring.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map=device,
        )
    else:
        print("Loading OpenVLA-7B in bfloat16 (no quantization) — matches "
              "OpenVLA's own tested/documented loading path. Needs ~14GB "
              "VRAM for the base model alone.")
        vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)

    if use_4bit:
        vla = prepare_model_for_kbit_training(vla)

    print("Attaching LoRA adapters...")
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    vla = get_peft_model(vla, lora_config)

    if gradient_checkpointing:
        vla.gradient_checkpointing_enable()
        vla.enable_input_require_grads()
        print("Gradient checkpointing enabled.")

    vla.print_trainable_parameters()
    actual_device = next(vla.parameters()).device
    print(f"OpenVLA-7B placed on: {actual_device}")
    if device.startswith("cuda") and actual_device.type != "cuda":
        print(f"WARNING: requested device={device} but model landed on "
              f"{actual_device} — downstream tensors must be moved to "
              f"{actual_device}, not the originally requested {device}.")

    efe_head = EFEHead(hidden_dim=4096, latent_dim=128).to(actual_device)

    captured = {}
    def _capture_hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        captured["h"] = hidden.mean(dim=1).float()

    try:
        target_module = vla.base_model.model.language_model.model.norm
    except AttributeError:
        target_module = None

    if target_module is None:
        raise RuntimeError(
            "Could not find language_model.model.norm on the loaded model "
            "to attach the EFE hook. Run `print(vla)` to inspect the real "
            "module tree and update the attribute path here."
        )

    target_module.register_forward_hook(_capture_hook)

    print("Verifying EFE hook fires with a dummy forward pass...")
    try:
        dummy_prompt = "In: What action should the robot take to test?\nOut:"
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        dummy_inputs = processor(dummy_prompt, dummy_image, return_tensors="pt")
        dummy_inputs = {
            k: (v.to(actual_device, dtype=torch.bfloat16) if v.is_floating_point()
                else v.to(actual_device))
            for k, v in dummy_inputs.items()
        }
        with torch.no_grad():
            vla(**dummy_inputs)
    except Exception as e:
        raise RuntimeError(
            f"Dummy forward pass to verify the EFE hook failed outright: {e}."
        ) from e

    if captured.get("h") is None:
        raise RuntimeError(
            "EFE hook was registered but did NOT fire during a dummy "
            "forward pass. Run `print(vla)` and inspect the real call "
            "graph, then update the hook target."
        )
    print(f"EFE hook verified — captured hidden state shape: "
          f"{tuple(captured['h'].shape)}")
    captured.clear()

    return vla, processor, efe_head, captured, actual_device


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — Dataset
# ─────────────────────────────────────────────────────────────────────────────

class EFEVLADataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, max_episodes: Optional[int] = None,
                 holdout_fraction: float = 0.0, holdout_seed: int = 123,
                 train_split: bool = True):
        all_files = sorted(glob.glob(f"{data_dir}/*.npz"))
        assert len(all_files) > 0, (
            f"No data found in {data_dir} -- run "
            f"01_bridge_real_data_colab.py first, then point --data_dir at "
            f"the resulting .../all_frames directory"
        )

        if holdout_fraction > 0:
            episode_files = []
            for category in ["occluded", "open"]:
                cat_files = [f for f in all_files
                             if os.path.basename(f).startswith(f"{category}_")]
                if not cat_files:
                    continue
                shuffled = cat_files.copy()
                np.random.default_rng(holdout_seed).shuffle(shuffled)
                n_holdout = max(1, int(len(shuffled) * holdout_fraction))
                holdout_set = set(shuffled[:n_holdout])
                if train_split:
                    kept = [f for f in cat_files if f not in holdout_set]
                else:
                    kept = [f for f in cat_files if f in holdout_set]
                episode_files.extend(kept)
            episode_files = sorted(episode_files)
            print(f"  [holdout] fraction={holdout_fraction} seed={holdout_seed} "
                  f"train_split={train_split} -> {len(episode_files)}/{len(all_files)} "
                  f"episode files kept")
        else:
            episode_files = all_files

        if max_episodes is not None:
            episode_files = episode_files[:max_episodes]

        self.index = []
        n_occ_episodes, n_open_episodes = 0, 0
        for ep_file in episode_files:
            fname = os.path.basename(ep_file)
            if fname.startswith("occluded_"):
                category = "occluded"
                n_occ_episodes += 1
            elif fname.startswith("open_"):
                category = "open"
                n_open_episodes += 1
            else:
                category = "unknown"
            with np.load(ep_file, allow_pickle=True) as d:
                n_steps = int(d["n_steps"])
            for step_idx in range(n_steps):
                self.index.append((ep_file, step_idx, category))

        print(f"Loaded {len(episode_files)} real BridgeData V2 episodes "
              f"({len(self.index)} total transitions) from {data_dir}"
              + (f"  [max_episodes={max_episodes}]" if max_episodes else "")
              + (f"  [holdout_fraction={holdout_fraction} EXCLUDED from training]"
                 if holdout_fraction > 0 and train_split else ""))
        print(f"  occluded-task episodes: {n_occ_episodes}  |  "
              f"open-task episodes: {n_open_episodes}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_file, step_idx, category = self.index[idx]
        with np.load(ep_file, allow_pickle=True) as d:
            rgb    = Image.fromarray(d["rgb"][step_idx].astype(np.uint8))
            action = torch.tensor(d["action"][step_idx], dtype=torch.float32)
            instr  = str(d["instruction"])
        return rgb, instr, action, category


def collate_fn(batch, processor, action_tokenizer):
    images, instrs, actions, categories = zip(*batch)

    full_texts = []
    action_token_strs = []
    for instr, action in zip(instrs, actions):
        action_str = action_tokenizer(action.cpu().numpy())
        action_token_strs.append(action_str)
        full_texts.append(
            f"In: What action should the robot take to {instr}?\nOut:{action_str}"
        )

    inputs = processor(full_texts, list(images), padding=True, return_tensors="pt")

    labels = inputs["input_ids"].clone()
    labels[:] = -100
    for i, action_str in enumerate(action_token_strs):
        action_only_ids = processor.tokenizer(
            action_str, add_special_tokens=False
        )["input_ids"]
        n_action_tokens = len(action_only_ids)
        if n_action_tokens == 0:
            continue
        row_ids = inputs["input_ids"][i]
        attn = inputs["attention_mask"][i]
        real_len = int(attn.sum().item())
        end = real_len
        start = max(end - n_action_tokens, 0)
        labels[i, start:end] = row_ids[start:end]

    actions = torch.stack(actions)
    return inputs, labels, actions, list(categories)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — Training loop
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_example_task_loss(outputs, labels: torch.Tensor,
                                   vocab_size: int) -> torch.Tensor:
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)                      # (B, seq_len-1)

    mask  = (shift_labels != -100).float()
    denom = mask.sum(dim=-1).clamp(min=1.0)
    per_example_loss = (per_token_loss * mask).sum(dim=-1) / denom   # (B,)
    return per_example_loss


def train(vla, processor, efe_head, captured, data_dir: str,
          n_epochs: int = 2, batch_size: int = 4, lr: float = 1e-4,
          lambda_e: float = 1e-3, lambda_mu: float = 1e-3,
          lambda_tau: float = 0.01,
          forage_target_sigma2: float = 3.0,
          max_episodes: Optional[int] = None,
          holdout_fraction: float = 0.0, holdout_seed: int = 123,
          hetero_warmup_steps: int = 150,
          soft_ceiling_s_max: float = 1.5,
          soft_ceiling_beta: float = 0.1,
          norm_alpha: float = 0.05,
          lambda_rank: float = 0.05,
          rank_margin: float = 0.2,
          device=None):
    if device is None:
        device = next(vla.parameters()).device

    if lambda_e != 1e-3:
        print(f"NOTE: lambda_e={lambda_e} was passed, but this term is "
              f"deprecated since FIX 12 -- lambda_e no longer affects "
              f"training. Logged only.")

    ActionTokenizer = _load_action_tokenizer_class(OPENVLA_REPO_DIR)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    dataset = EFEVLADataset(data_dir, max_episodes=max_episodes,
                             holdout_fraction=holdout_fraction,
                             holdout_seed=holdout_seed, train_split=True)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor, action_tokenizer),
    )

    trainable_params = (
        [p for p in vla.parameters() if p.requires_grad] +
        list(efe_head.parameters())
    )
    opt = torch.optim.AdamW(trainable_params, lr=lr)

    sigma2_ema = torch.tensor(1.0, device=device)
    ema_alpha  = 0.05

    # per-category running sigma2, for an in-training early signal.
    cat_sigma2_ema = {"occluded": torch.tensor(1.0, device=device),
                       "open":     torch.tensor(1.0, device=device)}

    task_loss_mean_ema = torch.tensor(1.0, device=device)
    task_loss_std_ema  = torch.tensor(1.0, device=device)

    sigma2_ceiling = efe_head.sigma2_ceiling()
    saturation_streak = 0
    SATURATION_WARN_STREAK = 5
    SATURATION_THRESHOLD = 0.95

    global_step = 0

    print(f"\nTraining EFE-VLA (real OpenVLA-7B + LoRA + EFE head) "
          f"for {n_epochs} epochs, batch_size={batch_size}, "
          f"lambda_mu={lambda_mu}, sigma2_ceiling={sigma2_ceiling:.2f} "
          f"(hard, widened -- soft ceiling s_max={soft_ceiling_s_max} "
          f"beta={soft_ceiling_beta} is the intended binding constraint), "
          f"hetero_warmup_steps={hetero_warmup_steps}, "
          f"holdout_fraction={holdout_fraction}, holdout_seed={holdout_seed}  "
          f"[FIX 14: warmup + soft ceiling + normalized hetero coupling]")
    if holdout_fraction <= 0:
        print(f"  WARNING: holdout_fraction=0 -- training on ALL episodes, "
              f"no data held out. Any eval against this checkpoint using "
              f"the same data_dir will NOT be a genuine held-out test.")

    for epoch in range(n_epochs):
        for step, (inputs, labels, actions, categories) in enumerate(loader):
            inputs = {
                k: (v.to(device, dtype=torch.bfloat16) if v.is_floating_point()
                    else v.to(device))
                for k, v in inputs.items()
            }
            labels  = labels.to(device)
            actions = actions.to(device)
            B = labels.shape[0]

            # ── per-example forward passes (FIX 13, unchanged) ──
            L_task_list, mu_list, log_var_list = [], [], []
            for i in range(B):
                single_inputs = {k: v[i:i+1] for k, v in inputs.items()}
                single_labels = labels[i:i+1]

                outputs_i = vla(**single_inputs, labels=single_labels)
                if outputs_i.loss is None:
                    raise RuntimeError(
                        f"outputs.loss is None for example {i} — labels may "
                        f"be all -100 (no action tokens found). Check "
                        f"collate_fn's label masking for this batch."
                    )
                L_task_list.append(outputs_i.loss)

                h_i = captured.get("h")
                if h_i is None:
                    raise RuntimeError(
                        "EFE hook stopped producing captured['h'] mid-training. "
                        "It was verified working before training started — "
                        "do not silently skip batches, investigate."
                    )
                _, mu_i, log_var_i = efe_head(h_i)   # each (1, latent_dim)
                mu_list.append(mu_i)
                log_var_list.append(log_var_i)

            L_task_per_example = torch.stack(L_task_list)        # (B,)
            mu      = torch.cat(mu_list, dim=0)                  # (B, latent_dim)
            log_var = torch.cat(log_var_list, dim=0)             # (B, latent_dim)

            # Diagnostic only — logged, not in L_total.
            L_EFE = efe_head.efe_loss(mu, log_var)

            s = log_var.mean(dim=-1)                 
            sigma2_per_example = torch.exp(s)                      # (B,)
            sigma2      = sigma2_per_example.mean()
            sigma2_std  = sigma2_per_example.std(unbiased=False)   # spread, not just mean
            sigma2_ema  = (1 - ema_alpha) * sigma2_ema + ema_alpha * sigma2.detach()

            with torch.no_grad():
                batch_mean = L_task_per_example.mean()
                batch_std  = L_task_per_example.std(unbiased=False).clamp(min=1e-3)
                task_loss_mean_ema = (1 - norm_alpha) * task_loss_mean_ema + norm_alpha * batch_mean
                task_loss_std_ema  = (1 - norm_alpha) * task_loss_std_ema  + norm_alpha * batch_std

            z = (L_task_per_example - task_loss_mean_ema) / (task_loss_std_ema + 1e-6)
            L_task_normalized = F.softplus(z)

            global_step += 1
            in_warmup = global_step <= hetero_warmup_steps

            if in_warmup:
                L_task_hetero = L_task_per_example.mean()
                L_rank = torch.zeros((), device=device, dtype=L_task_hetero.dtype)
            else:
                ceiling_penalty = efe_head.soft_ceiling_penalty(
                    s, soft_ceiling_s_max, soft_ceiling_beta
                )
                L_task_hetero = (
                    L_task_normalized * torch.exp(-s) + 0.5 * s + ceiling_penalty
                ).mean()
                L_rank = occlusion_ranking_loss(s, categories, margin=rank_margin)

            L_tau    = (efe_head.log_tau - torch.log(sigma2_ema + 1e-6)).pow(2)
            L_mu_reg = mu.pow(2).mean()

            L_total = (L_task_hetero + lambda_mu * L_mu_reg + lambda_tau * L_tau
                       + lambda_rank * L_rank)

            opt.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()

            # ── per-category running EMA ──
            with torch.no_grad():
                for cat in ("occluded", "open"):
                    idxs = [i for i, c in enumerate(categories) if c == cat]
                    if not idxs:
                        continue
                    cat_mean = sigma2_per_example[idxs].mean()
                    cat_sigma2_ema[cat] = (
                        (1 - ema_alpha) * cat_sigma2_ema[cat] + ema_alpha * cat_mean
                    )

            if step % 20 == 0:
                L_task_mean = L_task_per_example.mean().item()
                warmup_tag = " [WARMUP]" if in_warmup else ""
                print(f"  epoch {epoch} step {step:4d} (global {global_step}){warmup_tag} | "
                      f"L_task={L_task_mean:.4f}  L_task_hetero={L_task_hetero.item():.4f}  "
                      f"L_rank={L_rank.item():.4f}  "
                      f"L_EFE(diag)={L_EFE.item():.4f}  "
                      f"sigma2={sigma2.item():.4f}  sigma2_std={sigma2_std.item():.4f}  "
                      f"tau={torch.exp(efe_head.log_tau).item():.4f}  "
                      f"| occ_ema={cat_sigma2_ema['occluded'].item():.4f}  "
                      f"open_ema={cat_sigma2_ema['open'].item():.4f}  "
                      f"gap={cat_sigma2_ema['occluded'].item() - cat_sigma2_ema['open'].item():+.4f}")

                if sigma2.item() >= SATURATION_THRESHOLD * sigma2_ceiling:
                    saturation_streak += 1
                    if saturation_streak >= SATURATION_WARN_STREAK:
                        print(f"  WARNING: sigma2 >= {SATURATION_THRESHOLD:.0%} "
                              f"of HARD clamp ceiling ({sigma2_ceiling:.2f}) for "
                              f"{saturation_streak} consecutive logged steps. "
                              f"With the widened clamp this should be rare -- "
                              f"if it persists, the soft ceiling isn't binding "
                              f"and soft_ceiling_beta likely needs to increase.")
                else:
                    saturation_streak = 0

    vla.save_pretrained("efe_vla_lora_adapters")
    torch.save(efe_head.state_dict(), "efe_head.pt")
    print("\nSaved LoRA adapters -> efe_vla_lora_adapters/")
    print("Saved EFE head       -> efe_head.pt")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--n_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--no_grad_checkpointing", action="store_true")
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--forage_target_sigma2", type=float, default=3.0,
                    help="DEPRECATED since FIX 12 -- logging/CLI "
                         "compatibility only, no longer affects training.")
    p.add_argument("--lambda_e", type=float, default=1e-3,
                         "compatibility only, no longer affects training.")
    p.add_argument("--lambda_mu", type=float, default=1e-3,
                    help="Weight on mu.pow(2).mean() — keeps mu_head from "
                         "sitting at random init (heteroscedastic term "
                         "only touches log_var, not mu).")
    p.add_argument("--holdout_fraction", type=float, default=0.2)
    p.add_argument("--holdout_seed", type=int, default=123)
    p.add_argument("--hetero_warmup_steps", type=int, default=150,
                    help="Global steps to train on raw L_task before "
                         "switching on the heteroscedastic per-example "
                         "coupling")
    p.add_argument("--soft_ceiling_s_max", type=float, default=1.5,
                    help="Soft target for s=log_var.mean(-1) above which "
                         "the ceiling penalty engages")
    p.add_argument("--soft_ceiling_beta", type=float, default=0.1,
                    help="Weight on the soft ceiling penalty "
                         "beta*relu(s-s_max)^2")
    p.add_argument("--norm_alpha", type=float, default=0.05,
                    help="EMA decay for the running mean/std used to "
                         "normalize L_task before it enters the hetero "
                         "term.")
    p.add_argument("--lambda_rank", type=float, default=0.05,
                    help="Weight on the occluded-vs-open ranking hinge "
                         "loss. This is the only term that "
                         "explicitly supervises occ > open.")
    p.add_argument("--rank_margin", type=float, default=0.2,
                    help="Margin for occlusion_ranking_loss(): "
                         "s_occ_mean must exceed s_open_mean by at least "
                         "this much within a batch, or the hinge fires.")
    args, _unknown = p.parse_known_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected")

    print(f"Looking for training data at: {args.data_dir}")

    vla, processor, efe_head, captured, actual_device = load_efe_openvla(
        device, gradient_checkpointing=not args.no_grad_checkpointing
    )
    train(vla, processor, efe_head, captured, args.data_dir,
          n_epochs=args.n_epochs, batch_size=args.batch_size,
          lambda_e=args.lambda_e, lambda_mu=args.lambda_mu,
          forage_target_sigma2=args.forage_target_sigma2,
          max_episodes=args.max_episodes,
          holdout_fraction=args.holdout_fraction,
          holdout_seed=args.holdout_seed,
          hetero_warmup_steps=args.hetero_warmup_steps,
          soft_ceiling_s_max=args.soft_ceiling_s_max,
          soft_ceiling_beta=args.soft_ceiling_beta,
          norm_alpha=args.norm_alpha,
          lambda_rank=args.lambda_rank,
          rank_margin=args.rank_margin,
          device=actual_device)
