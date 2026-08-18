import os
import glob
import json
import numpy as np
import torch
from PIL import Image
from scipy import stats

from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

from importlib import import_module
train_module = import_module("02_train_efe_vla_real")
EFEHead = train_module.EFEHead
DEFAULT_DATA_DIR = train_module._DEFAULT_BRIDGE_OUT_DIR  # parent dir, not .../all_frames

SIGMA2_COLLAPSE_STD_THRESHOLD = 1e-3


def get_holdout_episode_files(data_dir: str, holdout_fraction: float,
                              holdout_seed: int):
    all_files = sorted(glob.glob(f"{data_dir}/*.npz"))
    assert len(all_files) > 0, f"No data found in {data_dir}"

    holdout_files = {"occluded": [], "open": []}
    for category in ["occluded", "open"]:
        cat_files = [f for f in all_files
                     if os.path.basename(f).startswith(f"{category}_")]
        if not cat_files:
            continue
        shuffled = cat_files.copy()
        np.random.default_rng(holdout_seed).shuffle(shuffled)
        n_holdout = max(1, int(len(shuffled) * holdout_fraction))
        holdout_files[category] = shuffled[:n_holdout]

    print(f"Held-out episodes (fraction={holdout_fraction}, seed={holdout_seed}): "
          f"occluded={len(holdout_files['occluded'])}  "
          f"open={len(holdout_files['open'])}")
    return holdout_files


def load_trained_model(lora_dir: str, efe_head_path: str, device: str = "cuda:0"):
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    print("Loading base OpenVLA-7B in bfloat16 (matches training)...")
    base = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    print(f"Loading trained LoRA adapters from {lora_dir}...")
    vla = PeftModel.from_pretrained(base, lora_dir).to(device)
    vla.eval()

    efe_head = EFEHead(hidden_dim=4096, latent_dim=128).to(device)
    if efe_head_path and os.path.exists(efe_head_path):
        efe_head.load_state_dict(torch.load(efe_head_path, map_location=device))
        print(f"Loaded EFE head from {efe_head_path}")
    else:
        print(f"WARNING: {efe_head_path} not found -- EFE head is UNTRAINED "
              f"(random init). sigma2 values will be meaningless.")
    efe_head.eval()

    captured = {}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        captured["h"] = h.mean(dim=1).float()
    try:
        vla.base_model.model.language_model.model.norm.register_forward_hook(hook)
    except AttributeError:
        print("WARNING: hidden-state hook path not found -- sigma2 will be all "
              "zero. Run print(vla) and fix the path in load_trained_model().")

    return vla, processor, efe_head, captured


@torch.no_grad()
def predict_and_measure(vla, processor, efe_head, captured, rgb_img,
                        instruction: str, device: str):
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, rgb_img, return_tensors="pt")
    inputs = {
        k: (v.to(device, dtype=torch.bfloat16) if v.is_floating_point()
            else v.to(device))
        for k, v in inputs.items()
    }

    pred_action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    pred_action = np.asarray(pred_action, dtype=np.float32)

    h = captured.get("h")
    sigma2 = 0.0
    if h is not None:
        z, mu, log_var = efe_head(h)
        # FIX A: sigma2 = exp(s) where s = log_var.mean(-1), matching
        # 02_train_efe_vla_real.py's FIX 15(e) exactly -- NOT
        # exp(log_var).mean(-1), which is a different quantity by
        # Jensen's inequality and was never what training optimized.
        s = log_var.mean(dim=-1)
        sigma2 = torch.exp(s).mean().item()

    return pred_action, sigma2


def evaluate_offline(vla, processor, efe_head, captured, data_dir: str,
                     holdout_fraction: float = 0.2, holdout_seed: int = 123,
                     steps_per_episode: int = 5, device: str = "cuda:0",
                     seed: int = 42):
    rng = np.random.default_rng(seed)
    holdout_files = get_holdout_episode_files(data_dir, holdout_fraction, holdout_seed)

    results = {
        "occluded": {"errors": [], "sigma2s": [], "episode_sigma2s": []},
        "open":     {"errors": [], "sigma2s": [], "episode_sigma2s": []},
    }

    for category in ["occluded", "open"]:
        files = holdout_files[category]
        if not files:
            print(f"WARNING: no held-out files for category={category}")
            continue

        for ep_idx, ep_file in enumerate(files):
            with np.load(ep_file, allow_pickle=True) as d:
                n_steps = int(d["n_steps"])
                instr   = str(d["instruction"])
                step_idxs = rng.choice(
                    n_steps, size=min(steps_per_episode, n_steps), replace=False
                )
                episode_sigma2s = []
                for step_idx in step_idxs:
                    rgb_img   = Image.fromarray(d["rgb"][step_idx].astype(np.uint8))
                    gt_action = d["action"][step_idx].astype(np.float32)

                    pred_action, sigma2 = predict_and_measure(
                        vla, processor, efe_head, captured, rgb_img, instr, device)

                    err = float(np.linalg.norm(
                        pred_action[:len(gt_action)] - gt_action))
                    results[category]["errors"].append(err)
                    results[category]["sigma2s"].append(sigma2)
                    episode_sigma2s.append(sigma2)

                if episode_sigma2s:
                    results[category]["episode_sigma2s"].append(
                        float(np.mean(episode_sigma2s))
                    )

            if (ep_idx + 1) % 20 == 0:
                print(f"  [{category}] {ep_idx+1}/{len(files)} episodes evaluated")

    return results


def _report_two_sample_tests(occ_vals, open_vals, label: str):

    occ_vals = np.asarray(occ_vals, dtype=np.float64)
    open_vals = np.asarray(open_vals, dtype=np.float64)

    print(f"\n  [{label}] occluded n={len(occ_vals)}  open n={len(open_vals)}")
    for name, vals in (("occluded", occ_vals), ("open", open_vals)):
        median = np.median(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        print(f"    {name:9s}: mean={vals.mean():.4f} std={vals.std():.4f}  "
              f"median={median:.4f}  IQR=[{q1:.4f}, {q3:.4f}]")

    combined_std = np.concatenate([occ_vals, open_vals]).std()
    collapsed = combined_std < SIGMA2_COLLAPSE_STD_THRESHOLD
    if collapsed:
        print(f"    WARNING: combined sigma2 std ({combined_std:.6f}) is below "
              f"the collapse threshold ({SIGMA2_COLLAPSE_STD_THRESHOLD}) -- "
              f"the EFE head may be producing near-constant output at eval "
              f"time. Treat any 'significant' result below with real "
              f"suspicion; a near-constant signal can still test significant "
              f"with enough samples while carrying no real discrimination.")

    t_stat, t_p = stats.ttest_ind(occ_vals, open_vals, equal_var=False)  # Welch's
    u_stat, u_p = stats.mannwhitneyu(occ_vals, open_vals, alternative="two-sided")

    occ_higher_mean = occ_vals.mean() > open_vals.mean()
    occ_higher_median = np.median(occ_vals) > np.median(open_vals)

    print(f"    Welch's t-test      : t={t_stat:.3f}  p={t_p:.4f}  "
          f"({'occ>open' if occ_higher_mean else 'occ<=open'} by mean)")
    print(f"    Mann-Whitney U      : U={u_stat:.1f}  p={u_p:.4f}  "
          f"({'occ>open' if occ_higher_median else 'occ<=open'} by median)")

    t_sig = t_p < 0.05
    u_sig = u_p < 0.05
    if t_sig != u_sig:
        print(f"    NOTE: the two tests DISAGREE on significance "
              f"(t-test {'sig' if t_sig else 'not sig'}, "
              f"Mann-Whitney {'sig' if u_sig else 'not sig'}). This usually "
              f"means the effect is being driven by a few extreme values "
              f"rather than a consistent shift -- worth plotting the raw "
              f"distributions before claiming either result.")
    elif t_sig and u_sig and (occ_higher_mean != occ_higher_median):
        print(f"    NOTE: both tests significant but disagree on DIRECTION "
              f"between mean and median -- likely outlier-driven. Investigate "
              f"before writing this up.")
    elif t_sig and u_sig and occ_higher_mean and occ_higher_median:
        print(f"    -> SIGNIFICANT and consistent in the expected direction "
              f"across both tests.")
    else:
        print(f"    -> Not significant, or significant in the wrong direction.")

    return {
        "n_occluded": int(len(occ_vals)), "n_open": int(len(open_vals)),
        "occluded_mean": float(occ_vals.mean()), "occluded_std": float(occ_vals.std()),
        "occluded_median": float(np.median(occ_vals)),
        "open_mean": float(open_vals.mean()), "open_std": float(open_vals.std()),
        "open_median": float(np.median(open_vals)),
        "welch_t": float(t_stat), "welch_p": float(t_p),
        "mannwhitney_u": float(u_stat), "mannwhitney_p": float(u_p),
        "possible_collapse": bool(collapsed),
    }


def summarize(results: dict):
    print("\n" + "=" * 66)
    print("  REAL OFFLINE EVALUATION -- held-out BridgeData V2 episodes")
    print("=" * 66)

    stats_out = {}
    for cat in ["occluded", "open"]:
        errs = results[cat]["errors"]
        s2s  = results[cat]["sigma2s"]
        if not errs:
            continue
        print(f"\n  {cat.upper()} tasks (n={len(errs)} frames, "
              f"{len(results[cat]['episode_sigma2s'])} episodes):")
        print(f"    Action prediction L2 error : {np.mean(errs):.4f} ± {np.std(errs):.4f}")
        stats_out[cat] = {
            "n_frames": len(errs),
            "n_episodes": len(results[cat]["episode_sigma2s"]),
            "error_mean": float(np.mean(errs)), "error_std": float(np.std(errs)),
        }

    if results["occluded"]["sigma2s"] and results["open"]["sigma2s"]:
        print(f"\n  UNCERTAINTY-OCCLUSION TEST -- PER-FRAME (less strict; "
              f"frames within an episode are correlated, inflates n):")
        frame_stats = _report_two_sample_tests(
            results["occluded"]["sigma2s"], results["open"]["sigma2s"],
            label="per-frame"
        )
        stats_out["per_frame"] = frame_stats

        if results["occluded"]["episode_sigma2s"] and results["open"]["episode_sigma2s"]:
            print(f"\n  UNCERTAINTY-OCCLUSION TEST -- PER-EPISODE (stricter; "
                  f"the more defensible number if this disagrees with per-frame):")
            episode_stats = _report_two_sample_tests(
                results["occluded"]["episode_sigma2s"], results["open"]["episode_sigma2s"],
                label="per-episode"
            )
            stats_out["per_episode"] = episode_stats

            per_frame_sig = frame_stats["welch_p"] < 0.05 and frame_stats["mannwhitney_p"] < 0.05
            per_episode_sig = episode_stats["welch_p"] < 0.05 and episode_stats["mannwhitney_p"] < 0.05
            if per_frame_sig and not per_episode_sig:
                print(f"\n  NOTE: per-frame test is significant but per-episode is "
                      f"NOT. This is a real warning sign -- it suggests the "
                      f"per-frame result may be inflated by treating correlated "
                      f"frames from the same episode as independent samples. "
                      f"Lead with the per-episode result if writing this up.")
    print("=" * 66)
    return stats_out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=f"{DEFAULT_DATA_DIR}/all_frames")
    p.add_argument("--lora_dir", type=str, default="efe_vla_lora_adapters")
    p.add_argument("--efe_head", type=str, default="efe_head.pt")
    p.add_argument("--holdout_fraction", type=float, default=0.2,
                    help="MUST match what you trained with, or this is not "
                         "a genuine held-out eval.")
    p.add_argument("--holdout_seed", type=int, default=123,
                    help="MUST match what you trained with.")
    p.add_argument("--steps_per_episode", type=int, default=5)
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    vla, processor, efe_head, captured = load_trained_model(
        args.lora_dir, args.efe_head, device)

    results = evaluate_offline(
        vla, processor, efe_head, captured, args.data_dir,
        holdout_fraction=args.holdout_fraction, holdout_seed=args.holdout_seed,
        steps_per_episode=args.steps_per_episode, device=device)
    stats_out = summarize(results)

    with open("offline_eval_results.json", "w") as f:
        json.dump(stats_out, f, indent=2)
    print("\nSaved -> offline_eval_results.json  (per-frame + per-episode, "
          "Welch's t-test + Mann-Whitney U, means/medians/IQR)")
