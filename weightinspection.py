import torch
import numpy as np

from importlib import import_module
train_module = import_module("02_train_efe_vla_real")
EFEHead = train_module.EFEHead


def inspect_checkpoint(efe_head_path: str = "efe_head.pt",
                       hidden_dim: int = 4096, latent_dim: int = 128,
                       n_probe: int = 200, seed: int = 123):
    print("=" * 64)
    print("  DIRECT WEIGHT INSPECTION — efe_head.pt")
    print("=" * 64)

    efe_head = EFEHead(hidden_dim=hidden_dim, latent_dim=latent_dim)
    state = torch.load(efe_head_path, map_location="cpu")
    efe_head.load_state_dict(state)
    efe_head.eval()

    # ── 1. Structural check: weight matrix statistics ──
    log_var_W = efe_head.log_var_head.weight.detach()
    log_var_b = efe_head.log_var_head.bias.detach()
    mu_W      = efe_head.mu_head.weight.detach()

    print("\n[1] STRUCTURAL CHECK (weight matrix itself)")
    print(f"    log_var_head.weight  : norm={log_var_W.norm().item():.4f}  "
          f"mean_abs={log_var_W.abs().mean().item():.6f}  "
          f"max_abs={log_var_W.abs().max().item():.4f}")
    print(f"    log_var_head.bias    : norm={log_var_b.norm().item():.4f}  "
          f"mean={log_var_b.mean().item():.4f}")
    print(f"    mu_head.weight       : norm={mu_W.norm().item():.4f}")
    print(f"    log_tau              : {efe_head.log_tau.item():.4f}  "
          f"(tau = {torch.exp(efe_head.log_tau).item():.4f})")

    init_std = 0.01  # what log_var_head.weight was initialized with
    ratio = log_var_W.abs().mean().item() / init_std
    print(f"\n    log_var_head weight magnitude vs. its own init (std=0.01): "
          f"{ratio:.2f}x")
    if ratio < 1.5:
        print("    WARNING: barely moved from initialization. This head has")
        print("    learned almost nothing — gradient may not be reaching it,")
        print("    independent of any collapse-to-a-constant question below.")

    # ── 2. Functional check: does it respond to RANDOM, unrelated input? ──
    print(f"\n[2] FUNCTIONAL CHECK — {n_probe} random 4096-dim probe vectors")
    print("    (nothing to do with real images — pure noise, mean=0 std=1,")
    print("     same scale roughly comparable to a normalized hidden state)")

    torch.manual_seed(seed)
    probe_h = torch.randn(n_probe, hidden_dim)

    with torch.no_grad():
        _, mu, log_var = efe_head(probe_h)
        s = log_var.mean(dim=-1)
        sigma2 = torch.exp(s)

    print(f"\n    sigma2 over {n_probe} random probes:")
    print(f"      mean   = {sigma2.mean().item():.4f}")
    print(f"      std    = {sigma2.std().item():.6f}")
    print(f"      min    = {sigma2.min().item():.4f}")
    print(f"      max    = {sigma2.max().item():.4f}")
    print(f"      median = {sigma2.median().item():.4f}")

    log_var_floor = efe_head.log_var_clamp[0]
    floor_val = float(np.exp(log_var_floor))
    frac_at_floor = (sigma2 - floor_val).abs().lt(1e-4).float().mean().item()
    print(f"\n    fraction of probes landing within 1e-4 of the clamp floor "
          f"({floor_val:.4f}): {frac_at_floor:.1%}")

    print("\n" + "=" * 64)
    print("  VERDICT")
    print("=" * 64)
    if sigma2.std().item() < 1e-3 or frac_at_floor > 0.8:
        print("""
  COLLAPSED. Even completely random, unrelated noise vectors produce
  near-identical sigma2 output. This head is a constant function of its
  input right now — this has NOTHING to do with real occlusion data,
  it's a structural property of the current weights. Any eval result
  (real or synthetic) run against THIS checkpoint is measuring noise,
  not a real signal.

  -> Do not re-run eval against this checkpoint. Retrain is needed.
  -> If you already retrained with FIX 16 (softplus fix) and get this
     same verdict, the exploit is not fully closed -- come back with
     this script's output and the new training log's L_task_hetero
     values (specifically: any negative values still appearing).
""")
    elif sigma2.std().item() < 0.05:
        print("""
  WEAK BUT NOT FULLY COLLAPSED. Some spread exists across random noise,
  but it's small. The head is minimally responsive to input. Real-data
  eval results from this checkpoint should be treated with real
  suspicion -- a weak-but-real signal on unrelated noise inputs doesn't
  guarantee a MEANINGFUL signal on real occluded-vs-open content, but at
  least the earlier "hard collapse to a single constant" explanation is
  ruled out.

  -> Worth running the real-data eval once more against THIS checkpoint
     specifically to see if occ/open separate at all, but keep
     expectations modest.
""")
    else:
        print("""
  RESPONSIVE. The head produces meaningfully varied output even for
  random, structurally unrelated input -- it is NOT a dead/collapsed
  function. This means the earlier real-data eval floor result (if it
  was scored against THIS exact checkpoint) is more likely a genuine
  property of how real BridgeData images map through this head, not a
  structural collapse -- worth investigating on the real-data side
  specifically (e.g. the train/eval h-context question raised earlier),
  rather than assuming the weights themselves are broken.

  -> Re-run the real-data eval against this exact checkpoint and check
     whether sigma2_std (already logged by your eval script) is
     meaningfully nonzero there too. If real data ALSO shows near-zero
     std despite this synthetic test showing real variation, that
     points at something specific to how h is computed from real
     images/prompts, not the head's weights themselves.
""")
    print("=" * 64)

    return {
        "log_var_weight_norm": float(log_var_W.norm().item()),
        "log_var_weight_vs_init_ratio": float(ratio),
        "probe_sigma2_mean": float(sigma2.mean().item()),
        "probe_sigma2_std": float(sigma2.std().item()),
        "probe_fraction_at_floor": float(frac_at_floor),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--efe_head", type=str, default="efe_head.pt")
    p.add_argument("--n_probe", type=int, default=200)
    args, _unknown = p.parse_known_args()

    inspect_checkpoint(efe_head_path=args.efe_head, n_probe=args.n_probe)
