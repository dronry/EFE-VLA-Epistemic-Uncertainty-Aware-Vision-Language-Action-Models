import os
import json
import numpy as np
from typing import List, Dict
from PIL import Image

# ── Colab path setup ──────────────────────────────────────────────────────
_DRIVE_ROOT = "/content/drive/MyDrive"
if os.path.isdir(_DRIVE_ROOT):
    DEFAULT_OUT_DIR = f"{_DRIVE_ROOT}/bridge_real_data"
else:
    DEFAULT_OUT_DIR = "/content/bridge_real_data"

# ── keyword-based task split (real language instructions from BridgeData V2) ──
OCCLUDED_KEYWORDS = [
    "drawer", "microwave", "oven", "cabinet", "door", "container",
    "pot", "lid", "box", "inside", "close the", "open the",
]
OPEN_KEYWORDS = [
    "push", "pick up", "move", "put down", "sweep", "stack",
]


def classify_instruction(instr: str) -> str:
    instr_l = instr.lower()
    if any(k in instr_l for k in OCCLUDED_KEYWORDS):
        return "occluded"
    if any(k in instr_l for k in OPEN_KEYWORDS):
        return "open"
    return "unclassified"


def download_and_split(
    out_dir: str = DEFAULT_OUT_DIR,
    max_episodes_per_class: int = 5000,
    data_gcs_dir: str = "gs://gresearch/robotics",
    max_episodes_scanned: int = 20000,
):

    import shutil
    if os.path.isdir(out_dir):
        print(f"Clearing existing {out_dir}/ to avoid mixing stale frames "
              f"from earlier runs (e.g. saved under a previous, since-fixed "
              f"action-parsing version) with freshly saved ones.")
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/occluded", exist_ok=True)
    os.makedirs(f"{out_dir}/open", exist_ok=True)

    print(f"Saving into: {out_dir}")
    print(f"Streaming BridgeData V2 from {data_gcs_dir} ...")
    print(f"Target: {max_episodes_per_class} episodes EACH for occluded/open ")
    print("(First run downloads/caches shards — can take a while depending "
          "on the caps; each trajectory is ~5-20MB of JPEG frames)")

    builder = tfds.builder_from_directory(
        builder_dir=f"{data_gcs_dir}/bridge/0.1.0/"
    )
    ds = builder.as_dataset(split="train")

    meta = {"occluded": [], "open": [], "unclassified": 0}
    n_scanned = 0

    for ep_idx, episode in enumerate(ds):
        n_scanned += 1
        if n_scanned > max_episodes_scanned:
            print(f"  Hit max_episodes_scanned={max_episodes_scanned} raw "
                  f"episodes without both classes reaching their cap — "
                  f"stopping early. Check the per-class counts below; the "
                  f"rarer class may be genuinely scarce in the raw stream, "
                  f"not just under-matched by the keyword lists.")
            break

        if len(meta["occluded"]) >= max_episodes_per_class and \
           len(meta["open"]) >= max_episodes_per_class:
            break

        steps = list(episode["steps"])
        if not steps:
            continue

        # language instruction is attached per-episode (repeated per step)
        if ep_idx == 0:
            # One-time diagnostic: print the ACTUAL keys present on this
            # step, since a KeyError here means runtime structure doesn't
            # match the TFDS catalog docs assumption (nesting, renamed
            # field, or steps[0] not yet materialized as a plain dict).
            print("DEBUG steps[0] keys:", list(steps[0].keys()))
            if "observation" in steps[0]:
                print("DEBUG observation keys:", list(steps[0]["observation"].keys()))

        if "observation" in steps[0] and "natural_language_instruction" in steps[0]["observation"]:
            instr_bytes = steps[0]["observation"]["natural_language_instruction"].numpy()
        elif "language_instruction" in steps[0]:
            instr_bytes = steps[0]["language_instruction"].numpy()
        else:
            meta["unclassified"] += 1
            continue  # can't classify without an instruction field — skip this episode
        instr = instr_bytes.decode("utf-8") if isinstance(instr_bytes, bytes) else str(instr_bytes)
        category = classify_instruction(instr)

        if category == "unclassified":
            meta["unclassified"] += 1
            continue   # skip ambiguous-category episodes for a clean split

        if len(meta[category]) >= max_episodes_per_class:
            continue

        ep_dir = f"{out_dir}/{category}"
        all_rgb, all_proprio, all_action = [], [], []

        for step_idx, step in enumerate(steps):
            rgb    = step["observation"]["image"].numpy()   # real camera frame
            state  = step["observation"]["state"].numpy()       # real proprioception

            if ep_idx == 0 and step_idx == 0:
                if isinstance(step["action"], dict):
                    print("DEBUG action keys:", list(step["action"].keys()))
                    for k, v in step["action"].items():
                        print(f"  action[{k!r}] shape/dtype:",
                              getattr(v, "shape", None), getattr(v, "dtype", None))
                else:
                    print("DEBUG action is a flat Tensor, shape:",
                          step["action"].shape)

            if isinstance(step["action"], dict):
                world_vector   = step["action"]["world_vector"].numpy()
                rotation_delta = step["action"]["rotation_delta"].numpy()
                gripper        = np.array(
                    [1.0 if bool(step["action"]["open_gripper"].numpy()) else 0.0],
                    dtype=np.float32,
                )
                action = np.concatenate(
                    [world_vector, rotation_delta, gripper]
                ).astype(np.float32)
            else:
                action = step["action"].numpy()                     
            all_rgb.append(rgb)
            all_proprio.append(state)
            all_action.append(action)

        frame_path = f"{ep_dir}/ep{ep_idx:05d}.npz"
        np.savez_compressed(
            frame_path,
            rgb=np.stack(all_rgb),          # (T, H, W, 3)
            proprio=np.stack(all_proprio),  # (T, state_dim)
            action=np.stack(all_action),    # (T, 7)
            instruction=instr,              # single string, shared across steps
            n_steps=len(steps),
        )
        frame_paths = [frame_path]

        meta[category].append({
            "episode": ep_idx, "instruction": instr,
            "n_steps": len(steps), "frames": frame_paths,
        })

        n_total_saved = len(meta["occluded"]) + len(meta["open"])
        if n_total_saved % 20 == 0:
            print(f"  scanned={n_scanned}  saved: occluded="
                  f"{len(meta['occluded'])}/{max_episodes_per_class}  "
                  f"open={len(meta['open'])}/{max_episodes_per_class}  "
                  f"skipped_unclassified={meta['unclassified']}")

    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Real trajectories saved (raw episodes scanned: {n_scanned}):")
    print(f"  Occluded-task episodes : {len(meta['occluded'])} / {max_episodes_per_class}")
    print(f"  Open-task episodes     : {len(meta['open'])} / {max_episodes_per_class}")
    print(f"  Skipped (unclassified) : {meta['unclassified']}")
    if len(meta["occluded"]) < max_episodes_per_class or len(meta["open"]) < max_episodes_per_class:
        print(f"  NOTE: at least one class did not reach its cap — either "
              f"max_episodes_scanned was hit first, or the raw dataset ran "
              f"out. This run is NOT balanced; check the counts above "
              f"before training.")
    print(f"\nSaved to {out_dir}/occluded/ and {out_dir}/open/")
    return meta


def build_combined_training_set(out_dir: str = DEFAULT_OUT_DIR):
    combined = f"{out_dir}/all_frames"
    os.makedirs(combined, exist_ok=True)
    n_linked = 0
    for category in ["occluded", "open"]:
        src_dir = f"{out_dir}/{category}"
        if not os.path.isdir(src_dir):
            continue
        for fname in os.listdir(src_dir):
            if fname.endswith(".npz"):
                src_path = os.path.abspath(f"{src_dir}/{fname}")
                dst_path = f"{combined}/{category}_{fname}"
                if not os.path.exists(dst_path):
                    os.symlink(src_path, dst_path)
                    n_linked += 1
    print(f"Combined training set written to {combined}/ "
          f"({n_linked} symlinks, no extra disk used)")
    return combined


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--max_episodes_per_class", type=int, default=5000,
    p.add_argument("--max_episodes_scanned", type=int, default=20000,
    args, _unknown = p.parse_known_args()

    meta = download_and_split(
        out_dir=args.out_dir,
        max_episodes_per_class=args.max_episodes_per_class,
        max_episodes_scanned=args.max_episodes_scanned,
    )
    build_combined_training_set(out_dir=args.out_dir)

    print("Next step: python 02_train_efe_vla_real.py "
          f"--data_dir {args.out_dir}/all_frames")
