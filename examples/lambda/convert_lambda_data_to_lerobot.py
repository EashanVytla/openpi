"""
Lambda (AI2-THOR) -> LeRobot for OpenPI/π0.x
- state_t  = [base_x, base_y, base_yaw(rad), ee_x, ee_y, ee_z, ee_roll, ee_pitch, ee_yaw, gripper]
- action_t = [dx, dy, dz, droll, dpitch, dyaw, dgrip]  (EE deltas scaled to target Δt; dgrip = grip_next - grip_now)
- Prints per-trajectory and overall Δ(sim_time) stats
"""

import shutil
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

REPO_NAME = "EashanVytla/lambda"

def main(
    data_dir: str,
    *,
    push_to_hub: bool = False,
    target_hz: float = 10.0,        # <-- pick your desired fixed rate (e.g., 10.0 or 5.0)
    dt_clip_mult: float = 5.0,      # <-- clip scale factor in [1/dt_clip_mult, dt_clip_mult] to avoid explosions
):
    raw_arg = data_dir
    data_path = Path(os.path.expandvars(data_dir)).expanduser()
    try:
        data_path = data_path.resolve(strict=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Data file not found.\n"
            f"  --data_dir argument: {raw_arg}\n"
            f"  Expanded & resolved: {data_path}\n"
        ) from e

    # Clean output dir if exists
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    target_dt = 1.0 / float(target_hz)
    eps = 1e-3  # 1 ms minimum to avoid division blow-ups

    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=int(round(target_hz)),   # advertise your intended fixed step
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (10,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    overall_dts = []  # Δ(sim_time) across all trajectories

    with h5py.File(str(data_path), "r") as hdf_file:
        for trajectory_name, trajectory_group in hdf_file.items():
            print(f"\nTrajectory: {trajectory_name}")

            # sort by numeric suffix to get the natural order
            step_items = sorted(trajectory_group.items(), key=lambda kv: int(kv[0].split("_")[-1]))

            sim_times = []

            for i, (timestep_name, timestep_group) in enumerate(step_items):
                try:
                    step_idx = int(timestep_name.split("_")[-1])
                except ValueError:
                    print(f"  Skipping {timestep_name}, no numeric suffix.")
                    continue

                # Read JSON
                raw_meta = timestep_group.attrs["metadata"]
                if isinstance(raw_meta, bytes):
                    raw_meta = raw_meta.decode("utf-8")
                meta = json.loads(raw_meta)

                steps = meta.get("steps", [])
                if not isinstance(steps, list) or len(steps) == 0:
                    print(f"  Skipping {timestep_name}: no steps[]")
                    continue
                step = steps[-1]  # if multiple updates at same ts, take last

                # sim_time for stats
                st = step.get("sim_time", None)
                if st is not None:
                    sim_times.append(float(st))

                # image (from this group)
                key = f"rgb_{step_idx}"
                if key not in timestep_group:
                    rgb_keys = [k for k in timestep_group.keys() if k.startswith("rgb_")]
                    if not rgb_keys:
                        print(f"  Skipping {timestep_name}: no RGB dataset found.")
                        continue
                    key = sorted(rgb_keys)[0]
                rgb_np = np.array(timestep_group[key])
                if rgb_np.dtype != np.uint8:
                    rgb_np = rgb_np.astype(np.uint8)
                if rgb_np.ndim != 3 or rgb_np.shape[2] not in (3, 4):
                    raise ValueError(f"Unexpected RGB shape {rgb_np.shape} at {trajectory_name}/{timestep_name}")
                if rgb_np.shape[2] == 4:
                    rgb_np = rgb_np[..., :3]
                rgb_image = Image.fromarray(rgb_np)
                if rgb_image.size != (256, 256):
                    rgb_image = rgb_image.resize((256, 256), Image.BILINEAR)

                # --- state_t (10-D) from current step ---
                base_x, base_y, _base_z, base_yaw_deg = step["global_state_body"]
                base_yaw = float(np.deg2rad(base_yaw_deg))
                ee_x, ee_y, ee_z, ee_roll, ee_pitch, ee_yaw = step["global_state_ee"]
                grip_now = 1.0 if step.get("held_objs", []) else 0.0

                state_vec = np.array(
                    [base_x, base_y, base_yaw, ee_x, ee_y, ee_z, ee_roll, ee_pitch, ee_yaw, grip_now],
                    dtype=np.float32,
                )

                # --- action_t (7-D): scale EE delta to target_dt, compute dgrip from next step ---
                # raw 6D delta (future - current)
                ee_delta = step.get("delta_global_state_ee", None)
                if not (isinstance(ee_delta, (list, tuple, np.ndarray)) and len(ee_delta) >= 6):
                    ee_delta = [0, 0, 0, 0, 0, 0]
                ee_delta = np.asarray(ee_delta[:6], dtype=np.float64)

                # compute dt from next step's sim_time if available
                if i + 1 < len(step_items):
                    nxt_name, nxt_group = step_items[i + 1]
                    nxt_raw = nxt_group.attrs["metadata"]
                    if isinstance(nxt_raw, bytes):
                        nxt_raw = nxt_raw.decode("utf-8")
                    nxt_meta = json.loads(nxt_raw)
                    nxt_steps = nxt_meta.get("steps", [])
                    nxt_step = nxt_steps[-1] if (isinstance(nxt_steps, list) and len(nxt_steps) > 0) else {}
                    st_next = nxt_step.get("sim_time", None)
                    # dgrip from next step
                    grip_next = 1.0 if nxt_step.get("held_objs", []) else 0.0
                else:
                    st_next = None
                    grip_next = grip_now

                # dt logic (robust to jitter / negatives / missing)
                if st is not None and st_next is not None:
                    dt = float(st_next) - float(st)
                else:
                    dt = target_dt  # fallback if missing

                if dt <= eps:
                    # skip pathological non-increasing timestamps by treating as target_dt
                    # (alternatively: continue to next frame to drop it)
                    dt = target_dt

                # scale factor toward target_dt
                scale = target_dt / dt
                # clip to avoid crazy scaling on outliers
                lo, hi = 1.0 / dt_clip_mult, dt_clip_mult
                if scale < lo or scale > hi:
                    scale = max(min(scale, hi), lo)

                ee_delta_scaled = (ee_delta * scale).astype(np.float32)

                dgrip = float(grip_next - grip_now)  # don't scale a binary signal
                action_vec = np.concatenate([ee_delta_scaled, np.array([dgrip], dtype=np.float32)], axis=0)

                task_text = meta.get("nl_command", "")

                dataset.add_frame(
                    {
                        "image": rgb_image,
                        "state": state_vec,
                        "actions": action_vec,
                        "task": task_text,
                    }
                )

            # per-trajectory Δ(sim_time) stats
            if len(sim_times) >= 2:
                dts = np.diff(np.array(sim_times, dtype=np.float64))
                if dts.size > 0:
                    overall_dts.extend(dts.tolist())
                    mean_dt = float(dts.mean())
                    std_dt = float(dts.std())
                    p10, p50, p90 = np.percentile(dts, [10, 50, 90]).tolist()
                    print(
                        "  Δ(sim_time) stats (s): "
                        f"mean={mean_dt:.4f}, std={std_dt:.4f}, "
                        f"p10={p10:.4f}, median={p50:.4f}, p90={p90:.4f}, "
                        f"min={dts.min():.4f}, max={dts.max():.4f}, n={dts.size}"
                    )
            else:
                print("  sim_time missing or insufficient to compute Δ.")

            # finalize episode
            dataset.save_episode()

    # overall Δ(sim_time) summary
    if len(overall_dts) > 0:
        dts = np.asarray(overall_dts, dtype=np.float64)
        mean_dt = float(dts.mean())
        std_dt = float(dts.std())
        p10, p50, p90 = np.percentile(dts, [10, 50, 90]).tolist()
        print(
            "\nOverall Δ(sim_time) stats (s): "
            f"mean={mean_dt:.4f}, std={std_dt:.4f}, "
            f"p10={p10:.4f}, median={p50:.4f}, p90={p90:.4f}, "
            f"min={dts.min():.4f}, max={dts.max():.4f}, n={dts.size}"
        )

    if push_to_hub:
        dataset.push_to_hub(
            tags=["lambda", "mobile-manipulator", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)
