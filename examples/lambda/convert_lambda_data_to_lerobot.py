"""
Convert Lambda (AI2-THOR) logs -> LeRobot format for OpenPI/π0.x

- state_t  = [base_x, base_y, base_yaw(rad), ee_x, ee_y, ee_z, ee_roll, ee_pitch, ee_yaw, gripper]
- action_t = [dx, dy, dz, droll, dpitch, dyaw, dgrip] = NEXT step's EE deltas + gripper delta
- fps is set to 10 (fixed Δt assumption for consumers); we also print actual Δ(sim_time) stats.

Usage:
uv run examples/lambda/convert_lambda_data_to_lerobot.py --data_dir /path/to/sim_dataset.hdf5
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

def main(data_dir: str, *, push_to_hub: bool = False):
    raw_arg = data_dir
    data_path = Path(os.path.expandvars(data_dir)).expanduser()
    try:
        data_path = data_path.resolve(strict=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Data file not found.\n"
            f"  --data_dir argument: {raw_arg}\n"
            f"  Expanded & resolved: {data_path}\n"
            f"Tip: if you quoted with ~ inside quotes, the shell may not expand it. Pass unquoted, or let this script expand."
        ) from e

    # Clean output dir if exists
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,  # fixed step assumption for downstream consumers
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {  # 10-D mobile-manipulator-aware proprio
                "dtype": "float32",
                "shape": (10,),
                "names": ["state"],
            },
            "actions": {  # 7-D (LIBERO-style) actions in EE delta + dgrip
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    with h5py.File(str(data_path), "r") as hdf_file:
        for trajectory_name, trajectory_group in hdf_file.items():
            print(f"\nTrajectory: {trajectory_name}")
            dataset.start_episode()

            # --- 1) Gather per-step buffers (sorted by numeric suffix) ---
            buffers = []  # each: {"image": PIL.Image, "step": dict, "meta": dict, "idx": int, "sim_time": float}
            for timestep_name, timestep_group in sorted(
                trajectory_group.items(), key=lambda kv: int(kv[0].split("_")[-1])
            ):
                try:
                    step_idx = int(timestep_name.split("_")[-1])
                except ValueError:
                    print(f"  Skipping {timestep_name}, no numeric suffix.")
                    continue

                raw_meta = timestep_group.attrs["metadata"]
                if isinstance(raw_meta, bytes):
                    raw_meta = raw_meta.decode("utf-8")
                meta = json.loads(raw_meta)

                # Many logs wrap the single update in a 1-length list
                steps = meta.get("steps", [])
                if not isinstance(steps, list) or len(steps) == 0:
                    print(f"  Skipping {timestep_name}: no steps[]")
                    continue
                # If multiple updates exist at the same timestamp, take the last one
                step = steps[-1]

                # Read RGB from this timestep group
                rgb_np = np.array(timestep_group[f"rgb_{step_idx}"])
                if rgb_np.dtype != np.uint8:
                    rgb_np = rgb_np.astype(np.uint8)
                if rgb_np.ndim != 3 or rgb_np.shape[2] not in (3, 4):
                    raise ValueError(f"Unexpected RGB shape {rgb_np.shape} at {trajectory_name}/{timestep_name}")
                rgb_image = Image.fromarray(rgb_np[..., :3])  # drop alpha if present
                if rgb_image.size != (256, 256):
                    rgb_image = rgb_image.resize((256, 256), Image.BILINEAR)

                sim_time = step.get("sim_time", None)  # may be missing on some steps
                buffers.append(
                    {"image": rgb_image, "step": step, "meta": meta, "idx": step_idx, "sim_time": sim_time}
                )

            if len(buffers) < 2:
                print("  Not enough steps for actions; skipping episode.")
                continue

            # --- 2) Emit frames: action_t uses NEXT step's deltas ---
            for i in range(len(buffers) - 1):
                cur = buffers[i]["step"]
                nxt = buffers[i + 1]["step"]

                # ----- state_t (10-D) from current step -----
                # base: [x, y, z, yaw_deg]
                base_x, base_y, _base_z, base_yaw_deg = cur["global_state_body"]
                base_yaw = float(np.deg2rad(base_yaw_deg))  # convert degrees -> radians (consistent with EE angles)
                # EE: [x, y, z, roll, pitch, yaw] (these look like radians already)
                ee_x, ee_y, ee_z, ee_roll, ee_pitch, ee_yaw = cur["global_state_ee"]

                grip_now = 1.0 if cur.get("held_objs", []) else 0.0

                state_vec = np.array(
                    [base_x, base_y, base_yaw, ee_x, ee_y, ee_z, ee_roll, ee_pitch, ee_yaw, grip_now],
                    dtype=np.float32,
                )

                # ----- action_t (7-D) from NEXT step's EE deltas + dgrip -----
                dx, dy, dz, droll, dpitch, dyaw = nxt.get("delta_global_state_ee", [0, 0, 0, 0, 0, 0])
                grip_next = 1.0 if nxt.get("held_objs", []) else 0.0
                dgrip = grip_next - grip_now

                action_vec = np.array([dx, dy, dz, droll, dpitch, dyaw, dgrip], dtype=np.float32)

                # ----- task text (constant per episode typically) -----
                task_text = buffers[i]["meta"].get("nl_command", "")

                dataset.add_frame(
                    {
                        "image": buffers[i]["image"],
                        "state": state_vec,
                        "actions": action_vec,
                        "task": task_text,
                    }
                )

            # --- 3) Print Δ(sim_time) stats to see sampling regularity ---
            sim_times = [b["sim_time"] for b in buffers if b["sim_time"] is not None]
            if len(sim_times) >= 2:
                sim_times = np.asarray(sim_times, dtype=np.float64)
                dts = np.diff(sim_times)
                if dts.size > 0:
                    mean_dt = float(dts.mean())
                    std_dt = float(dts.std())
                    min_dt = float(dts.min())
                    max_dt = float(dts.max())
                    p10, p50, p90 = np.percentile(dts, [10, 50, 90]).tolist()
                    print(
                        "  Δ(sim_time) stats (s): "
                        f"mean={mean_dt:.4f}, std={std_dt:.4f}, min={min_dt:.4f}, p10={p10:.4f}, "
                        f"median={p50:.4f}, p90={p90:.4f}, max={max_dt:.4f}"
                    )
                else:
                    print("  Only one valid sim_time; cannot compute Δ.")
            else:
                print("  sim_time missing or insufficient to compute Δ.")

            dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["lambda", "mobile-manipulator", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)
