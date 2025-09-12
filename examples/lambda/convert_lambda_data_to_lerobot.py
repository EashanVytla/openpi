"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
import h5py
import json
import numpy as np
from PIL import Image
from pathlib import Path
import os

REPO_NAME = "EashanVytla/lambda"  # Name of the output dataset, also used for the Hugging Face Hub

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
            f"Tips: If you quoted the path like --data_dir=\"~/file.hdf5\", "
            f"the shell won’t expand ~. Either omit the quotes or just pass any form "
            f"and let this script expand it."
        ) from e

    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


    with h5py.File(str(data_path), 'r') as hdf_file:
        for trajectory_name, trajectory_group in hdf_file.items():
            print(f"Trajectory: {trajectory_name}")
            # Iterate through each timestep group within the trajectory
            for index, (timestep_name, timestep_group) in enumerate(trajectory_group.items()):
                print(f"Step: {timestep_name}")

                # Read and decode the JSON metadata
                metadata = json.loads(timestep_group.attrs['metadata'])
                print(f"Metadata: {timestep_group.attrs['metadata']}")

                rgb_np = np.array(trajectory_group[timestep_name]['rgb_{}'.format(index)])
                rgb_image = Image.fromarray(rgb_np)

                # dataset.add_frame(
                #     {
                #         "image": rgb_image,
                #         "state": step["observation"]["state"],
                #         "actions": metadata["action"],
                #         "task": metadata["nl_command"],
                #     }
                # )
                # dataset.save_episode()


    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
