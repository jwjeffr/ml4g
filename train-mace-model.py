#!/usr/bin/env python3
"""
Train MACE models on all processed TCE-LIB datasets (energy-only, safe configuration).
"""

import subprocess
from pathlib import Path


DATASETS_DIR = Path("dataset")
OUTPUT_DIR = Path("mace-model")

MODEL_TYPE = "MACE"
HIDDEN_IRREPS = "128x0e + 128x1o"
R_MAX = 5.0
MAX_EPOCHS = 100
BATCH_SIZE = 32
DEVICE = "cuda"
SEED = 42


def train_mace_on_dataset(dataset_path: Path):

    """
    train mace on the saved dataset
    mace mostly uses a CLI for this, so call a subprocess
    """
    
    train_file = dataset_path / "train.extxyz"
    valid_file = dataset_path / "valid.extxyz"
    test_file = dataset_path / "test.extxyz"

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1",
        "--module", "mace.cli.run_train",
        f"--name=dataset",
        f"--train_file={train_file}",
        f"--valid_file={valid_file}",
        f"--test_file={test_file}",
        "--model=MACE",
        "--hidden_irreps=64x0e + 64x1o",
        "--r_max=5.0",
        f"--max_num_epochs={MAX_EPOCHS}",
        "--batch_size=16",
        # "--distributed",  # keep disabled for now
        "--launcher=torchrun",
        "--ema",
        "--ema_decay=0.99",
        "--device=cuda",
        "--E0s=average",
        f"--seed={SEED}",
        "--energy_key=energy",
        "--compute_forces=True",   # <-- changed back to True
        "--forces_weight=0.0",     # still zeroed out
        "--stress_weight=0.0",
        "--loss=weighted",
        f"--model_dir={output_dir}",
        f"--log_dir={output_dir}",
        f"--results_dir={output_dir}",
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"[✓] Finished training, model saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[✗] Error training: {e}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    train_mace_on_dataset(Path("dataset"))


if __name__ == "__main__":
    main()