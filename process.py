from pathlib import Path
from ase.io import read, write
import numpy as np
import random

# --- Configuration ---
DATASETS_DIR = Path("datasets")
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1  # remaining 0.1 = test
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def add_dummy_forces(extxyz_path: Path):
    atoms_list = read(extxyz_path, ":")
    modified = False
    for atoms in atoms_list:
        if "forces" not in atoms.arrays:
            atoms.arrays["forces"] = np.zeros((len(atoms), 3))
            modified = True

    write(
        extxyz_path,
        atoms_list,
        format="extxyz",
        columns=["symbols", "positions", "forces"],  # 🔑 ensure header includes forces:R:3
    )

    if modified:
        print(f"[+] Added and wrote dummy forces to {extxyz_path.name}")
    else:
        print(f"[✓] Forces already present in {extxyz_path.name}")


def split_and_write_dataset(dataset_path: Path):

    configurations = read(dataset_path, index=":", format="extxyz")
    print(configurations)

    assert len(configurations) > 0

    random.shuffle(configurations)
    n_total = len(configurations)
    n_train = int(n_total * TRAIN_RATIO)
    n_valid = int(n_total * VALID_RATIO)
    n_test = n_total - n_train - n_valid

    splits = {
        "train": configurations[:n_train],
        "valid": configurations[n_train:n_train + n_valid],
        "test": configurations[n_train + n_valid:],
    }

    for split, configurations in splits.items():

        out_path = Path("dataset") / f"{split}.extxyz"
        write(out_path, configurations, format="extxyz")
        print(f"[✓] Wrote {len(configurations)} configs to {out_path}")

        add_dummy_forces(out_path)


def main():

    split_and_write_dataset("dataset/noble-hea-surrogate.xyz")


if __name__ == "__main__":
    main()