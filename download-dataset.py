from pathlib import Path

from tce.datasets import Dataset
from ase import io


def main():

    dataset = Dataset.from_dir(Path("noble_hea_surrogate"))
    io.write("dataset/noble-hea-surrogate.xyz", dataset.configurations, format="extxyz")


if __name__ == "__main__":

    main()
