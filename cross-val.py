from pathlib import Path

from tce.training import ClusterExpansion
from tce.calculator import TCECalculator, ASEProperty
from mace.calculators.mace import MACECalculator
from ase import io
import numpy as np
import matplotlib.pyplot as plt


# need to define constructors for Calculator objects so we can calculate energies in ASE
# we define them as constructors so each ase.Atoms instance has a unique Calculator instance

# constructor for CE model
CE_MODEL = ClusterExpansion.load(Path("ce-model") / "IrPdPtRhRu.pkl")
def ce_constructor() -> TCECalculator:

    return TCECalculator(cluster_expansions={ASEProperty.ENERGY: CE_MODEL})


# constructor for MACE model
def mace_constructor() -> MACECalculator:
    model_dir = Path("mace-model")
    # Prefer compiled model if it exists
    compiled = model_dir / "dataset_compiled.model"
    regular = model_dir / "dataset.model"
    model_path = compiled if compiled.exists() else regular

    return MACECalculator(
        model_paths=[str(model_path)],   # ✅ new API; no deprecation warning
        device="cuda",
        default_dtype="float64",         # ✅ matches stored model dtype → no conversion spam
        dispersion=False,
        energy_key="energy",
        forces_key="forces",
    )


def main():

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(9, 4))

    # load in train, test, and validation configurations
    # we'll calculate predicted energies for these later
    # all are lists of Atoms objects
    train = io.read("dataset/train.extxyz", index=":", format="extxyz")
    train_energies_actual = np.fromiter((atoms.get_potential_energy() / len(atoms) for atoms in train), dtype=float)
    test = io.read("dataset/test.extxyz", index=":", format="extxyz")
    test_energies_actual = np.fromiter((atoms.get_potential_energy() / len(atoms) for atoms in test), dtype=float)
    validation = io.read("dataset/valid.extxyz", index=":", format="extxyz")
    validation_energies_actual = np.fromiter((atoms.get_potential_energy() / len(atoms) for atoms in validation), dtype=float)

    # compute energies predictef by CE model
    # we normalize by the number of atoms just to make the numbers nicer, but it doesn't change the stats at all
    train_energies_predicted_ce = np.zeros_like(train_energies_actual)
    for i, atoms in enumerate(train):
        atoms.calc = ce_constructor()
        train_energies_predicted_ce[i] = atoms.get_potential_energy() / len(atoms)
    test_energies_predicted_ce = np.zeros_like(test_energies_actual)
    for i, atoms in enumerate(test):
        atoms.calc = ce_constructor()
        test_energies_predicted_ce[i] = atoms.get_potential_energy() / len(atoms)
    validation_energies_predicted_ce = np.zeros_like(validation_energies_actual)
    for i, atoms in enumerate(validation):
        atoms.calc = ce_constructor()
        validation_energies_predicted_ce[i] = atoms.get_potential_energy() / len(atoms)

    # do the same for the mace model
    train_energies_predicted_mace = np.zeros_like(train_energies_actual)
    for i, atoms in enumerate(train):
        atoms.calc = mace_constructor()
        train_energies_predicted_mace[i] = atoms.get_potential_energy() / len(atoms)
    test_energies_predicted_mace = np.zeros_like(test_energies_actual)
    for i, atoms in enumerate(test):
        atoms.calc = mace_constructor()
        test_energies_predicted_mace[i] = atoms.get_potential_energy() / len(atoms)
    validation_energies_predicted_mace = np.zeros_like(validation_energies_actual)
    for i, atoms in enumerate(validation):
        atoms.calc = mace_constructor()
        validation_energies_predicted_mace[i] = atoms.get_potential_energy() / len(atoms)

    # make a parity plot for the CE model
    axs[0].scatter(train_energies_actual, train_energies_predicted_ce, edgecolor="black", label="train", zorder=6, alpha=0.6, marker="o")
    axs[0].scatter(test_energies_actual, test_energies_predicted_ce, edgecolor="black", label="test", zorder=7, alpha=0.6, marker="s")
    axs[0].scatter(validation_energies_actual, validation_energies_predicted_ce, edgecolor="black", label="validation", zorder=8, alpha=0.6, marker="h")
    axs[0].plot(axs[0].get_xlim(), axs[0].get_xlim(), color="black", linestyle="--")
    axs[0].set_title("CE")
    axs[0].legend()
    axs[0].grid()

    # do the same for the MACE model
    axs[1].scatter(train_energies_actual, train_energies_predicted_mace, edgecolor="black", label="train", zorder=6, alpha=0.6, marker="o")
    axs[1].scatter(test_energies_actual, test_energies_predicted_mace, edgecolor="black", label="test", zorder=7, alpha=0.6, marker="s")
    axs[1].scatter(validation_energies_actual, validation_energies_predicted_mace, edgecolor="black", label="validation", zorder=8, alpha=0.6, marker="h")
    axs[1].plot(axs[1].get_xlim(), axs[1].get_xlim(), color="black", linestyle="--")
    axs[1].set_title("MACE")
    axs[1].grid()

    fig.supxlabel("actual energy (eV / atom)")
    fig.supylabel("predicted energy (eV / atom)")
    fig.tight_layout()
    fig.savefig("figures/cross-val.pdf", bbox_inches="tight")


if __name__ == "__main__":

    # MACE seems to change font size upon import (ugh...)
    plt.rcParams["font.size"] = 10
    main()
