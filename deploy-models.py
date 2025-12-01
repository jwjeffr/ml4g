from pathlib import Path
from typing import Optional, Callable

from ase import build, Atoms, io
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from tce.training import ClusterExpansion
from tce.calculator import TCECalculator, ASEProperty
from mace.calculators.mace import MACECalculator
import numpy as np


def equilibrate(
    initial_configuration: Atoms,
    beta: float,
    num_steps: int,
    save_every: int,
    calculator_constructor: Callable[[], Calculator],
    directory: Path,
    rng: Optional[np.random.Generator] = None,
) -> list[Atoms]:

    trajectory = []
    current_configuration = initial_configuration.copy()
    current_configuration.calc = calculator_constructor()
    for step in range(num_steps):

        if step % save_every == 0:
            print(f"simulation {step / num_steps:.2%} done")
            current_configuration.calc = SinglePointCalculator(current_configuration, energy=current_configuration.get_potential_energy())
            io.write(directory / f"frame_{step:.0f}.xyz", current_configuration, format="extxyz")

        # try to swap two atoms of differing types
        attempt = current_configuration.copy()
        attempt.calc = calculator_constructor()
        first_site, second_site = rng.integers(len(current_configuration), size=2)

        attempt.symbols[first_site], attempt.symbols[second_site] = attempt.symbols[second_site], attempt.symbols[first_site]

        # compute energy difference and accept or reject according to Metropolis criterion
        change_in_energy = attempt.get_potential_energy() - current_configuration.get_potential_energy()

        # always accept if change in energy is negative, sometimes accept if it's positive (escapes local minima)
        if np.exp(-beta * change_in_energy) > rng.random():
            current_configuration = attempt

    return trajectory


def main():

    ce_model = ClusterExpansion.load(Path("ce-model") / "IrPdPtRhRu.pkl")
    def ce_constructor() -> TCECalculator:

        return TCECalculator(cluster_expansions={ASEProperty.ENERGY: ce_model})
    
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

    rng = np.random.default_rng(seed=0)
    beta = 11.6
    initial_configuration = build.bulk(
        ce_model.type_map[0],
        crystalstructure=ce_model.cluster_basis.lattice_structure.name.lower(),
        a=ce_model.cluster_basis.lattice_parameter,
        cubic=True
    ).repeat((5, 5, 5))
    initial_configuration.symbols = rng.choice(ce_model.type_map, size=len(initial_configuration))

    ce_trajectory_dir = Path("ce-trajectory")
    ce_trajectory_dir.mkdir(parents=True, exist_ok=True)
    equilibrate(
        initial_configuration,
        beta,
        num_steps=50_000,
        save_every=100,
        calculator_constructor=ce_constructor,
        directory=ce_trajectory_dir,
        rng=rng,
    )

    mace_trajectory_dir = Path("mace-trajectory")
    mace_trajectory_dir.mkdir(parents=True, exist_ok=True)
    equilibrate(
        initial_configuration,
        beta,
        num_steps=50_000,
        save_every=100,
        calculator_constructor=mace_constructor,
        directory=mace_trajectory_dir,
        rng=rng
    )


if __name__ == "__main__":

    main()
