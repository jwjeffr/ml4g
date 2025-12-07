from pathlib import Path
import os
os.environ["OVITO_GUI_MODE"] = "1"

from ovito.pipeline import StaticSource, Pipeline
from ovito.io.ase import ase_to_ovito
from ovito.vis import ColorLegendOverlay, Viewport
from ovito.qt_compat import QtCore
import numpy as np
from scipy.stats import qmc
from ase import io
import matplotlib.pyplot as plt


def main():

    # load in all configurations, we'll visualize these with ovito
    # each io.read call returns a list of Atoms objects, so just sum them
    configurations = (
        io.read("dataset/test.extxyz", index=":", format="extxyz") +
        io.read("dataset/train.extxyz", index=":", format="extxyz") +
        io.read("dataset/valid.extxyz", index=":", format="extxyz")
    )

    num_configs = len(configurations)

    # generate points using Poisson sampling, makes sure there is a specific amount of space between each configuration
    engine = qmc.PoissonDisk(d=2, radius=0.01, rng=np.random.default_rng(seed=0))
    sample = engine.random(num_configs)
    sample *= 2000.0
    points = np.column_stack((*sample.T, np.zeros(num_configs)))

    # convert ase configurations to ovito objects and then add them to the scene according to the sampled points above
    # these are defined by Rodrigues vectors
    axis = np.array([1, -1, 0])
    axis = axis / np.linalg.norm(axis) * 30 * np.pi / 180
    pipeline = None
    for point, configuration in zip(points, configurations):
        data = ase_to_ovito(configuration)
        pipeline = Pipeline(source=StaticSource(data=data))
        pipeline.add_to_scene(translation=point, rotation=axis)
    if not pipeline:
        raise ValueError

    # initialize the viewport and render it
    vp = Viewport()
    legend = ColorLegendOverlay(
        property="particles/Particle Type",
        alignment=QtCore.Qt.AlignmentFlag.AlignBottom | QtCore.Qt.AlignmentFlag.AlignLeft,
        orientation=QtCore.Qt.Orientation.Horizontal,
        title=" ",
        pipeline=pipeline,
        font_size=0.15,
        border_enabled=True
    )
    vp.overlays.append(legend)
    size = (2000, 2000)
    vp.zoom_all(size)
    vp.render_image(filename="figures/visualized.png", size=size)

    # plot a histogram of the energies
    energies = np.fromiter(
        (atoms.get_potential_energy() / len(atoms) for atoms in configurations),
        dtype=float
    )
    plt.hist(energies, linewidth=1.0, zorder=6, edgecolor="black")
    plt.grid()
    plt.xlabel("energy (eV / atom)")
    plt.ylabel("counts")
    plt.tight_layout()
    plt.savefig("figures/dataset-energies.pdf", bbox_inches="tight")


if __name__ == "__main__":

    plt.rcParams["font.size"] = 15
    main()