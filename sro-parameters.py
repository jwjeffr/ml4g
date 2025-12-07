import re
from collections import defaultdict

from ovito import io
from cowley_sro_parameters import nearest_neighbor_topology, sro_modifier
import matplotlib.pyplot as plt


def main():

    # compute the Cowley short range order parameters for both simulations

    pipelines = {
        "CE": io.import_file("ce-trajectory/frame_*.xyz"),
        "MACE": io.import_file("mace-trajectory/frame_*.xyz")
    }

    fig, axs = plt.subplots(nrows=1, ncols=len(pipelines), sharex=True, sharey=True, figsize=(9, 4))

    for ax, (key, pipeline) in zip(axs, pipelines.items()):

        pipeline.modifiers.append(nearest_neighbor_topology(num_neighbors=12))
        pipeline.modifiers.append(sro_modifier())

        # the Pt ones are the interesting ones
        sros_with_platinum = defaultdict(list)
        for frame in pipeline.frames:
            for attribute in frame.attributes:
                if not attribute.startswith("sro_") or "Pt" not in attribute:
                    continue
                _, pair = attribute.split("_")
                first_type, second_type = re.findall(r'[A-Z][^A-Z]*', pair)
                if first_type != "Pt":
                    continue
                sros_with_platinum[first_type, second_type].append(frame.attributes[attribute])
        for pair, val in sros_with_platinum.items():
            ax.plot(val, label=pair[1])
        ax.grid()
        ax.set_title(key)

    axs[0].legend(title=r"atom type $\alpha$", loc="lower left")
    fig.supxlabel("frame")
    fig.supylabel(r"Pt-$\alpha$ Cowley SRO parameter")
    fig.tight_layout()
    fig.savefig("figures/sro.pdf", bbox_inches="tight")


if __name__ == "__main__":

    main()