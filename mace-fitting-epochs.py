import json
import matplotlib.pyplot as plt


def main():

    losses = []
    rmses_per_atom = []
    with open("mace-model/dataset_run-42_train.txt", "r") as file:

        for line in file:
            data = json.loads(line)
            if not data["mode"] == "eval":
                continue
            losses.append(data["loss"])
            rmses_per_atom.append(1000.0 * data["rmse_e_per_atom"])

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    axs[0].plot(losses)
    axs[0].grid()
    axs[0].set_ylabel("loss")
    axs[1].plot(rmses_per_atom)
    axs[1].grid()
    axs[1].set_ylabel("RMSE per atom (meV / atom)")

    fig.supxlabel("epoch")
    fig.tight_layout()
    fig.savefig("figures/epochs.pdf", bbox_inches="tight")


if __name__ == "__main__":

    main()
