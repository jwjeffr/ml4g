from pathlib import Path

from sklearn.preprocessing import StandardScaler
from tce.constants import ClusterBasis, LatticeStructure
from tce.training import train, get_type_map
from tce.topology import topological_feature_vector_factory
from ase import io
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
import numpy as np


def get_cluster_basis(lattice_parameter: float, lattice_structure: LatticeStructure) -> ClusterBasis:

    return ClusterBasis(
        lattice_parameter=lattice_parameter,
        lattice_structure=lattice_structure,
        max_adjacency_order=3,
        max_triplet_order=1
    )


def main():


    configurations_train = io.read("dataset/train.extxyz", index=":", format="extxyz") + io.read("dataset/valid.extxyz", index=":", format="extxyz")
    configurations_test = io.read("dataset/test.extxyz", index=":", format="extxyz")
    cluster_basis = get_cluster_basis(lattice_parameter=3.85, lattice_structure=LatticeStructure.FCC)

    types = get_type_map(configurations_train)
    feature_computer = topological_feature_vector_factory(
        basis=cluster_basis,
        type_map=types
    )

    ce = train(
        configurations=configurations_train,
        basis=cluster_basis,
        model=Pipeline([
            ("scale", StandardScaler()),
            ("fit", ElasticNetCV(random_state=0, max_iter=10_000))
        ]),
        feature_computer=feature_computer,
    )

    test_features = [feature_computer(configuration) for configuration in configurations_test]
    test_targets = [configuration.get_potential_energy() for configuration in configurations_test]

    score = ce.model.score(test_features, test_targets)
    errors = ce.model.predict(test_features) - test_targets
    rmse = np.sqrt(np.mean(errors ** 2)).item()
    print(f"cross-validated info: {score=}, {rmse=}")

    model_path = Path("ce-model") / f"{''.join(types)}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    ce.save(model_path)


if __name__ == "__main__":

    main()