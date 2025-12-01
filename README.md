# ml4g

Install the necessary packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download and pre-process the dataset (need to add forces for MACE):

```bash
python download-dataset.py
python process.py
```

Visualize the dataset (need to use `xvfb-run` on Palmetto to run OVITO in headless mode):

```bash
xvfb-run python dataset-vis.py
```

Train the CE and MACE models on the dataset:

```bash
python train-ce-model.py
python train-mace-model.py
```

Compute cross-validation info and loss curve for MACE:

```bash
python cross-val.py
python mace-fitting-epochs.py
```

Deploy the models in a Metropolis-Hastings simulation using a script:

```bash
python deploy-models.py
```

or using a slurm job:

```bash
sbatch deploy-models.slurm
```

and, finally, compute the SRO parameters along the trajectories:

```bash
python sro-parameters.py
```
