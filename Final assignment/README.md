# Semantic Segmentation with EfficientNet-based Encoder

Hi :)

This repository provides a concise and reproducible implementation of a semantic segmentation project.  
The model used in this project is derived from the **EfficientNet** family and uses a **pre-trained encoder as the feature-extraction backbone for semantic segmentation while equipping an untrained U-net decoder**. 

EfficientNet is a family of convolutional neural networks designed to achieve strong performance while remaining computationally efficient. More information about the original EfficientNet implementation can be found [here](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

## Repository Overview

The purpose of this repository is to make the experiment easy to understand, reproduce, and evaluate.

In short, this project contains:

- A semantic segmentation model based on an EfficientNet encoder and U-net decoder located in `model.py`;
- A `requirements.txt` file containing the Python dependencies;
- Training and/or inference scripts required to reproduce the experiment (e.g., Hyperparameter optimization/`train_hypo.py` or mere training/`train.py`);
- Supporting files needed to run the experiment in a slurm environment (e.g., `Dockerfile`);
- Additional test steps to ensure smoothless execution; 

## Prerequisites

**Before running the project**, make sure all required Python packages are installed in your local environment.
The dependency file is located under the `Perquisites` directory. From the root directory of the repository, run:

```bash
pip install -r Perquisites/requirements.txt
```

**Second**, make sure you download the data as the second step describes in `Documents/README-Slurm.md` while factoring in a minor change. The training data is downloaded from exact same root but the container host and content (e.g.,dependencies) changed so please use `Dockerfile_Slurm_Server.sh` instead. When the right Dockerfile was put in place the following steps applies: 

```bash
chmod +x Dockerfile_Slurm_Server.sh
sbatch Dockerfile_Slurm_Server.sh
```

After the job finishes, you should see:

- a `data/` directory
- a `container.sif` file

> Note that we first add execution rights to the file to avoid any errors. You only have to do this once.
> For any other issues related to data download please check this [discussion](https://github.com/orgs/TUE-ARIA/discussions/62)

## On-Server execution

Under this section, two on-server execution scenarios are covered:

- training the model
- conducting hyperparameter optimization

### Training

If one wants to only train the model, then follow the remaining steps from the third step onwards from `Documents/README-Slurm.md`.

Please note that `Documents/README-Slurm.md` covers everything apart from the CodeCarbon initialization.

In order to use the online-mode of CodeCarbon, the following are required:

- An API key
- The project name
- The experiment ID

The official CodeCarbon cloud/API documentation can be found here:

https://docs.codecarbon.io/latest/how-to/cloud-api/

First, create an account on the CodeCarbon dashboard:

https://dashboard.codecarbon.io/

Then authenticate the local/server environment by running:

```bash
codecarbon login
```

This command authenticates the environment, creates a default project, and stores the CodeCarbon credentials in a `.codecarbon.config` file.

If the API key is passed explicitly in the code, obtain it from the CodeCarbon dashboard/account settings and store it in a variable before starting the tracker.

The project name is the name under which the emissions of the current experiment will be grouped in the CodeCarbon dashboard. A project can be created or managed from the CodeCarbon dashboard.

The `experiment_id` identifies the specific experiment/run group where the emissions will be logged. This can also be created or obtained from the CodeCarbon dashboard.

After the API key, project name, and experiment ID have been obtained, store everything in variables and pass the arguments accordingly in `train.py`, under:

```python
# 9.1 CodeCarbon setting initialization and tracker start
```

For example:

```python
# 9.1 CodeCarbon setting initialization and tracker start

CODECARBON_PROJECT_NAME = "NNCV"
CODECARBON_API_KEY = "your_api_key_here"
CODECARBON_EXPERIMENT_ID = "a1ac9d91-e078-4104-a93e-df86c1b25b90"

tracker = EmissionsTracker(
    project_name=CODECARBON_PROJECT_NAME,
    measure_power_secs=15,
    api_key=CODECARBON_API_KEY,
    save_to_api=True,
    experiment_id=CODECARBON_EXPERIMENT_ID
)

tracker.start()
```

The tracker should be started before the training process begins so that the energy consumption and carbon emissions of the model training are recorded and sent to the CodeCarbon online dashboard.

### Hyperparameter optimization

If one is interested in conducting hyperparameter optimization, then replace the executable Python script in `main.sh` as shown below:

```bash
wandb login
python3 train.py \
```

with:

```bash
wandb login
python3 train_hypo.py \
    --data-dir ./dat
```

After replacing the executable Python script, proceed normally as described in `Documents/README-Slurm.md`.

You will use the `jobscript_slurm.sh` file to submit a job to the SLURM cluster. This script specifies the resources and command.

Submit the job with the following command:

```bash
chmod +x jobscript_slurm.sh
sbatch jobscript_slurm.sh
```

SLURM will queue and execute your job when resources are available.
