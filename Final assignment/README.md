# Semantic Segmentation with EfficientNet-based Encoder

Hi :)

This repository provides a concise and reproducible implementation of a semantic segmentation experiment.  
The model used in this project is derived from the **EfficientNet** family and uses a pre-trained encoder as the feature-extraction backbone for semantic segmentation while equipping an untrained U-net decoder. 

EfficientNet is a family of convolutional neural networks designed to achieve strong performance while remaining computationally efficient. More information about the original EfficientNet implementation can be found [here](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

## Repository Overview

The purpose of this repository is to make the experiment easy to understand, reproduce, and evaluate.

In short, this project contains:

- A semantic segmentation model based on an EfficientNet encoder and U-net decoder;
- A `requirements.txt` file containing the Python dependencies;
- Training and/or inference scripts required to reproduce the experiment (e.g., Hyperparameter optimization/`train_hypo.py` or mere training/`train.py`);
- Supporting files needed to run the experiment in a slurm environment (e.g., `Dockerfile`);
- Additional test steps to ensure smoothless execution; 

## Prerequisites

Before running the project, make sure all required Python packages are installed in your local environment.
The dependency file is located under the `Perquisites` directory. From the root directory of the repository, run:

```bash
pip install -r Perquisites/requirements.txt

## Prerequisites

Before running the project, make sure all required Python packages are installed in your local environment.
The dependency file is located under the `Perquisites` directory. From the root directory of the repository, run:

```bash
pip install -r Perquisites/requirements.txt
```

Second, make sure you download the data as the second step describes in `Documents/README-Slurm.md` while factoring in a minor change. The training data is downloaded from exact same root but the container host and content (e.g., dependencies) changed so please use `Dockerfile_Slurm_Server` instead. When the right Dockerfile was put in place the following steps applies: 

```bash
chmod +x download_docker_and_data.sh
sbatch download_docker_and_data.sh
```

After the job finishes, you should see:

- a `data/` directory
- a `container.sif` file

> Note that we first add execution rights to the file to avoid any errors. You only have to do this once.
> For any other issues related to data download please check this [discussion](https://github.com/orgs/TUE-ARIA/discussions/62)
