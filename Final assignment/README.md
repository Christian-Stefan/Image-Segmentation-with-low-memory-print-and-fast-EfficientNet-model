# Semantic Segmentation with EfficientNet-based Encoder

Hi :)

This repository provides a concise and reproducible implementation of a semantic segmentation experiment.  
The model used in this project is derived from the **EfficientNet** family and uses a pre-trained encoder as the feature-extraction backbone for semantic segmentation.

EfficientNet is a family of convolutional neural networks designed to achieve strong performance while remaining computationally efficient. More information about the original EfficientNet implementation can be found here:

[EfficientNet - TensorFlow TPU GitHub Repository](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

## Repository Overview

The purpose of this repository is to make the experiment easy to understand, reproduce, and evaluate.

In short, this project contains:

- A semantic segmentation model based on an EfficientNet encoder
- Training and/or inference scripts required to reproduce the experiment
- A `requirements.txt` file containing the Python dependencies
- Supporting files needed to run the experiment in a local environment

## Prerequisites

Before running the project, make sure all required Python packages are installed in your local environment.

The dependency file is located under the `Perquisites` directory.

From the root directory of the repository, run:

```bash
pip install -r Perquisites/requirements.txt
