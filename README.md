# Image Segmentation with Low Memory Print and Fast EfficientNet Model

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## Introduction and Prerequisites
The project at hand is meant to explore root causes and effects of a trade-off portraying the distorted relationship between computational cost (steaming from A.I. indispensable development practices) and the gain/performance (expressed in terms of accuracy). This relentless pursuit of [state-of-the-art (SOTA) architectures](https://en.wikipedia.org/wiki/State_of_the_art) has introduced severe energy inefficiencies and diminishing returns. Furthermore, this resource-intensive trajectory yields prohibitively expensive models and hinders reproducibility by demanding extreme hardware infrastructure. 

Addressing these sustainability concerns, this work investigates the scaling dynamics of inference speed, model size, and accuracy against reduced [FLOPs](https://en.wikipedia.org/wiki/FLOPS), all of which are reflected in the research questions as shown below:

![Research Question 1](Final%20assignment/Results/RQ.png)
![Research Question 1.1](Final%20assignment/Results/RQ11.png)
![Research Question 1.2](Final%20assignment/Results/RQ12.png)

As prerequisites, a comprehensive [Cityscapes dataset](https://www.cityscapes-dataset.com/) has been used to aid/support development purposes. Furthermore, central to carrying out the development in a stable and fast fashion was the [HPC](https://www.surf.nl/diensten/rekenen/snellius-de-nationale-supercomputer) made readily available to us by the [SURF Snellius team](https://www.surf.nl/diensten/rekenen/snellius-de-nationale-supercomputer).

## Results
The best model (derived from the [EfficientNet](https://en.wikipedia.org/wiki/EfficientNet) family, equipping a pre-trained [encoder](https://en.wikipedia.org/wiki/Autoencoder) and a [U-Net](https://en.wikipedia.org/wiki/U-Net) as a decoder) is the representative mark to the feasible performance region that has been secured throughout this project. Here, the accuracy, the model's capacity, and its memory print are all "living in harmony" as shown in the experiment summary down below. 

![Experiments Summary](Final%20assignment/Results/ExperimentsSummary.png)

Not only did this work bring into existence a more "consumption aware" model, but it also shed light on the lacks that make accurately estimating the energy consumed throughout the development process particularly hardly achievable. 

As for the improvement of the model's performance, the entire course demonstrating the elevation from its baseline to the final model is depicted in the attachments down below.

### Baseline vs. Final Model Performance
![First Half Of The Results](Final%20assignment/Results/FirstHalfOfTheResults.png)
![Best Results From The Last Models](Final%20assignment/Results/BestResultsFromTheLastModels.png)

Tapping into this optimal zone became possible as a positive effect of having employed [data-augmentation](https://en.wikipedia.org/wiki/Data_augmentation) focused techniques and weighting-labels, as shown down below in the last two figures.

### Augmentation & Label Weighting 
![Effective Data Augmentation Techniques](Final%20assignment/Results/EffectiveDataAugmentatioTechniques.png)
![Weighting Table](Final%20assignment/Results/WeightingTable.png)
