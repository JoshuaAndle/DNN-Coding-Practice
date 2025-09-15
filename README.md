# Toy Project for Practicing Pytorch

Project implements a minimal version of the CLIP model with one transformer layer in each branch.
Two directories are included:
1. Torch-Library which implements the model using existing torch classes
2. Torch-Scratch which implements layers (e.g. Conv2d, Linear, MultiHead Attention), optimizers, schedulers, and loss functions in Pytorch from scratch.

This project is mainly for coding practice, so there is a fair amount of missing polish/features, but the aim is to ensure the accuracy obtained by Torch-Scratch/main.py matches Torch-Library/main.py
Current accuracy for the first two epochs is ~30-40% for Torch-Library and ~30% for Torch-Scratch, likely due to slight differences such as initialization.
