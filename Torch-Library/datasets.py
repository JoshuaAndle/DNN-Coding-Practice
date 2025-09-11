import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


### Prepare a specified dataset, downloading to the specified location
def prepare_dataset(dataset_name, location="./data", train=True):
    ### Not needed for torchvision downloads but just to be safe in case of other dataset use in the future
    if os.path.isdir(location) == False:
        os.makedirs(location)



    if dataset_name == "CIFAR10":
        mean=[0.491, 0.482, 0.446]
        std=[0.247, 0.243, 0.261]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
        dataset = torchvision.datasets.CIFAR10(root=location, train=train, transform=transform, download=True)

    else:
        print("Error: No valid dataset implemented for dataset name {} in datasets.py".format(dataset_name))

    return dataset


### Prepare a dataloader for a given dataset object
def prepare_dataloader(dataset, batch_size, train=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader












