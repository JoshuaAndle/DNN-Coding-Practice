### Package Libraries
import torch
import torchvision
import torch.nn as nn


### Project Libraries
import configs
import datasets


def main():
    args = configs.args_parser()

    ### Set up dataset 
    ### Note: Not worrying about validation split in this toy project
    train_dataset = datasets.prepare_dataset(args.dataset, location=args.dataset_location, train=True)
    test_dataset = datasets.prepare_dataset(args.dataset, location=args.dataset_location, train=False)

    train_dataloader = datasets.prepare_dataloader(train_dataset, batch_size=args.batch_size, train=True)
    test_dataloader = datasets.prepare_dataloader(test_dataset, batch_size=args.batch_size, train=False)


    ### Set up model


    ### Set up optimizer

    ### Set up loss function

    ### Set up learning rate scheduler

    ### Apply training loop





    print("main() run complete")














if __name__ == '__main__':
    main()