import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="CIFAR10", choices=["CIFAR10"], help="Name of dataset to train on")
    parser.add_argument('--dataset_location', type=str, default="./data", help="Location to download datasets to")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size used for dataloaders")
    parser.add_argument('--epochs', type=int, default=2, help="Number of epochs for training")

    parser.add_argument('--optimizer_type', type=str, default='adam', choices=['adam', 'radam', 'adamw', 'sgd'], help="Optimizer to use for training")
    parser.add_argument('--lr_schedule_type', type=str, default='cos', choices=['exp', 'cos', 'multistep'], help="Scheduler for optimizer")
    parser.add_argument('--loss_type', type=str, default='cross_entropy', choices=["cross_entropy"], help="Loss criterion for training")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for optimizer")


    # parser.add_argument("--bool_arg", type=bool, default=True, help="")
    # parser.add_argument('--flag_arg', action='store_true', help="")
    # parser.add_argument('--int_arg', type=int, default=0, help="")
    # parser.add_argument('--float_arg', type=float, default=0.0, help="")
    # parser.add_argument('--str_arg', type=str, default='', choices=['', '', ''], help="")

    args = parser.parse_args()
    return args
