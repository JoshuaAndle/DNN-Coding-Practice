"""
 Settings taken from github repository (seems to have been privated, so no URL) for:
@article{wang2024clip,
  title={CLIP model is an Efficient Online Lifelong Learner},
  author={Wang, Leyuan and Xiang, Liuyu and Wei, Yujie and Wang, Yunlong and He, Zhaofeng},
  journal={arXiv preprint arXiv:2405.15155},
  year={2024}
}
"""

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

### Set up optimizer
def Get_Optimizer(model, optimizer_type, lr):

    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type == "adam":
        opt = optim.Adam(params, lr=lr, weight_decay=0)
    elif optimizer_type == "adamw":
        opt = optim.AdamW(params, lr=lr, weight_decay=0.00001)
    elif optimizer_type == "sgd":
        opt = optim.SGD(params, lr=lr, weight_decay=1e-4)
    else:
        raise NotImplementedError("Please select the optimizer_type [adam, sgd]")
    return opt



### Set up learning rate scheduler
def Get_LR_Scheduler(lr_schedule_type, n_epochs, opt):
    if lr_schedule_type == "exp":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=1/1.1)
    elif lr_schedule_type == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,
                                                                   T_0=1,
                                                                   T_mult=2)
    elif lr_schedule_type == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(opt,
                                                   milestones=[n_epochs//4, n_epochs//3, n_epochs//2],
                                                   gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)

    return scheduler


### Set up loss function
def Get_Loss_Criterion(loss_type):
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(reduction="mean")


