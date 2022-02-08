from torch import nn, optim
from torch.optim import lr_scheduler

from .. import easyoptim
from ..easyoptim import easy_lr_scheduler


def build_optim(optim_cfg: dict, model: nn.Module) -> optim.Optimizer:
    """Build optimizer from `optim_cfg`
    `optim_cfg` is part of config which defines fields about optimizer

    structure of `optim_cfg` is
    {
        'TYPE': (str) optimizer name, such as ``Adam``, ``SGD``,
        'PARAM': (dict) optimizer init params except first param `params`
    }

    Note:
        Optimizer is initialized by reflection, please ensure optim_cfg['TYPE'] is in `torch.optim`

    Examples:
        optim_cfg = {
            'TYPE': 'Adam',
            'PARAM': {
                'lr': 1e-3,
                'betas': (0.9, 0.99)
                'eps': 1e-8,
                'weight_decay': 0
            }
        }
        An `Adam` optimizer will be built.

    Args:
        optim_cfg (dict): optimizer config
        model (nn.Module): model defined by user

    Returns:
        optimizer (optim.Optimizer)
    """

    optim_type = optim_cfg['TYPE']
    if hasattr(optim, optim_type):
        Optim = getattr(optim, optim_type)
    else:
        Optim = getattr(easyoptim, optim_type)
    optim_param = optim_cfg['PARAM'].copy()
    optimizer = Optim(model.parameters(), **optim_param)
    return optimizer


def build_lr_scheduler(lr_scheduler_cfg: dict, optimizer: optim.Optimizer):
    """Build lr_scheduler from `lr_scheduler_cfg`
    `lr_scheduler_cfg` is part of config which defines fields about lr_scheduler

    structure of `lr_scheduler_cfg` is
    {
        'TYPE': (str) lr_scheduler name, such as ``MultiStepLR``, ``CosineAnnealingLR``,
        'PARAM': (dict) lr_scheduler init params except first param `optimizer`
    }

    Note:
        LRScheduler is initialized by reflection, please ensure
        lr_scheduler_cfg['TYPE'] is in `torch.optim.lr_scheduler` or `easytorch.easyoptim.easy_lr_scheduler`,
        if the `type` is not found in `torch.optim.lr_scheduler`,
        it will continue to be search in `easytorch.easyoptim.easy_lr_scheduler`

    Examples:
        lr_scheduler_cfg = {
            'TYPE': 'MultiStepLR',
            'PARAM': {
                'milestones': [100, 200, 300],
                'gamma': 0.1
            }
        }
        An `MultiStepLR` lr_scheduler will be built.

    Args:
        lr_scheduler_cfg (dict): lr_scheduler config
        optimizer (nn.Module): optimizer

    Returns:
        LRScheduler
    """

    lr_scheduler_type = lr_scheduler_cfg['TYPE']
    if hasattr(lr_scheduler, lr_scheduler_type):
        Scheduler = getattr(lr_scheduler, lr_scheduler_type)
    else:
        Scheduler = getattr(easy_lr_scheduler, lr_scheduler_type)
    scheduler_param = lr_scheduler_cfg['PARAM'].copy()
    scheduler_param['optimizer'] = optimizer
    scheduler = Scheduler(**scheduler_param)
    return scheduler
