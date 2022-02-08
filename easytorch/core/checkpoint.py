import os
import glob
from logging import Logger

import torch
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

from ..utils import get_logger, get_local_rank

DEFAULT_LOGGER = get_logger('easytorch-checkpoint')


def get_ckpt_dict(model: nn.Module, optimizer: optim.Optimizer, epoch: int) -> dict:
    """Generate checkpoint dict.
    checkpoint dict format:
    {
        'epoch': current epoch ([1, num_epochs]),
        'model_state_dict': state_dict of model,
        'optim_state_dict': state_dict of optimizer
    }
    if model is a module wrapper, use `model.module`

    Args:
        model (nn.Module): the model to be saved
        optimizer (optim.Optimizer): the optimizer to be saved
        epoch: current epoch

    Returns:
        checkpoint dict (dict): generated checkpoint dict
    """

    if isinstance(model, DDP):
        _model = model.module
    else:
        _model = model
    return {
        'epoch': epoch,
        'model_state_dict': _model.state_dict(),
        'optim_state_dict': optimizer.state_dict()
    }


def get_last_ckpt_path(ckpt_save_dir: str, name_pattern: str = '*.pt') -> str:
    """Get last checkpoint path in `ckpt_save_dir`
    checkpoint files will be sorted by name

    Args:
        ckpt_save_dir (str): checkpoint save directory
        name_pattern (str): name pattern for checkpoint file, default is '*.pt'

    Returns:
        checkpoint path (str): last checkpoint path in `ckpt_save_dir`
    """

    ckpt_list = glob.glob(os.path.join(ckpt_save_dir, name_pattern))
    ckpt_list.sort()
    return ckpt_list[-1]


def load_ckpt(ckpt_save_dir: str, ckpt_path: str = None, use_gpu: bool = True, logger: Logger = DEFAULT_LOGGER) -> dict:
    """Load checkpoint
    if param `ckpt_path` is None, load the last checkpoint in `ckpt_save_dir`,
    else load checkpoint from `ckpt_path`

    Args:
        ckpt_save_dir (str): checkpoint save directory
        ckpt_path (str): checkpoint path, default is None
        use_gpu (bool): set to ``True`` to load checkpoint to GPU
        logger (Logger): logger, default is Logger('easytorch')

    Returns:
        checkpoint dict loaded from file
    """

    if ckpt_path is None:
        ckpt_path = get_last_ckpt_path(ckpt_save_dir)
    if use_gpu:
        map_location = 'cuda:{}'.format(get_local_rank())
    else:
        map_location = 'cpu'
    logger.info('load ckpt from \'{}\''.format(ckpt_path))
    return torch.load(ckpt_path, map_location=map_location)


def save_ckpt(ckpt: dict, ckpt_path: str, logger: Logger = DEFAULT_LOGGER):
    """Save checkpoint

    Args:
        ckpt (dict): saved checkpoint dict
        ckpt_path (str): checkpoint save path
        logger (Logger): logger, default is Logger('easytorch')
    """

    torch.save(ckpt, ckpt_path)
    logger.info('ckpt {} saved'.format(ckpt_path))


def need_to_remove_last_ckpt(last_epoch: int, ckpt_save_strategy: int or list or tuple) -> bool:
    """Judging whether to remove last checkpoint by `ckpt_save_strategy`

    `ckpt_save_strategy` should be None, an int value, a list or a tuple
    if `ckpt_save_strategy` is None, remove last checkpoint file every epoch
    if `ckpt_save_strategy` is an int value `n`, save checkpoint every n epoch,
        remove last checkpoint file when last_epoch % ckpt_save_strategy != 0
    if `ckpt_save_strategy` is a list or a tuple `l`, save checkpoint when epoch in `l`,
        remove last checkpoint file when last_epoch not in ckpt_save_strategy

    Args:
        last_epoch (int): last epoch num
        ckpt_save_strategy (int or list or tuple): checkpoint save strategy

    Returns:
        last checkpoint delete flag (bool): `True` means delete last checkpoint
    """

    if ckpt_save_strategy is None:
        return True
    elif isinstance(ckpt_save_strategy, int) and last_epoch % ckpt_save_strategy != 0:
        return True
    elif isinstance(ckpt_save_strategy, (list, tuple)) and last_epoch not in ckpt_save_strategy:
        return True
    else:
        return False


def backup_last_ckpt(last_ckpt_path: str, epoch: int, ckpt_save_strategy: int or list or tuple):
    """Backup last checkpoint when last checkpoint needs to be removed (by call need_to_remove_last_ckpt())
    if last checkpoint file name is `a.pt`, rename `a.pt` to `a.pt.bak`

    Args:
        last_ckpt_path (str): last checkpoint file path
        epoch (int): current epoch num
        ckpt_save_strategy (int or list or tuple): checkpoint save strategy
    """

    last_epoch = epoch - 1

    # rename last ckpt to .bak
    if need_to_remove_last_ckpt(last_epoch, ckpt_save_strategy) and last_epoch != 0:
        os.rename(last_ckpt_path, last_ckpt_path + '.bak')


def clear_ckpt(ckpt_save_dir: str, name_pattern: str = '*.pt.bak'):
    """Clear all backed up checkpoint files

    Args:
        ckpt_save_dir (str): checkpoint save directory
        name_pattern (str): backed up checkpoint file name pattern
    """

    ckpt_list = glob.glob(os.path.join(ckpt_save_dir, name_pattern))
    for ckpt in ckpt_list:
        os.remove(ckpt)
