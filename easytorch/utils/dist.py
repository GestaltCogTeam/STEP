import functools
import random
from typing import Tuple

import torch


# default master rank
MASTER_RANK = 0


def get_rank() -> int:
    """Get the rank of current process group.

    If DDP is initialized, return `torch.distributed.get_rank()`.
    Else return 0

    Returns:
        rank (int)
    """

    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def get_local_rank() -> int:
    """Get the local rank of current process group in multiple compute nodes.

    Returns:
        local_rank (int)
    """

    return get_rank() % torch.cuda.device_count() if torch.cuda.device_count() != 0 else 0


def get_world_size() -> int:
    """Get the number of processes in the current process group.

    If DDP is initialized, return ```torch.distributed.get_world_size()```.
    Else return 1

    Returns:
        world_size (int)
    """

    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def is_rank(rank: int) -> bool:
    """Checking if the rank of current process group is equal to ```rank```.

    Notes:
        ```rank``` must be less than ```world_size```

    Args:
        rank (int): rank

    Returns:
        result (bool)
    """

    if rank >= get_world_size():
        raise ValueError('Rank is out of range')

    return get_rank() == rank


def is_master() -> bool:
    """Checking if current process is master process.

    The rank of master process is ```MASTER_RANK```

    Returns:
        result (bool)
    """

    return is_rank(MASTER_RANK)


def master_only(func):
    """An function decorator that the function is only executed in the master process.

    Examples:
        @master_only
        def func(x):
            return 2 ** x

    Args:
        func: function

    Returns:
        wrapper func
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_master():
            return func(*args, **kwargs)

    return wrapper


def get_dist_backend(dist_node_num: int = 1, backend: str = None, init_method: str = None) -> Tuple[str, str]:
    """Get pytorch dist backend and init method.
    
    Note:
        Default `backend` is 'nccl', default `init_method` is 'tcp://127.0.0.1:{random port}'

    Args:
        backend (str): The backend to use.
        init_method (str): URL specifying how to initialize the process group.

    Returns:
        (backend, init_method)
    """

    backend = 'nccl' if backend is None else backend
    if init_method is None:
        if dist_node_num == 1:
            init_method = 'tcp://127.0.0.1:{:d}'.format(random.randint(50000, 65000))
        else:
            raise ValueError('The init_method cannot be None in multiple compute nodes')
    return backend, init_method
