from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..utils import get_rank, get_world_size


def build_data_loader(dataset: Dataset, data_cfg: dict):
    """Build dataloader from `data_cfg`
    `data_cfg` is part of config which defines fields about data, such as `CFG.TRAIN.DATA`

    structure of `data_cfg` is
    {
        'BATCH_SIZE': (int, optional) batch size of data loader (default: ``1``),
        'SHUFFLE': (bool, optional) data reshuffled option (default: ``False``),
        'NUM_WORKERS': (int, optional) num workers for data loader (default: ``0``),
        'PIN_MEMORY': (bool, optional) pin_memory option (default: ``False``),
        'PREFETCH': (bool, optional) set to ``True`` to use `BackgroundGenerator` (default: ``False``)
            need to install `prefetch_generator`, see https://pypi.org/project/prefetch_generator/
    }

    Args:
        dataset (Dataset): dataset defined by user
        data_cfg (dict): data config

    Returns:
        data loader
    """

    if data_cfg.get('PREFETCH', False):
        from ..utils.data_prefetcher import DataLoaderX
        return DataLoaderX(
            dataset,
            batch_size=data_cfg.get('BATCH_SIZE', 1),
            shuffle=data_cfg.get('SHUFFLE', False),
            num_workers=data_cfg.get('NUM_WORKERS', 0),
            pin_memory=data_cfg.get('PIN_MEMORY', False)
        )
    else:
        return DataLoader(
            dataset,
            batch_size=data_cfg.get('BATCH_SIZE', 1),
            shuffle=data_cfg.get('SHUFFLE', False),
            num_workers=data_cfg.get('NUM_WORKERS', 0),
            pin_memory=data_cfg.get('PIN_MEMORY', False)
        )


def build_data_loader_ddp(dataset: Dataset, data_cfg: dict):
    """Build ddp dataloader from `data_cfg`
    `data_cfg` is part of config which defines fields about data, such as `CFG.TRAIN.DATA`

    structure of `data_cfg` is
    {
        'BATCH_SIZE': (int, optional) batch size of data loader (default: ``1``),
        'SHUFFLE': (bool, optional) data reshuffled option (default: ``False``),
        'NUM_WORKERS': (int, optional) num workers for data loader (default: ``0``),
        'PIN_MEMORY': (bool, optional) pin_memory option (default: ``False``),
        'PREFETCH': (bool, optional) set to ``True`` to use `BackgroundGenerator` (default: ``False``)
            need to install `prefetch_generator`, see https://pypi.org/project/prefetch_generator/
    }

    Args:
        dataset (Dataset): dataset defined by user
        data_cfg (dict): data config

    Returns:
        data loader
    """

    ddp_sampler = DistributedSampler(
        dataset,
        get_world_size(),
        get_rank(),
        shuffle=data_cfg.get('SHUFFLE', False)
    )
    if data_cfg.get('PREFETCH', False):
        from ..utils.data_prefetcher import DataLoaderX
        return DataLoaderX(
            dataset,
            batch_size=data_cfg.get('BATCH_SIZE', 1),
            shuffle=False,
            sampler=ddp_sampler,
            num_workers=data_cfg.get('NUM_WORKERS', 0),
            pin_memory=data_cfg.get('PIN_MEMORY', False)
        )
    else:
        return DataLoader(
            dataset,
            batch_size=data_cfg.get('BATCH_SIZE', 1),
            shuffle=False,
            sampler=ddp_sampler,
            num_workers=data_cfg.get('NUM_WORKERS', 0),
            pin_memory=data_cfg.get('PIN_MEMORY', False)
        )
