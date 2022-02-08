import logging

import torch
from torch.utils.tensorboard import SummaryWriter


class AvgMeter(object):
    """Average meter.
    """

    def __init__(self):
        self._sum = 0.
        self._count = 0

    def reset(self):
        """Reset counter.
        """

        self._sum = 0.
        self._count = 0

    def update(self, value: float, n: int = 1):
        """Update sum and count.

        Args:
            value (float): value.
            n (int): number.
        """

        self._sum += value * n
        self._count += n

    @property
    def avg(self) -> float:
        """Get average value.

        Returns:
            avg (float)
        """

        return self._sum / self._count if self._count != 0 else 0


class MeterPool:
    """Meter container
    """

    def __init__(self):
        self._pool = {}

    def register(self, name: str, meter_type: str, fmt: str = '{:f}', plt: bool = True):
        """Init an average meter and add it to meter pool.

        Args:
            name (str): meter name (must be unique).
            meter_type (str): meter type.
            fmt (str): meter output format.
            plt (bool): set ```True``` to plot it in tensorboard
                when calling ```plt_meters```.
        """

        self._pool[name] = {
            'meter': AvgMeter(),
            'index': len(self._pool.keys()),
            'format': fmt,
            'type': meter_type,
            'plt': plt
        }

    def update(self, name: str, value: float):
        """Update average meter.

        Args:
            name (str): meter name.
            value (str): value.
        """

        self._pool[name]['meter'].update(value)

    def get_avg(self, name: str) -> float:
        """Get average value.

        Args:
            name (str): meter name.

        Returns:
            avg (float)
        """

        return self._pool[name]['meter'].avg

    def print_meters(self, meter_type: str, logger: logging.Logger = None):
        """Print the specified type of meters.

        Args:
            meter_type (str): meter type
            logger (logging.Logger): logger
        """

        print_list = []
        for i in range(len(self._pool.keys())):
            for name, value in self._pool.items():
                if value['index'] == i and value['type'] == meter_type:
                    print_list.append(
                        ('{}: ' + value['format']).format(name, value['meter'].avg)
                    )
        print_str = '{}:: [{}]'.format(meter_type, ', '.join(print_list))
        if logger is None:
            print(print_str)
        else:
            logger.info(print_str)

    def plt_meters(self, meter_type: str, step: int, tensorboard_writer: SummaryWriter):
        """Plot the specified type of meters in tensorboard.

        Args:
            meter_type (str): meter type.
            step (int): Global step value to record
            tensorboard_writer (SummaryWriter): tensorboard SummaryWriter
        """

        for name, value in self._pool.items():
            if value['plt'] and value['type'] == meter_type:
                tensorboard_writer.add_scalar(name, value['meter'].avg, global_step=step)
        tensorboard_writer.flush()

    def reset(self):
        """Reset all meters.
        """

        for _, value in self._pool.items():
            value['meter'].reset()


class MeterPoolDDP(MeterPool):
    # TODO(Yuhao Wang): not support

    def to_tensor(self):
        tensor = torch.empty((len(self._pool.keys()), 2))
        for i in range(len(self._pool.keys())):
            for _, value in self._pool.items():
                if value['index'] == i:
                    tensor[i][0] = float(value['meter'].count)
                    tensor[i][1] = value['meter'].avg
        return tensor

    def update_tensor(self, tensor):
        if tensor.shape[0] != len(self._pool.keys()):
            raise ValueError('Invalid tensor shape!')
        for i in range(len(self._pool.keys())):
            for _, value in self._pool.items():
                if value['index'] == i:
                    value['meter'].update(tensor[i][1], tensor[i][0])
