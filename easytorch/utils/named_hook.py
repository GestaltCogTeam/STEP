from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch
from torch import nn


class NamedHook(metaclass=ABCMeta):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, module: nn.Module, *args, **kwargs):
        pass


class NamedForwardHook(NamedHook, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, module: nn.Module, inputs: Tuple[torch.Tensor], outputs: Tuple[torch.Tensor]):
        pass


class NamedBackwardHook(NamedHook, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, module: nn.Module, input_grads: Tuple[torch.Tensor], output_grads: Tuple[torch.Tensor]):
        pass
