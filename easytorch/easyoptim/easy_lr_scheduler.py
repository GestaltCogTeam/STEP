import math
import warnings

from torch.optim.lr_scheduler import _LRScheduler


class MultiCosineAnnealingWarmupLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \lr_mult \times \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        final_epoch (int): Number of total iterations
        T_0 (list): Number of iterations for the restart
        lr_mult (list): A factor multiplied with learning rate at iteration T_0, must have the same shape as T_0
        warmup_begin (int, optional): Number of iterations for the beginning warm up, notice that the first decay T_mult will be reduced by this param. Default: 0
        warmup_factor (float, optional): A factor that the learning rate will be multiplied at first epoch. Default: 0.01
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.


    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """
    def __init__(self, optimizer, final_epoch, T_0=None, lr_mult=None, warmup_begin=0, warmup_factor=0.01, eta_min=0, last_epoch=-1, verbose=False):
        if T_0 and not isinstance(T_0, list):
            raise ValueError("Expected list object or None type T_0, but got {}".format(type(T_0)))
        if lr_mult and not isinstance(lr_mult, list):
            raise ValueError("Expected list object or None type lr_mult, bug got {}".format(type(lr_mult)))
        if not T_0 and not lr_mult:
            raise ValueError("Expected T_0 and lr_mult has the same length, but got NoneType and {}".format(len(lr_mult)))
        if T_0 and not lr_mult:
            raise ValueError("Expected T_0 and lr_mult has the same length, but got {} and NoneType".format(len(T_0)))
        if T_0 and lr_mult and len(T_0) != len(lr_mult):
                raise ValueError("Expected T_0 and lr_mult has the same length, but got {} and {}".format(len(T_0), len(lr_mult)))

        self.final_epoch = final_epoch
        self.T_0_list = T_0 if T_0 else []
        self.lr_mult_list = lr_mult if lr_mult else []
        self.warmup_begin = warmup_begin
        self.warmup_factor = warmup_factor
        self.eta_min = eta_min

        # add number at beggining
        self.T_0_list.insert(0, 0)
        self.lr_mult_list.insert(0, 1)

        # calculate T_i accroding to given T_0 list
        self.T_0_expand= self.T_0_list.copy()
        self.T_0_expand.append(final_epoch)

        if self.warmup_begin > self.T_0_expand[1]:
            raise ValueError("the warmup_begin iteration is bigger than the first T_i, please use smaller warmup_begin or bigger T_0[0]")

        self.T_i_list = [self.T_0_expand[i+1] - self.T_0_expand[i] - 1 for i in range(len(self.T_0_expand)-1)]
        self.T_i_list[0] = self.T_i_list[0] - self.warmup_begin # subtract warmup at beginning


        # initial T_i, lr_mult and T_cur
        self.T_i = self.T_i_list[0]
        self.lr_mult = self.lr_mult_list[0]
        self.T_cur = 0

        super(MultiCosineAnnealingWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch <= self.warmup_begin:
            if self.warmup_begin != 0:
                lr = [(base_lr - self.warmup_factor * base_lr) * (self.last_epoch/self.warmup_begin) + self.warmup_factor * base_lr
                      for base_lr in self.base_lrs]
            else:
                lr = [base_lr for base_lr in self.base_lrs]
        else:
            lr = [self.eta_min + (self.lr_mult * base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                    for base_lr in self.base_lrs]

        return lr

    def step(self, epoch=None):
        """Step could be called after every batch update

        Example:
            >>> T_0 = [20, 30, 50]
            >>> lr_mult = [0.8, 0.6, 0.5]
            >>> scheduler = MultiCosineAnnealingWarmupLR(optimizer, final_epoch=150, T_0, lr_mult, warmup_begin=5, warmup_factor=0.001, eta_min=1e-7)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>     scheduler.step()
        """

        def locate(self, epoch: int):
            """return the number of which section dose T_0 locate in T_0_list
            """
            for i in range(1, len(self.T_0_list)):
                if epoch == 0:
                    return 0
                elif epoch >= self.T_0_list[i-1] and epoch < self.T_0_list[i]:
                    return i-1
                else:
                    continue
            return len(self.T_0_list) - 1


        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            section = locate(self, epoch) # the section of the T_0_list
            self.T_0 = self.T_0_list[section]
            self.T_i = self.T_i_list[section]
            self.T_cur = epoch - self.T_0
            if epoch < self.T_0_expand[1]:
                self.T_cur = self.T_cur - self.warmup_begin # subtract warmup_begin at first section
            self.lr_mult = self.lr_mult_list[section]
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            section = locate(self, epoch)
            self.T_0 = self.T_0_list[section]
            self.T_i = self.T_i_list[section]
            if epoch == 0:
                self.T_cur = 0
            else:
                self.T_cur = epoch - self.T_0 - 1
            if epoch < self.T_0_expand[1]:
                self.T_cur = self.T_cur - self.warmup_begin
            self.lr_mult = self.lr_mult_list[section]

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
