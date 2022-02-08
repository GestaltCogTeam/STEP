import os
import time
import logging
from abc import ABCMeta, abstractmethod

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from .meter_pool import MeterPool
from .checkpoint import get_ckpt_dict, load_ckpt, save_ckpt, backup_last_ckpt, clear_ckpt
from .data_loader import build_data_loader, build_data_loader_ddp
from .optimizer_builder import build_optim, build_lr_scheduler
from ..utils import TimePredictor, get_logger, get_rank, get_local_rank, is_master, master_only, setup_random_seed


class Runner(metaclass=ABCMeta):
    def __init__(self, cfg: dict, use_gpu: bool = True):
        # default logger
        self.logger = get_logger('easytorch')

        # setup random seed
        # each rank has different seed in distributed mode
        self.seed = cfg.get('SEED')
        if self.seed is not None:
            setup_random_seed(
                self.seed + get_rank(),
                deterministic=cfg.get('DETERMINISTIC', True),
                cudnn_enabled=cfg.get('CUDNN_ENABLED', False),
                cudnn_benchmark=cfg.get('CUDNN_BENCHMARK', False)
            )

        # param
        self.use_gpu = use_gpu
        self.model_name = cfg['MODEL']['NAME']
        self.ckpt_save_dir = cfg['TRAIN']['CKPT_SAVE_DIR']
        self.logger.info('ckpt save dir: \'{}\''.format(self.ckpt_save_dir))
        self.ckpt_save_strategy = None
        self.num_epochs = None
        self.start_epoch = None

        self.val_interval = 1
        self.test_interval = 1

        # create model
        self.model = self.build_model(cfg)

        # declare optimizer and lr_scheduler
        self.optim = None
        self.scheduler = None

        # declare data loader
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        # declare meter pool
        self.meter_pool = None

        # declare tensorboard_writer
        self.tensorboard_writer = None

    def init_logger(self, logger: logging.Logger = None, logger_name: str = None,
                    log_file_name: str = None, log_level: int = logging.INFO):
        """Initialize logger.

        Args:
            logger (logging.Logger, optional): specified logger.
            logger_name (str, optional): specified name of logger.
            log_file_name (str, optional): logger file name.
            log_level (int, optional): log level, default is INFO.
        """

        if logger is not None:
            self.logger = logger
        elif logger_name is not None:
            if log_file_name is not None:
                log_file_name = '{}_{}.log'.format(log_file_name, time.strftime("%Y%m%d%H%M%S", time.localtime()))
                log_file_path = os.path.join(self.ckpt_save_dir, log_file_name)
            else:
                log_file_path = None
            self.logger = get_logger(logger_name, log_file_path, log_level)
        else:
            raise TypeError('At least one of logger and logger_name is not None')

    def to_running_device(self, src: torch.Tensor or torch.Module) -> torch.Tensor or torch.Module:
        """Move `src` to the running device. If `self.use_gpu` is ```True```,
        the running device is GPU, else the running device is CPU.

        Args:
            src (torch.Tensor or torch.Module): source

        Returns:
            target (torch.Tensor or torch.Module)
        """

        if self.use_gpu:
            return src.cuda()
        else:
            return src.cpu()

    @staticmethod
    @abstractmethod
    def define_model(cfg: dict) -> nn.Module:
        """It must be implement to define the model for training or inference.

        Users can select different models by param in cfg.
        
        Args:
            cfg (dict): config

        Returns:
            model (nn.Module)
        """

        pass

    @staticmethod
    @abstractmethod
    def build_train_dataset(cfg: dict) -> Dataset:
        """It must be implement to build dataset for training.

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """

        pass

    @staticmethod
    def build_val_dataset(cfg: dict):
        """It can be implement to build dataset for validation (not necessary).

        Args:
            cfg (dict): config

        Returns:
            val dataset (Dataset)
        """

        raise NotImplementedError()

    @staticmethod
    def build_test_dataset(cfg: dict):
        """It can be implement to build dataset for validation (not necessary).

        Args:
            cfg (dict): config

        Returns:
            val dataset (Dataset)
        """

        raise NotImplementedError()

    def build_train_data_loader(self, cfg: dict) -> DataLoader:
        """Build train dataset and dataloader.
        Build dataset by calling ```self.build_train_dataset```,
        build dataloader by calling ```build_data_loader``` or
        ```build_data_loader_ddp``` when DDP is initialized

        Args:
            cfg (dict): config

        Returns:
            train data loader (DataLoader)
        """

        dataset = self.build_train_dataset(cfg)
        if torch.distributed.is_initialized():
            return build_data_loader_ddp(dataset, cfg['TRAIN']['DATA'])
        else:
            return build_data_loader(dataset, cfg['TRAIN']['DATA'])

    def build_val_data_loader(self, cfg: dict) -> DataLoader:
        """Build val dataset and dataloader.
        Build dataset by calling ```self.build_train_dataset```,
        build dataloader by calling ```build_data_loader```.

        Args:
            cfg (dict): config

        Returns:
            val data loader (DataLoader)
        """

        dataset = self.build_val_dataset(cfg)
        return build_data_loader(dataset, cfg['VAL']['DATA'])

    def build_test_data_loader(self, cfg: dict) -> DataLoader:
        """Build val dataset and dataloader.
        Build dataset by calling ```self.build_train_dataset```,
        build dataloader by calling ```build_data_loader```.

        Args:
            cfg (dict): config

        Returns:
            val data loader (DataLoader)
        """

        dataset = self.build_test_dataset(cfg)
        return build_data_loader(dataset, cfg['TEST']['DATA'])

    def build_model(self, cfg: dict) -> nn.Module:
        """Build model.

        Initialize model by calling ```self.define_model```,
        Moves model to the GPU.

        If DDP is initialized, initialize the DDP wrapper.

        Args:
            cfg (dict): config

        Returns:
            model (nn.Module)
        """

        model = self.define_model(cfg)
        model = self.to_running_device(model)
        if torch.distributed.is_initialized():
            model = DDP(model, device_ids=[get_local_rank()], find_unused_parameters=cfg.get("FIND_UNUSED_PARAMETERS", False))
        return model

    def get_ckpt_path(self, epoch: int) -> str:
        """Get checkpoint path.

        The format is "{ckpt_save_dir}/{model_name}_{epoch}"

        Args:
            epoch (int): current epoch.

        Returns:
            checkpoint path (str)
        """

        epoch_str = str(epoch).zfill(len(str(self.num_epochs)))
        ckpt_name = '{}_{}.pt'.format(self.model_name, epoch_str)
        return os.path.join(self.ckpt_save_dir, ckpt_name)

    def save_model(self, epoch: int):
        """Save checkpoint every epoch.

        checkpoint format is {
            'epoch': current epoch ([1, num_epochs]),
            'model_state_dict': state_dict of model,
            'optim_state_dict': state_dict of optimizer
        }

        Decide whether to delete the last checkpoint by the checkpoint save strategy.

        Args:
            epoch (int): current epoch.
        """

        ckpt_dict = get_ckpt_dict(self.model, self.optim, epoch)

        # backup last epoch
        # last_ckpt_path = self.get_ckpt_path(epoch - 1)
        # backup_last_ckpt(last_ckpt_path, epoch, self.ckpt_save_strategy)

        # save ckpt
        ckpt_path = self.get_ckpt_path(epoch)
        save_ckpt(ckpt_dict, ckpt_path, self.logger)

        # clear ckpt every 10 epoch or in the end
        if epoch % 10 == 0 or epoch == self.num_epochs:
            clear_ckpt(self.ckpt_save_dir)

    def load_model_resume(self, strict: bool = True):
        """Load last checkpoint in checkpoint save dir to resume training.

        Load model state dict.
        Load optimizer state dict.
        Load start epoch and set it to lr_scheduler.

        Args:
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        """

        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, use_gpu=self.use_gpu, logger=self.logger)
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            self.optim.load_state_dict(checkpoint_dict['optim_state_dict'])
            self.start_epoch = checkpoint_dict['epoch']
            if self.scheduler is not None:
                self.scheduler.last_epoch = checkpoint_dict['epoch']
            self.logger.info('resume training')
        except (IndexError, OSError, KeyError):
            pass

    def load_model(self, ckpt_path: str = None, strict: bool = True):
        """Load model state dict.
        if param `ckpt_path` is None, load the last checkpoint in `self.ckpt_save_dir`,
        else load checkpoint from `ckpt_path`

        Args:
            ckpt_path (str, optional): checkpoint path, default is None
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        """

        try:
            checkpoint_dict = load_ckpt(self.ckpt_save_dir, ckpt_path=ckpt_path, use_gpu=self.use_gpu,
                                        logger=self.logger)
            if isinstance(self.model, DDP):
                self.model.module.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
            else:
                self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=strict)
        except (IndexError, OSError):
            raise OSError('Ckpt file does not exist')

    def train(self, cfg: dict):
        """Train model.

        Train process:
        [init_training]
        for in train_epoch
            [on_epoch_start]
            for in train iters
                [train_iters]
            [on_epoch_end] ------> Epoch Val: val every n epoch
                                    [on_validating_start]
                                    for in val iters
                                        val iter
                                    [on_validating_end]
        [on_training_end]

        Args:
            cfg (dict): config
        """

        self.init_training(cfg)

        # train time predictor
        train_time_predictor = TimePredictor(self.start_epoch, self.num_epochs)

        # training loop
        for epoch_index in range(self.start_epoch, self.num_epochs):
            epoch = epoch_index + 1
            self.on_epoch_start(epoch)
            epoch_start_time = time.time()
            # start training
            self.model.train()

            # tqdm process bar
            data_iter = tqdm(self.train_data_loader) if get_local_rank() == 0 else self.train_data_loader

            # data loop
            for iter_index, data in enumerate(data_iter):
                loss = self.train_iters(epoch, iter_index, data)
                if loss is not None:
                    self.backward(loss)
            # update lr_scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_end_time = time.time()
            # epoch time
            self.update_epoch_meter('train_time', epoch_end_time - epoch_start_time)
            self.on_epoch_end(epoch)

            expected_end_time = train_time_predictor.get_expected_end_time(epoch)

            # estimate training finish time
            if epoch < self.num_epochs:
                self.logger.info('The estimated training finish time is {}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expected_end_time))))

        # log training finish time
        self.logger.info('The training finished at {}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        ))

        self.on_training_end()

    def setup_graph(self, data):
        pass

    def init_training(self, cfg: dict):
        """Initialize training

        Args:
            cfg (dict): config
        """

        # init training param
        self.num_epochs = cfg['TRAIN']['NUM_EPOCHS']
        self.start_epoch = 0
        self.ckpt_save_strategy = cfg['TRAIN'].get('CKPT_SAVE_STRATEGY')

        # train data loader
        self.train_data_loader = self.build_train_data_loader(cfg)
        self.register_epoch_meter('train_time', 'train', '{:.2f} (s)', plt=False)

        if cfg['TRAIN'].get('SETUP_GRAPH'):
            for data in self.train_data_loader:
                # data loop
                self.setup_graph(data)
                break

        # create optim
        self.optim = build_optim(cfg['TRAIN']['OPTIM'], self.model)
        self.logger.info('set optim: ' + str(self.optim))

        # create lr_scheduler
        if hasattr(cfg['TRAIN'], 'LR_SCHEDULER'):
            self.scheduler = build_lr_scheduler(cfg['TRAIN']['LR_SCHEDULER'], self.optim)
            self.logger.info('set lr_scheduler: ' + str(self.scheduler))
            self.register_epoch_meter('lr', 'train', '{:.2e}')

        # fine tune
        if hasattr(cfg['TRAIN'], 'FINETUNE_FROM'):
            self.load_model(cfg['TRAIN']['FINETUNE_FROM'])
            self.logger.info('start fine tuning')

        # resume
        # self.load_model_resume()

        # init tensorboard(after resume)
        if is_master():
            self.tensorboard_writer = SummaryWriter(
                os.path.join(self.ckpt_save_dir, 'tensorboard'),
                purge_step=(self.start_epoch + 1) if self.start_epoch != 0 else None
            )

        # init validation
        if hasattr(cfg, 'VAL'):
            self.init_validation(cfg)
        # init test
        if hasattr(cfg, 'TEST'):
            self.init_test(cfg)

    def on_epoch_start(self, epoch: int):
        """Callback at the start of an epoch.

        Args:
            epoch (int): current epoch
        """

        # print epoch num
        self.logger.info('epoch {:d} / {:d}'.format(epoch, self.num_epochs))
        # update lr meter
        if self.scheduler is not None:
            self.update_epoch_meter('lr', self.scheduler.get_last_lr()[0])

        # set epoch for sampler in distributed mode
        # see https://pytorch.org/docs/stable/data.html
        sampler = self.train_data_loader.sampler
        if torch.distributed.is_initialized() and isinstance(sampler, DistributedSampler) and sampler.shuffle:
            sampler.set_epoch(epoch)

    def on_epoch_end(self, epoch: int):
        """Callback at the end of an epoch.

        Args:
            epoch (int): current epoch.
        """

        # print train meters
        self.print_epoch_meters('train')
        # tensorboard plt meters
        self.plt_epoch_meters('train', epoch)
        # validate
        if self.val_data_loader is not None and epoch % self.val_interval == 0:
            self.validate(train_epoch=epoch)
        # test
        if self.test_data_loader is not None and epoch % self.test_interval == 0:
            self.test(train_epoch=epoch)
        # save model
        if is_master():
            self.save_model(epoch)
        # reset meters
        self.reset_epoch_meters()

    def on_training_end(self):
        """Callback at the end of training.
        """

        if is_master():
            # close tensorboard writer
            self.tensorboard_writer.close()

    @abstractmethod
    def train_iters(self, epoch: int, iter_index: int, data: torch.Tensor or tuple) -> torch.Tensor:
        """It must be implement to define training detail.

        If it returns `loss`, the function ```self.backward``` will be called.

        Args:
            epoch (int): current epoch.
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader

        Returns:
            loss (torch.Tensor)
        """

        pass

    def backward(self, loss: torch.Tensor):
        """Backward and update params.

        Args:
            loss (torch.Tensor): loss
        """

        self.optim.zero_grad()
        loss.backward()
        if hasattr(self, 'clip'):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optim.step()

    @torch.no_grad()
    @master_only
    def validate(self, cfg: dict = None, train_epoch: int = None):
        """Validate model.

        Args:
            cfg (dict, optional): config
            train_epoch (int, optional): current epoch if in training process.
        """

        # init validation if not in training process
        if train_epoch is None:
            self.init_validation(cfg)

        self.on_validating_start()

        val_start_time = time.time()
        self.model.eval()

        # val loop
        for iter_index, data in enumerate(self.val_data_loader):
            self.val_iters(iter_index, data)

        val_end_time = time.time()
        self.update_epoch_meter('val_time', val_end_time - val_start_time)
        # print val meters
        self.print_epoch_meters('val')
        if train_epoch is not None:
            # tensorboard plt meters
            self.plt_epoch_meters('val', train_epoch // self.val_interval)

        self.on_validating_end()

    @torch.no_grad()
    @master_only
    def test(self, cfg: dict = None, train_epoch: int = None):
        """test model.

        Args:
            cfg (dict, optional): config
            train_epoch (int, optional): current epoch if in training process.
        """

        # init test if not in training process
        if train_epoch is None:
            self.init_test(cfg)

        self.on_test_start()

        test_start_time = time.time()
        self.model.eval()

        # test loop
        for iter_index, data in enumerate(self.test_data_loader):
            self.test_iters(iter_index, data)

        test_end_time = time.time()
        self.update_epoch_meter('test_time', test_start_time - test_end_time)
        # print test meters
        self.print_epoch_meters('test')
        if train_epoch is not None:
            # tensorboard plt meters
            self.plt_epoch_meters('test', train_epoch // self.test_interval)

        self.on_test_end()

    @master_only
    def init_validation(self, cfg: dict):
        """Initialize validation

        Args:
            cfg (dict): config
        """

        self.val_interval = cfg['VAL'].get('INTERVAL', 1)
        self.val_data_loader = self.build_val_data_loader(cfg)
        self.register_epoch_meter('val_time', 'val', '{:.2f} (s)', plt=False)

    @master_only
    def init_test(self, cfg: dict):
        """Initialize test

        Args:
            cfg (dict): config
        """

        self.test_interval = cfg['TEST'].get('INTERVAL', 1)
        self.test_data_loader = self.build_test_data_loader(cfg)
        self.register_epoch_meter('test_time', 'test', '{:.2f} (s)', plt=False)

    @master_only
    def on_validating_start(self):
        """Callback at the start of validating.
        """

        pass

    @master_only
    def on_validating_end(self):
        """Callback at the end of validating.
        """

        pass
    @master_only
    def on_test_start(self):
        """Callback at the start of testing.
        """

        pass

    @master_only
    def on_test_end(self):
        """Callback at the end of testing.
        """

        pass

    def val_iters(self, iter_index: int, data: torch.Tensor or tuple):
        """It can be implement to define testing detail (not necessary).

        Args:
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader
        """

        raise NotImplementedError()

    def test_iters(self, iter_index: int, data: torch.Tensor or tuple):
        """It can be implement to define testing detail (not necessary).

        Args:
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader
        """

        raise NotImplementedError()

    @master_only
    def register_epoch_meter(self, name, meter_type, fmt='{:f}', plt=True):
        if self.meter_pool is None:
            self.meter_pool = MeterPool()
        self.meter_pool.register(name, meter_type, fmt, plt)

    @master_only
    def update_epoch_meter(self, name, value):
        self.meter_pool.update(name, value)

    @master_only
    def print_epoch_meters(self, meter_type):
        self.meter_pool.print_meters(meter_type, self.logger)

    @master_only
    def plt_epoch_meters(self, meter_type, step):
        self.meter_pool.plt_meters(meter_type, step, self.tensorboard_writer)

    @master_only
    def reset_epoch_meters(self):
        self.meter_pool.reset()
