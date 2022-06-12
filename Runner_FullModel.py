import time
import math
from torch import nn
import torch
import torchvision
from easytorch.utils.dist import master_only
from sklearn.metrics import mean_absolute_error

from losses.losses_torch import masked_mae, masked_rmse, masked_mape, metric
from dataloader.dataset import MTSDataset

from easytorch import Runner

from models.model import STEP
from utils.load_data import re_max_min_normalization, standard_re_transform
from utils.log import clock, load_pkl
from utils.log import TrainLogger

class FullModelRunner(Runner):
    def __init__(self, cfg: dict, use_gpu: bool = True):
        super().__init__(cfg, use_gpu=use_gpu)
        logger  = TrainLogger()
        self.clip = 3
        self._lambda = 1
        dataset_name = cfg['DATASET_NAME']
        # scaler
        if dataset_name in ['PEMS04', 'PEMS08']:
            _min = load_pkl("datasets/" + dataset_name + "/min.pkl")
            _max = load_pkl("datasets/" + dataset_name + "/max.pkl")
            self.scaler = re_max_min_normalization
            self.scaler_args    = {'min': _min, 'max':_max}
        elif dataset_name in ['PEMS-BAY', 'METR-LA']:
            mean = load_pkl("datasets/" + dataset_name + "/mean.pkl")
            std  = load_pkl("datasets/" + dataset_name + "/std.pkl")
            self.scaler         = standard_re_transform
            self.scaler_args    = {'mean': mean, 'std':std}
        self.loss = masked_mae
        # self.loss = masked_mae_loss

        self.dataset_name = cfg['DATASET_NAME']
        self.output_seq_len = 12
        self.cl_len = self.output_seq_len
        self.if_cl = True

    def init_training(self, cfg):
        """Initialize training.

        Including loss, training meters, etc.

        Args:
            cfg (dict): config
        """
        self.register_epoch_meter("train_loss", 'train', '{:.4f}')
        self.register_epoch_meter("train_MAPE", 'train', '{:.4f}')
        self.register_epoch_meter("train_RMSE", 'train', '{:.4f}')

        super().init_training(cfg)

    def init_validation(self, cfg: dict):
        """Initialize validation.

        Including validation meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_validation(cfg)

        self.register_epoch_meter("val_loss", 'val', '{:.4f}')
        self.register_epoch_meter("val_MAPE", 'val', '{:.4f}')
        self.register_epoch_meter("val_RMSE", 'val', '{:.4f}')

    def init_test(self, cfg: dict):
        """Initialize test.

        Including test meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_test(cfg)

        self.register_epoch_meter("test_loss", 'test', '{:.4f}')
        self.register_epoch_meter("test_MAPE", 'test', '{:.4f}')
        self.register_epoch_meter("test_RMSE", 'test', '{:.4f}')

    @staticmethod
    def define_model(cfg: dict) -> nn.Module:
        """Define model.

        If you have multiple models, insert the name and class into the dict below,
        and select it through ```config```.

        Args:
            cfg (dict): config

        Returns:
            model (nn.Module)
        """
        return {
            'FullModel': STEP
        }[cfg['MODEL']['NAME']](cfg, **cfg.MODEL.PARAM)

    def build_train_dataset(self, cfg: dict):
        """Build train dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """
        raw_file_path = cfg["TRAIN"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["TRAIN"]["DATA"]["DIR"] + "/train_index.pkl"
        seq_len = cfg['TRAIN']['DATA']['SEQ_LEN']
        batch_size = cfg['TRAIN']['DATA']['BATCH_SIZE']
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, throw=False, pretrain=False)
        
        warmup_epoches = cfg['TRAIN']['WARMUP_EPOCHS']
        cl_epochs      = cfg['TRAIN']['CL_EPOCHS']
        self.init_lr   = cfg['TRAIN']['OPTIM']['PARAM']['lr']
        self.itera_per_epoch = math.ceil(len(dataset) / batch_size)
        self.warmup_steps = self.itera_per_epoch * warmup_epoches
        self.cl_steps     = self.itera_per_epoch * cl_epochs
        print("cl_steps:{0}".format(self.cl_steps))
        print("warmup_steps:{0}".format(self.warmup_steps))
        
        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        """Build val dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """
        raw_file_path = cfg["VAL"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["VAL"]["DATA"]["DIR"] + "/valid_index.pkl"
        seq_len = cfg['VAL']['DATA']['SEQ_LEN']
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, pretrain=False)
        print("val len: {0}".format(len(dataset)))
        return dataset

    @staticmethod
    def build_test_dataset(cfg: dict):
        """Build val dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """
        raw_file_path = cfg["TEST"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["TEST"]["DATA"]["DIR"] + "/test_index.pkl"
        seq_len = cfg['TEST']['DATA']['SEQ_LEN']
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, pretrain=False)
        print("test len: {0}".format(len(dataset)))
        return dataset

    def train_iters(self, epoch, iter_index, data):
        """Training details.

        Args:
            epoch (int): current epoch.
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader

        Returns:
            loss (torch.Tensor)
        """
        iter_num = (epoch-1) * self.itera_per_epoch + iter_index

        y, short_x, long_x = data
        Y   = self.to_running_device(y)
        S_X = self.to_running_device(short_x)
        L_X = self.to_running_device(long_x)

        output, theta, priori_adj  = self.model(S_X, L_X=L_X, label=Y, batch_seen=iter_num, epoch=epoch)
        output  = output.transpose(1,2)

        # # reg
        B, N, N = theta.shape
        theta = theta.view(B, N*N)
        tru = priori_adj.view(B, N*N)
        BCE_loss = nn.BCELoss()
        loss_g = BCE_loss(theta, tru)
        
        # curriculum learning
        if  iter_num < self.warmup_steps:   # warmupping
            self.cl_len = self.output_seq_len
        elif iter_num == self.warmup_steps:
            # init curriculum learning
            self.cl_len = 1
            for param_group in self.optim.param_groups:
                param_group["lr"] = self.init_lr
            print("======== Start curriculum learning... reset the learning rate to {0}. ========".format(self.init_lr))
        else:
            # begin curriculum learning
            if (iter_num - self.warmup_steps) % self.cl_steps == 0 and self.cl_len <= self.output_seq_len:
                self.cl_len += int(self.if_cl)

        # scale data and calculate loss
        if  "max" in self.scaler_args.keys():  # traffic flow
            predict     = self.scaler(output.transpose(1,2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            real_val    = self.scaler(Y.transpose(1,2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            real_val    = real_val[..., 0]
            mae_loss    = self.loss(predict[:, :self.cl_len, :], real_val[:, :self.cl_len, :])
        else:
            ## inverse transform for both predict and real value.
            predict     = self.scaler(output, **self.scaler_args)
            real_val    = self.scaler(Y[:,:,:,0], **self.scaler_args)
            mae_loss    = self.loss(predict[:, :self.cl_len, :], real_val[:, :self.cl_len, :], 0)

        if iter_num == 50:
            self._lambda = 1
        if iter_num == 100:
            self._lambda == 0.1
        if iter_num == 1000:
            self._lambda = 0.01
        self._lambda = 1 / (int(epoch/6)+1)

        loss = mae_loss + self._lambda * loss_g
        # metrics
        mape = masked_mape(predict,real_val,0.0)
        rmse = masked_rmse(predict,real_val,0.0)

        self.update_epoch_meter('train_loss', loss.item())
        self.update_epoch_meter('train_MAPE', mape.item())
        self.update_epoch_meter('train_RMSE', rmse.item())

        return loss

    def val_iters(self, iter_index, data):
        """Validation details.

        Args:
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader
        """
        y, short_x, long_x = data
        Y   = self.to_running_device(y)
        S_X = self.to_running_device(short_x)
        L_X = self.to_running_device(long_x)

        output, theta, priori_adj  = self.model(S_X, L_X=L_X, label=Y)
        output  = output.transpose(1,2)

        # scale data and calculate loss
        if  "max" in self.scaler_args.keys():  # traffic flow
            predict     = self.scaler(output.transpose(1,2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            real_val    = self.scaler(Y.transpose(1,2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            real_val    = real_val[..., 0]
            mae_loss    = self.loss(predict[:, :self.cl_len, :], real_val[:, :self.cl_len, :])
        else:
            ## inverse transform for both predict and real value.
            predict     = self.scaler(output, **self.scaler_args)
            real_val    = self.scaler(Y[:,:,:,0], **self.scaler_args)

        # metrics
        loss = self.loss(predict, real_val, 0.0)
        mape = masked_mape(predict,real_val,0.0)
        rmse = masked_rmse(predict,real_val,0.0)

        self.update_epoch_meter('val_loss', loss.item())
        self.update_epoch_meter('val_MAPE', mape.item())
        self.update_epoch_meter('val_RMSE', rmse.item())

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
        outputs = []
        y_list  = []
        for iter_index, data in enumerate(self.test_data_loader):
            preds, testy = self.test_iters(iter_index, data)
            outputs.append(preds)
            y_list.append(testy)
        yhat    = torch.cat(outputs,dim=0)
        y_list  = torch.cat(y_list, dim=0)

        # scale data and calculate loss
        if  "max" in self.scaler_args.keys():  # traffic flow
            real_val    = self.scaler(y_list.squeeze(-1), **self.scaler_args).transpose(1, 2)
            predict     = self.scaler(yhat.unsqueeze(-1), **self.scaler_args).transpose(1, 2)
            real_val    = real_val[..., 0]
            predict     = predict[..., 0]
        else:
            ## inverse transform for both predict and real value.
            real_val    = self.scaler(y_list[:,:,:,0], **self.scaler_args).transpose(1, 2)
            predict     = self.scaler(yhat, **self.scaler_args).transpose(1, 2)

        # summarize the results.
        amae    = []
        amape   = []
        armse   = []

        for i in range(12):
            # For horizon i, only calculate the metrics **at that time** slice here.
            pred    = predict[:,:,i]
            real    = real_val[:,:,i]
            dataset_name = self.dataset_name
            if dataset_name == 'PEMS04' or dataset_name == 'PEMS08':  # traffic flow dataset follows mae metric used in ASTGNN.
                mae     = mean_absolute_error(pred.cpu().numpy(), real.cpu().numpy())
                rmse    = masked_rmse(pred, real, 0.0).item()
                mape    = masked_mape(pred, real, 0.0).item()
                log     = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                log     = log.format(i+1, mae, rmse, mape)
                # print(log)
                amae.append(mae)
                amape.append(mape)
                armse.append(rmse)
            else:       # traffic speed datasets follow the metrics released by GWNet and DCRNN.
                metrics = metric(pred,real)
                log     = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                log     = log.format(i+1, metrics[0], metrics[2], metrics[1])
                # print(log)
                amae.append(metrics[0])     # mae
                amape.append(metrics[1])    # mape
                armse.append(metrics[2])    # rmse
            self.logger.info(log)

        # *** TODO: not test yet ***
        import numpy as np
        log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        self.logger.info(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
        # **************************

        test_end_time = time.time()
        self.update_epoch_meter('test_time', test_start_time - test_end_time)
        # print val meters
        self.print_epoch_meters('test')
        if train_epoch is not None:
            # tensorboard plt meters
            self.plt_epoch_meters('test', train_epoch // self.test_interval)

        self.on_test_end()

    def test_iters(self, iter_index: int, data: torch.Tensor or tuple):
        y, short_x, long_x = data
        Y   = self.to_running_device(y)
        S_X = self.to_running_device(short_x)
        L_X = self.to_running_device(long_x)

        output, theta, priori_adj  = self.model(S_X, L_X=L_X, label=Y)
        output  = output.transpose(1,2)
        return output, Y
    