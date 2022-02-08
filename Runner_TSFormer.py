from torch import nn

from losses.losses_torch import masked_mae, masked_rmse, masked_mape
from dataloader.dataset import MTSDataset

from easytorch import Runner

from models.MAE_TS.model import MAE_TS
from utils.load_data import re_max_min_normalization, standard_re_transform
from utils.log import TrainLogger, clock, load_pkl

class TSFormerRunner(Runner):
    def __init__(self, cfg: dict, use_gpu: bool = True):
        super().__init__(cfg, use_gpu=use_gpu)
        logger  = TrainLogger()
        self.clip = 5
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

    def init_training(self, cfg):
        """Initialize training.

        Including loss, training meters, etc.

        Args:
            cfg (dict): config
        """
        super().init_training(cfg)

        self.register_epoch_meter("train_loss", 'train', '{:.4f}')
        self.register_epoch_meter("train_MAPE", 'train', '{:.4f}')
        self.register_epoch_meter("train_RMSE", 'train', '{:.4f}')

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
            'TSFormer': MAE_TS
        }[cfg['MODEL']['NAME']](**cfg['MODEL'].get('PARAM', {}))

    @staticmethod
    def build_train_dataset(cfg: dict):
        """Build MNIST train dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """
        raw_file_path = cfg["TRAIN"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["TRAIN"]["DATA"]["DIR"] + "/train_index.pkl"
        seq_len = cfg['TRAIN']['DATA']['SEQ_LEN']
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, throw=True, pretrain=True)
        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """
        raw_file_path = cfg["VAL"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["VAL"]["DATA"]["DIR"] + "/valid_index.pkl"
        seq_len = cfg['VAL']['DATA']['SEQ_LEN']
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, throw=True, pretrain=True)
        return dataset

    @staticmethod
    def build_test_dataset(cfg: dict):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """
        raw_file_path = cfg["TEST"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["TEST"]["DATA"]["DIR"] + "/test_index.pkl"
        seq_len = cfg['TEST']['DATA']['SEQ_LEN']
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, throw=True, pretrain=True)
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
        input_data  = self.to_running_device(data[0])
        # abs_idx     = self.to_running_device(data[1]).long()

        output_masked_tokens, label_masked_tokens, plot_args = self.model(input_data)

        # scale data and calculate loss
        if  "max" in self.scaler_args.keys():  # traffic flow
            predict     = self.scaler(output_masked_tokens.transpose(1,2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            real_val    = self.scaler(label_masked_tokens.transpose(1,2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            mae_loss    = self.loss(predict, real_val)
        else:
            ## inverse transform for both predict and real value.
            predict     = self.scaler(output_masked_tokens, **self.scaler_args)
            real_val    = self.scaler(label_masked_tokens, **self.scaler_args)
            mae_loss    = self.loss(predict, real_val, 0)

        loss = mae_loss
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
        input_data  = self.to_running_device(data[0])
        # abs_idx     = self.to_running_device(data[1]).long()

        output_masked_tokens, label_masked_tokens, plot_args = self.model(input_data)

        # scale data and calculate loss
        if  "max" in self.scaler_args.keys():  # traffic flow
            predict     = self.scaler(output_masked_tokens.transpose(1,2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            real_val    = self.scaler(label_masked_tokens.transpose(1,2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            loss        = self.loss(predict, real_val)
        else:
            ## inverse transform for both predict and real value.
            predict     = self.scaler(output_masked_tokens, **self.scaler_args)
            real_val    = self.scaler(label_masked_tokens, **self.scaler_args)
            loss        = self.loss(predict, real_val, 0.0)

        # metrics
        mape = masked_mape(predict,real_val,0.0)
        rmse = masked_rmse(predict,real_val,0.0)

        self.update_epoch_meter('val_loss', loss.item())
        self.update_epoch_meter('val_MAPE', mape.item())
        self.update_epoch_meter('val_RMSE', rmse.item())

    def test_iters(self, iter_index, data):
        """Validation details.

        Args:
            iter_index (int): current iter.
            data (torch.Tensor or tuple): Data provided by DataLoader
        """
        input_data  = self.to_running_device(data[0])
        # abs_idx     = self.to_running_device(data[1]).long()

        output_masked_tokens, label_masked_tokens, plot_args = self.model(input_data)

        # scale data and calculate loss
        if  "max" in self.scaler_args.keys():  # traffic flow
            predict     = self.scaler(output_masked_tokens.transpose(1,2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            real_val    = self.scaler(label_masked_tokens.transpose(1,2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            loss        = self.loss(predict, real_val)
        else:
            ## inverse transform for both predict and real value.
            predict     = self.scaler(output_masked_tokens, **self.scaler_args)
            real_val    = self.scaler(label_masked_tokens, **self.scaler_args)
            loss        = self.loss(predict, real_val, 0.0)

        # metrics
        mape = masked_mape(predict,real_val,0.0)
        rmse = masked_rmse(predict,real_val,0.0)

        self.update_epoch_meter('test_loss', loss.item())
        self.update_epoch_meter('test_MAPE', mape.item())
        self.update_epoch_meter('test_RMSE', rmse.item())
