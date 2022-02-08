"""Everything is based on config.

`Config` is the set of all configurations. `Config` is is implemented by `dict`, We recommend using `EasyDict`.

Look at the following example:

cfg.py

```python
import os
from easydict import EasyDict

from my_runner import MyRunner

CFG = EasyDict()

CFG.DESC = 'my net'  # customized description
CFG.RUNNER = MyRunner
CFG.GPU_NUM = 1

CFG.MODEL = EasyDict()
CFG.MODEL.NAME = 'my_net'

CFG.TRAIN = EasyDict()

CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
CFG.TRAIN.CKPT_SAVE_STRATEGY = None

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = 'SGD'
CFG.TRAIN.OPTIM.PARAM = {
    'lr': 0.002,
    'momentum': 0.1,
}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 4
CFG.TRAIN.DATA.DIR = './my_data'
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.PIN_MEMORY = True
CFG.TRAIN.DATA.PREFETCH = True

CFG.VAL = EasyDict()

CFG.VAL.INTERVAL = 1

CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.DIR = 'mnist_data'

CFG._TRAINING_INDEPENDENT` = [
    'OTHER_CONFIG'
]

```

All configurations consists of two parts:
    1. Training dependent configuration: changing this will affect the training results.
    2. Training independent configuration: changing this will not affect the training results.

Notes:
    All training dependent configurations will be calculated MD5,
    this MD5 value will be the sub directory name of checkpoint save directory.
    If the MD5 value is `098f6bcd4621d373cade4e832627b4f6`,
    real checkpoint save directory is `{CFG.TRAIN.CKPT_SAVE_DIR}/098f6bcd4621d373cade4e832627b4f6`

Notes:
    Each configuration default is training dependent,
    except the key is in `TRAINING_INDEPENDENT_KEYS` or `CFG._TRAINING_INDEPENDENT`
"""

import os
import shutil
import types
import copy
import hashlib

TRAINING_INDEPENDENT_FLAG = '_TRAINING_INDEPENDENT'

TRAINING_INDEPENDENT_KEYS = {
    'DIST_BACKEND',
    'DIST_INIT_METHOD',
    'TRAIN.CKPT_SAVE_STRATEGY',
    'TRAIN.DATA.NUM_WORKERS',
    'TRAIN.DATA.PIN_MEMORY',
    'TRAIN.DATA.PREFETCH',
    'VAL'
}


def get_training_dependent_config(cfg: dict, except_keys: set or list = None) -> dict:
    """Get training dependent config.
    Recursively traversal each key,
    if the key is in `TRAINING_INDEPENDENT_KEYS` or `CFG._TRAINING_INDEPENDENT`, pop it.

    Args:
        cfg (dict): Config
        except_keys (set or list): the keys need to be excepted

    Returns:
        cfg (dict): Training dependent configs
    """
    cfg_copy = copy.deepcopy(cfg)

    if except_keys is None:
        except_keys = copy.deepcopy(TRAINING_INDEPENDENT_KEYS)
        if cfg_copy.get(TRAINING_INDEPENDENT_FLAG) is not None:
            except_keys.update(cfg_copy[TRAINING_INDEPENDENT_FLAG])

    # convert to set
    if isinstance(except_keys, list):
        except_keys = set(except_keys)

    if cfg_copy.get(TRAINING_INDEPENDENT_FLAG) is not None:
        cfg_copy.pop(TRAINING_INDEPENDENT_FLAG)

    pop_list = []
    dict_list = []
    for k, v in cfg_copy.items():
        if isinstance(v, dict):
            sub_except_keys = set([])
            for except_key in except_keys:
                if k == except_key:
                    pop_list.append(k)
                elif except_key.find(k) == 0 and except_key[len(k)] == '.':
                    sub_except_keys.add(except_key[len(k) + 1:])
            if len(sub_except_keys) != 0:
                new_v = get_training_dependent_config(v, sub_except_keys)
                dict_list.append((k, new_v))
        else:
            for except_key in except_keys:
                if k == except_key:
                    pop_list.append(k)

    for dict_key, dict_value in dict_list:
        cfg_copy[dict_key] = dict_value

    for pop_key in pop_list:
        cfg_copy.pop(pop_key)

    return cfg_copy


def config_str(cfg: dict, indent: str = '') -> str:
    """Get config string

    Args:
        cfg (dict): Config
        indent (str): if ``cfg`` is a sub config, ``indent`` += '    '

    Returns:
        Config string (str)
    """

    s = ''
    for k, v in cfg.items():
        if isinstance(v, dict):
            s += (indent + '{}:').format(k) + '\n'
            s += config_str(v, indent + '  ')
        elif isinstance(v, types.FunctionType):
            s += (indent + '{}: {}').format(k, v.__name__) + '\n'
        elif k == TRAINING_INDEPENDENT_FLAG:
            pass
        else:
            s += (indent + '{}: {}').format(k, v) + '\n'
    return s


def config_md5(cfg: dict) -> str:
    """Get MD5 value of config.

    Notes:
        Only training dependent configurations participate in the MD5 calculation.

    Args:
        cfg (dict): Config

    Returns:
        MD5 (str)
    """

    cfg_excepted = get_training_dependent_config(cfg)
    m = hashlib.md5()
    m.update(config_str(cfg_excepted).encode('utf-8'))
    return m.hexdigest()


def print_config(cfg: dict):
    """Print config

    Args:
        cfg (dict): Config
    """

    print('MD5: {}'.format(config_md5(cfg)))
    print(config_str(cfg))


def save_config(cfg: dict, file_path: str):
    """Save config

    Args:
        cfg (dict): Config
        file_path (str): file path
    """

    with open(file_path, 'w') as f:
        content = 'MD5: {}\n'.format(config_md5(cfg))
        content += config_str(cfg)
        f.write(content)


def copy_config_file(cfg_file_path: str, save_dir: str):
    """Copy config file to `save_dir`

    Args:
        cfg_file_path (str): config file path
        save_dir (str): save directory
    """

    if os.path.isfile(cfg_file_path) and os.path.isdir(save_dir):
        cfg_file_name = os.path.basename(cfg_file_path)
        shutil.copyfile(cfg_file_path, os.path.join(save_dir, cfg_file_name))


def import_config(path: str, verbose: bool = True) -> dict:
    """Import config by path

    Examples:
        ```
        cfg = import_config('config/my_config.py')
        ```
        is equivalent to
        ```
        from config.my_config import CFG as cfg
        ```

    Args:
        path (str): Config path
        verbose (str): set to ``True`` to print config

    Returns:
        cfg (dict): `CFG` in config file
    """

    if path.find('.py') != -1:
        path = path[:path.find('.py')].replace('/', '.').replace('\\', '.')
    cfg_name = path.split('.')[-1]
    cfg = __import__(path, fromlist=[cfg_name]).CFG

    if verbose:
        print_config(cfg)
    return cfg
