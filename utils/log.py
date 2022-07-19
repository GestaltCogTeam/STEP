
import time
import pickle
import time
import os
import shutil

def clock(func):
    def clocked(*args, **kw):
        t0 = time.perf_counter()
        result = func(*args, **kw)
        elapsed = time.perf_counter() - t0
        name = func.__name__
        print('%s: %0.8fs...' % (name, elapsed))
        return result
    return clocked


def load_pkl(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data     = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data     = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

# def load_pkl(file_path):
#     with open(file_path, 'rb') as f:
#         obj = pickle.load(f)
#     return obj

def dump_pkl(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


class TrainLogger():
    r"""
    Description:
    -----------
    Logger class. Function:
    - print all training hyperparameter setting
    - print all model    hyperparameter setting
    - save all the python file of model

    Args:
    -----------
    path: str
        Log path
    """
    
    def __init__(self):
        path        = 'log/'
        cur_time    = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cur_time    = cur_time.replace(" ", "-")
        try:
            # mkdir
            os.makedirs(path + cur_time)
            print("Backup current files to {0}".format(path+cur_time))
            # pwd = os.getcwd() + "/"
            # copy model files
            shutil.copytree('models',  path + cur_time + "/models")
            shutil.copytree('config',  path + cur_time + "/config")
            shutil.copytree('dataloader',  path + cur_time + "/dataloader")
            try:
                shutil.copyfile('demo_metr.sh',  path + cur_time + "/demo_metr.sh")
                shutil.copyfile('demo_pems04.sh',  path + cur_time + "/demo_pems04.sh")
            except:
                pass
            shutil.copyfile('Runner_FullModel.py',  path + cur_time + "/Runner_FullModel.py")
            shutil.copyfile('Runner_TSFormer.py',  path + cur_time + "/Runner_TSFormer.py")
            shutil.copyfile('main.py',  path + cur_time + "/main.py")
        except FileExistsError:
            pass
