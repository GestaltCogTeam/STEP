import os
import sys

import easydict
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from easydict import EasyDict
from Runner_FullModel import FullModelRunner
from utils.load_data import *

DATASET_NAME = "METR-LA"

GRAPH_NAME   = {"METR-LA": "adj_mx_la.pkl", "PEMS04": "adj_mx_04.pkl", "PEMS-BAY": "adj_mx_bay.pkl"}
NUM_NODES   = {"METR-LA": 207, "PEMS04":307, "PEMS-BAY":325}
adj_mx, adj_ori = load_adj("datasets/sensor_graph/" + GRAPH_NAME[DATASET_NAME], "doubletransition")

CFG = EasyDict()
BATCH_SIZE  = 32
# General Parameters
EPOCHES     = 100
NUM_WORKERS = 8
PIN_MEMORY  = True
PREFETCH    = True
GPU_NUM     = 1
SEED        = 0

# Model Parameters of TSFormer
PATCH_SIZE  = 12        # also the sequence length
WINDOW_SIZE = 288 * 7   # windows size of long history information
HIDDEN_DIM  = 96        # hidden dim of history dim
MASK_RATIO  = 0.75
L_TSFORMER  = 4
LM          = 2

# ==================================== Configuration ============================================= #
# General
CFG.DESCRIPTION = "FullModel@EZ torch"
CFG.RUNNER = FullModelRunner
CFG.DATASET_NAME = DATASET_NAME
CFG.FIND_UNUSED_PARAMETERS = True 
CFG.USE_GPU = True if GPU_NUM >0 else False
CFG.GPU_NUM = GPU_NUM
CFG.SEED = SEED
CFG.K    = 10
CFG.CUDNN_ENABLED = True

# Model
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STEP"
CFG.MODEL.PARAM = EasyDict()
CFG.MODEL.PARAM.TSFORMER = {
    "patch_size":PATCH_SIZE,
    "in_channel":1,
    "out_channel":HIDDEN_DIM,
    "dropout":0.1,
    "mask_size":WINDOW_SIZE/PATCH_SIZE,
    "mask_ratio":MASK_RATIO,
    "L":L_TSFORMER
}
CFG.MODEL.PARAM.BACKEND = EasyDict()
CFG.MODEL.PARAM.BACKEND.GWNET = {
    "num_nodes" : NUM_NODES[DATASET_NAME], 
    "supports"  :[torch.tensor(i) for i in adj_mx],         # the supports are not used
    "dropout"   : 0.3, 
    "gcn_bool"  : True, 
    "addaptadj" : True, 
    "aptinit"   : None, 
    "in_dim"    : 2,
    "out_dim"   : 12,
    "residual_channels" : 32,
    "dilation_channels" : 32,
    "skip_channels"     : 256,
    "end_channels"      : 512,
    "kernel_size"       : 2,
    "blocks"            : 4,
    "layers"            : 2
}

# Train
CFG.TRAIN = EasyDict()
CFG.TRAIN.SETUP_GRAPH = True
CFG.TRAIN.WARMUP_EPOCHS = 0
CFG.TRAIN.CL_EPOCHS     = 6
CFG.TRAIN.CLIP          = 3
# CFG.TRAIN.CKPT_SAVE_STRATEGY = "SaveEveryEpoch"       # delete pt to save space
CFG.TRAIN.NUM_EPOCHS = EPOCHES
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    '_'.join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
## DATA
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.SEQ_LEN = WINDOW_SIZE
CFG.TRAIN.DATA.DATASET_NAME = DATASET_NAME
CFG.TRAIN.DATA.DIR = "datasets/"+CFG.TRAIN.DATA.DATASET_NAME
CFG.TRAIN.DATA.PREFETCH = PREFETCH
CFG.TRAIN.DATA.BATCH_SIZE = BATCH_SIZE
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = NUM_WORKERS
CFG.TRAIN.DATA.PIN_MEMORY = PIN_MEMORY
# OPTIM
### OPTIM for GWNet
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.005,
    "weight_decay":1.0e-5,
    "eps":1.0e-8,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[1, 18, 36, 54, 72],
    "gamma":0.5
}

# Validate
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.SEQ_LEN = WINDOW_SIZE
CFG.VAL.DATA.DATASET_NAME = CFG.TRAIN.DATA.DATASET_NAME
CFG.VAL.DATA.DIR = CFG.TRAIN.DATA.DIR
CFG.VAL.DATA.PREFETCH = CFG.TRAIN.DATA.PREFETCH
CFG.VAL.DATA.BATCH_SIZE = CFG.TRAIN.DATA.BATCH_SIZE
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = CFG.TRAIN.DATA.NUM_WORKERS
CFG.VAL.DATA.PIN_MEMORY = PIN_MEMORY

# Test
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.SEQ_LEN = WINDOW_SIZE
CFG.TEST.DATA.DATASET_NAME = CFG.TRAIN.DATA.DATASET_NAME
CFG.TEST.DATA.DIR = CFG.TRAIN.DATA.DIR
CFG.TEST.DATA.PREFETCH = CFG.TRAIN.DATA.PREFETCH
CFG.TEST.DATA.BATCH_SIZE = CFG.TRAIN.DATA.BATCH_SIZE
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = CFG.TRAIN.DATA.NUM_WORKERS
CFG.TEST.DATA.PIN_MEMORY = PIN_MEMORY
