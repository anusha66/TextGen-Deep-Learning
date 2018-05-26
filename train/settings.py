import torch
file_loc = '../boxscore-data/rotowire/'
use_cuda = torch.cuda.is_available()

MAX_LENGTH = 800
MAX_SENTENCES = None
MAX_TRAIN_NUM = None

PRETRAIN = None
iterNum = None

EMBEDDING_SIZE = 600
LR = 0.01
EPOCH_TIME = 220
BATCH_SIZE = 2
GRAD_CLIP = 5
LAYER_DEPTH = 2

GET_LOSS = 1
SAVE_MODEL = 5

ENCODER_STYLE = 'HierarchicalRNN'
DECODER_STYLE = 'HierarchicalRNN'
OUTPUT_FILE = 'pretrain_copy_ms5'
COPY_PLAYER = True
TOCOPY = True

MAX_PLAYERS = 31
PLAYER_PADDINGS = ['<PAD' + str(i) + '>' for i in range(0, MAX_PLAYERS)]
