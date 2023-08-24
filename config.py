import torch

### Data
NEG_POS_RATIO = 3
N_NEIGHBORS = 8
IMG_SIZE = 1024
# AREA_THRESH = 1500
AREA_THRESH = 2500
CSV_DIR = "/home/ubuntu/project/cv/text_segmenter/data"
# CSV_DIR = "/Users/jongbeomkim/Desktop/workspace/text_segmenter/data"

### Optimizer
# "Optimized by SGD with $momentum = 0.9$ and $weight_decay = 5 \times 10^{-4}$.
MOMENTUM = 0.9
WEIGHT_DECAY = 5 * 1e-4
# "Learning rate is set to $10^{-3}$ for the first 100 iterations, and fixed at $10^{-2}$ for the rest."
INIT_LR = 1e-3
FIN_LR = 1e-2

### Training
SEED = 33
N_WORKERS = 4
BATCH_SIZE = 1
FEAT_MAP_SIZE = IMG_SIZE // 2
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# DEVICE = torch.device("cpu")
N_EPOCHS = 300