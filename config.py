import torch
from pathlib import Path

### Data
N_NEIGHBORS = 8
NEG_POS_RATIO = 3 # "$r$ is the negative-positive ratio and is set to 3 as a common practice."
N_NEIGHBORS = 8
IMG_SIZE = 1024
FEAT_MAP_SIZE = IMG_SIZE // 2
# AREA_THRESH = 1500
AREA_THRESH = 2500
CSV_DIR = "/home/ubuntu/project/cv/text_segmenter/sample_data"
# CSV_DIR = "/Users/jongbeomkim/Desktop/workspace/text_segmenter/sample_data"
LAMB = 0.2

### Architecture
PRETRAINED_VGG16 = True
MODE = "2s"

### Optimizer
# "Optimized by SGD with $momentum = 0.9$ and $weight_decay = 5 \times 10^{-4}$.
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
# "Learning rate is set to $10^{-3}$ for the first 100 iterations, and fixed at $10^{-2}$ for the rest."
INIT_LR = 1e-3
FIN_LR = 1e-2

### Training
SEED = 33
N_WORKERS = 4
BATCH_SIZE = 4
AUTOCAST = True
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
# DEVICE = torch.device("cpu")
N_EPOCHS = 300
N_VAL_EPOCHS = 10
CKPT_DIR = Path(__file__).parent/"checkpoints"

### Post-processing
COLORS = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 255, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 250),
    (240, 50, 230),
    (210, 255, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
)
