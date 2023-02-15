from config import *
from dataset import ImageDataset
from torch.utils.data import DataLoader
from ESRGAN.loss import VGGLoss
from ESRGAN.model import Generator, Discriminator, initialize_weights
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torch.optim as optim

# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True
import numpy as np


