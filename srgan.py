import torch.cuda.amp
import torchvision
from config import *
import torch.nn as nn
from dataset import *
from SRGAN.model import GeneratorResNet, Discriminator, FeatureExtractor
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

to_numpy = lambda x:x.detach().cpu().permute(0,2,3,1).numpy()[0]
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable

H = W = eval(dataset_facts['image']['lr_shape'])[-1]
channels = 3
srgan_epochs = srgan_facts['epochs']

generator = GeneratorResNet().to(device)
discriminator = Discriminator(input_shape=(3,1024,768)).to(device)
feature_extractor = FeatureExtractor().to(device)

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)

# Dataset
eric_train_dataset = ImageDataset(mode="train")
eric_train_dataloader = DataLoader(dataset=eric_train_dataset, batch_size=training_facts['batch_size'], shuffle=True)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=srgan_facts['lr'], betas=eval(srgan_facts['betas']))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=srgan_facts['lr'], betas=eval(srgan_facts['betas']))


writer = SummaryWriter(paths_[-1])
# Tensor = torch.cuda.FloatTensor

# ----------
#  Training
# ----------
step = 0
for epoch in range(srgan_facts['epochs']):
    for i, imgs in enumerate(eric_train_dataloader):

        # Configure model input
        imgs_lr,imgs_hr = imgs
        imgs_lr = imgs_lr.to(device)
        imgs_hr = imgs_hr.to(device)

        # Adversarial ground truths
        with torch.no_grad():
            valid = torch.Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))).to(device)
            fake = torch.Tensor((np.zeros((imgs_lr.size(0), *discriminator.output_shape)))).to(device)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, srgan_facts['epochs'], i, len(eric_train_dataloader), loss_D.item(), loss_G.item())
        )

        batches_done = epoch * len(eric_train_dataloader) + i
        if batches_done % 10  == 0:
            print(
                f"[{epoch + 1},{i + 1:5d}] Critic Loss: {loss_D / len(eric_train_dataloader):.5f} Generator Loss: {loss_G / len(eric_train_dataloader):.5f}")
            # Save image grid with upsampled inputs and SRGAN outputs
            with torch.no_grad():
                img_grid_real = make_grid(imgs_hr, normalize=True)
                img_grid_fake = make_grid(gen_hr, normalize=True)
                writer.add_image("Ground Truth", img_grid_real, global_step=batches_done)
                writer.add_image("Super-Resolved Image", img_grid_fake, global_step=batches_done)
            step += 1
            # save_image(img_grid, "images/%d.png" % batches_done, normalize=False)