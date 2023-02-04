import torch.cuda.amp
import torchvision
from dataset import *
from ESRGAN.model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# Results/ESRGAN/experiment_01_run_01/logs/

import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:35].eval().to(device)

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.loss = nn.MSELoss()

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)


filters = esrgan_facts['filters']
H = W = eval(dataset_facts['image']['lr_shape'])[-1]
num_res_blocks = esrgan_facts['num_res_blocks']
channels = 3
esrgan_epochs = esrgan_facts['epochs']

# Initialize generator and discriminator
generator = Generator(channels, filters=filters, num_res_blocks=num_res_blocks).to(device)
initialize_weights(generator)
discriminator = Discriminator(channels).to(device)
# feature_extractor = FeatureExtractor().to(device)

# Set feature extractor to inference mode
# feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device) # L1
VGG_loss = VGGLoss() # Feature Extractor

# Load the state

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=esrgan_facts['learning_rate'], betas=(esrgan_facts['beta1'], esrgan_facts['beta2']))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=esrgan_facts['learning_rate'], betas=(esrgan_facts['beta1'], esrgan_facts['beta2']))

# Scalers
g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Dataset
eric_train_dataset = ImageDataset(mode="train")
eric_train_dataloader = DataLoader(dataset=eric_train_dataset, batch_size=training_facts['batch_size'], shuffle=True)

# Training

generator.train()
discriminator.train()



# Writer
writer = SummaryWriter(paths_[-1])
# eriklindernoren

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

step = 0
# aladdinpersson
for epoch in range(esrgan_epochs):
    loop = tqdm(eric_train_dataloader, leave=True)
    for i, (lr, hr) in enumerate(loop):
        batches_done = epoch * len(eric_train_dataloader) + i

        # Configure model input
        lr = lr.to(device)
        hr = hr.to(device)

        with torch.cuda.amp.autocast():
            fake = generator(lr)
            critic_real = discriminator(hr)
            critic_fake = discriminator(fake.detach())
            gp = gradient_penalty(discriminator, hr, fake, device=device)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + esrgan_facts['lampda_gp'] * gp
            )

        optimizer_D.zero_grad()
        d_scaler.scale(loss_critic).backward()
        d_scaler.step(optimizer_D)
        d_scaler.update()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with torch.cuda.amp.autocast():
            l1_loss = 1e-2 * criterion_pixel(fake, hr)
            adversarial_loss = 5e-3 * -torch.mean(discriminator(fake))
            loss_for_vgg = VGG_loss(fake, hr)
            gen_loss = l1_loss + loss_for_vgg + adversarial_loss

        optimizer_G.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(optimizer_G)
        g_scaler.update()


        if i % 20 == 0:
            print(f"[{epoch + 1},{i + 1:5d}] Critic Loss: {loss_critic / len(eric_train_dataloader):.5f} Generator Loss: {gen_loss / len(eric_train_dataloader):.5f}")
            with torch.no_grad():
                img_grid_real = torchvision.utils.make_grid(hr.permute(0, 3, 1, 2), normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake.permute(0, 3, 1, 2), normalize=True)
                writer.add_scalar("Critic loss", loss_critic.item(), global_step=step)
                writer.add_scalar("Generator loss", gen_loss.item(), global_step=step)
                writer.add_image("Ground Truth", img_grid_real, global_step=i)
                writer.add_image("Super-Resolved Image", img_grid_fake, global_step=i)
            step += 1
        save_checkpoint(generator, optimizer_G, filename=f"{paths_[1]}/{checkpoint_facts['ESRGAN']['checkpoint_gen']}")
        save_checkpoint(discriminator, optimizer_D, filename=f"{paths_[1]}/{checkpoint_facts['ESRGAN']['checkpoint_disc']}")
        # plot_examples

        break
