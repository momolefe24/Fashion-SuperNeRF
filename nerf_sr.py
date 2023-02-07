import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from NeRF_SR.model import Net
from NeRF_SR.dataset import InputPipeline, DataLoader
import torchvision
from config import *

training_dataset = InputPipeline()
training_dataloader = DataLoader(training_dataset, batch_size=training_facts['batch_size'], shuffle=True)

def training_function(optimizer, step, do_train=True):
    if do_train:
        for epoch in range(nerf_facts['epochs']):
            running_loss = 0
            for batch_idx, data in enumerate(training_dataloader):
                print("Batch idx: ", batch_idx)
                hr_image, lr_rays_flat, lr_t_vals = data
                lr_rays_flat = lr_rays_flat.to(device, torch.float32)
                lr_t_vals = lr_t_vals.to(device, torch.float32)
                hr_image = hr_image.to(device)
                rgb, _ = net(lr_rays_flat, lr_t_vals)
                optimizer.zero_grad()
                loss = criterion(hr_image.permute(0, 3, 1, 2), rgb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if batch_idx % 20 == 0:
                    print(f"[{epoch + 1},{batch_idx + 1:5d}] loss: {running_loss / 85:.5f}")
                    with torch.no_grad():
                        rgb, _ = net(lr_rays_flat, lr_t_vals)
                        img_grid_real = torchvision.utils.make_grid(hr_image.permute(0, 3, 1, 2)[:4], normalize=True)
                        img_grid_rgb = torchvision.utils.make_grid(rgb[:4], normalize=True)
                        writer_real.add_image('Real', img_grid_real, global_step=step)
                        writer_nerf.add_image('Nerf', img_grid_rgb, global_step=step)
                    step += 1
        checkpoint = {'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        print("Finished traning")

net = Net(nerf_facts['num_layers']).to(device)
writer_real = SummaryWriter(f"{paths_[-1]}/NERF_real")
writer_nerf = SummaryWriter(f"{paths_[-1]}/NERF_nerf")
step = 0
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=2e-4, betas=eval(nerf_facts['betas']))
net.train()
torch.cuda.empty_cache()
training_function(optimizer, step)


# net(rays_flat)
