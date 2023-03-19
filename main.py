import torch
import torchvision.utils

from config import *
from Dataset.dataset import NeRF_Dataset
from NeRF.model import NeRF, NeRF_Fine

train_dataset = NeRF_Dataset(quality="lr", mode="train")
train_dataloader = DataLoader(dataset=train_dataset, batch_size=hyperparameter_facts['batch_size'], shuffle=True)

valid_dataset = NeRF_Dataset(quality="lr", mode="valid")
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=hyperparameter_facts['batch_size'], shuffle=True)

test_dataloader = NeRF_Dataset(quality="lr", mode="test")
test_dataloader = DataLoader(dataset=test_dataloader, batch_size=hyperparameter_facts['batch_size'], shuffle=False)

K = train_dataset.K
nerf = NeRF(K).to(device)
grad_vars = list(nerf.parameters())

nerf_fine = NeRF_Fine(K).to(device)
grad_vars += list(nerf_fine.parameters())

optimizer = torch.optim.Adam(params=grad_vars, lr=nerf_model['lrate'], betas=(0.9, 0.999))

if checkpoint_facts['load_checkpoint']:  # Load checkpoint
    checkpoint_file = os.path.join(paths_[1], checkpoint_facts['nerf_checkpoints']['checkpoint_nerf'])
    checkpoint_file_fine = os.path.join(paths_[1], checkpoint_facts['nerf_checkpoints']['checkpoint_nerf_fine'])
    load_checkpoint(checkpoint_file, nerf, optimizer, nerf_model['lrate'])
    load_checkpoint(checkpoint_file_fine, nerf_fine, optimizer, nerf_model['lrate'])

start = 0

writer = SummaryWriter(paths_[-1])
epochs = hyperparameter_facts['epochs']
for epoch in range(epochs):
    running_loss = 0
    batches = len(train_dataloader)
    global_step = start
    nerf.train()
    nerf_fine.train()
    for index, data in enumerate(train_dataloader):
        print("Batch Idx: ", index)
        start = start + 1
        target, pose = data
        target = target.squeeze(0)
        pose = pose.squeeze(0)
        target_s, rgb_map, disp_map, acc_map, weights, depth_map, fine_parameters = nerf(pose, target=target,
                                                                                         eval=False)
        viewdirs = fine_parameters['viewdirs'].to(device)
        weights = fine_parameters['weights'].to(device)
        z_vals = fine_parameters['z_vals'].to(device)
        rays_o = fine_parameters['rays_o'].to(device)
        rays_d = fine_parameters['rays_d'].to(device)
        fine_rgb_map, fine_disp_map, fine_acc_map, fine_weights, fine_depth_map = nerf_fine(viewdirs, weights, z_vals,
                                                                                            rays_o, rays_d, eval=False)
        target_s = target_s.to(device)
        # Learning
        optimizer.zero_grad()
        nerf_loss = img2mse(rgb_map, target_s)
        nerf_fine_loss = img2mse(fine_rgb_map, target_s)
        loss = nerf_loss + nerf_fine_loss
        loss.backward()

        # Writer loss

        psnr = mse2psnr(loss)
        optimizer.step()

        # Learning Decay
        decay_rate = 0.1
        decay_steps = nerf_model['lrate_decay']
        new_lrate = nerf_model['lrate'] * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # if index % 500 == 0:
        if index == 0:
            iters = index + epoch * batches + 1
            writer.add_scalar("NeRF", nerf_loss.item(), iters)
            writer.add_scalar("NeRF_Fine", nerf_fine_loss.item(), iters)
            writer.add_scalar("Multi-Task Loss", loss.item(), iters)
            with torch.no_grad():
                _, eval_rgb_map, eval_disp_map, eval_acc_map, eval_weights, eval_depth_map, eval_fine_parameters = nerf(
                    pose)
                ground_truth = target.unsqueeze(0)
                nerf_rgb_prediction = eval_rgb_map.unsqueeze(0).to('cpu')
                eval_fine_rgb_map, eval_fine_disp_map, eval_fine_acc_map, eval_fine_weights, eval_fine_depth_map = nerf_fine(
                    eval_fine_parameters['viewdirs'].to(device), eval_fine_parameters['weights'].to(device),
                    eval_fine_parameters['z_vals'].to(device), eval_fine_parameters['rays_o'].to(device),
                    eval_fine_parameters['rays_d'].to(device))
                eval_fine_rgb_map = torch.reshape(eval_fine_rgb_map, [100, 100, 3])
                nerf_fine_prediction = eval_fine_rgb_map.unsqueeze(0).to('cpu')
                concat = torch.cat([ground_truth, nerf_fine_prediction, nerf_rgb_prediction], 0).permute(0, 3, 1, 2)
                img_grid = torchvision.utils.make_grid(concat, normalize=True)
                writer.add_image("NeRF Images", img_grid, global_step=epoch + 1)
                print(f"Train stage: Neural Radiance Field "
                      f"Epoch[{epoch + 1:04d}/{epochs:04d}]({index + 1:05d}/{batches:05d}) "
                      f"NeRF: {nerf_loss.item():.6f} NeRF Fine: {nerf_fine_loss.item():.6f} "
                      f"Multi-Task Loss: {loss:.6f} .")
    if checkpoint_facts['save_checkpoint']:
        checkpoint_file = os.path.join(paths_[1], checkpoint_facts['nerf_checkpoints']['checkpoint_nerf'])
        checkpoint_file_fine = os.path.join(paths_[1], checkpoint_facts['nerf_checkpoints']['checkpoint_nerf_fine'])
        save_checkpoint(nerf, optimizer, checkpoint_file)
        save_checkpoint(nerf_fine, optimizer, checkpoint_file_fine)
