from config import *
from dataset import ImageDataset
from torch.utils.data import DataLoader
import torchvision
from ESRGAN.model import Generator, Discriminator, initialize_weights, ContentLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
def load_without_penalty_checkpoint():
    print("=>Loading checkpoint")
    checkpoint_file = os.path.join(paths_[1], checkpoint_facts[experiment_type]['best_checkpoint_gen'])
    model = Generator().to(device)
    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict)
    return model

generator = Generator().to(device)
discriminator = Discriminator().to(device)
best_psnr_value = 0.0
writer = SummaryWriter(paths_[-1])

# Optimizers
p_optimizer = optim.Adam(generator.parameters(), esrgan_facts['p_optimizer_lr'], eval(esrgan_facts['betas']))
d_optimizer = optim.Adam(discriminator.parameters(), esrgan_facts['d_optimizer_lr'], eval(esrgan_facts['betas']))
g_optimizer = optim.Adam(generator.parameters(), esrgan_facts['g_optimizer_lr'],  eval(esrgan_facts['betas']))


# Schedulers
milestones = [esrgan_facts['epochs'] * 0.125, esrgan_facts['epochs'] * 0.250, esrgan_facts['epochs'] * 0.500, esrgan_facts['epochs'] * 0.750]
p_scheduler = CosineAnnealingLR(p_optimizer, esrgan_facts['p_epochs'] // 4, 1e-7)               # Generator model scheduler during generator training.
d_scheduler = MultiStepLR(d_optimizer, list(map(int, milestones)), 0.5)         # Discriminator model scheduler during adversarial training.
g_scheduler = MultiStepLR(g_optimizer, list(map(int, milestones)), 0.5)         # Generator model scheduler during adversarial training.


# Loss Functions
PSNR_CRITERION = nn.MSELoss().to(device)
PIXEL_CRITERION = nn.L1Loss().to(device)
CONTENT_CRITERION = ContentLoss().to(device)
ADVERSARIAL_CRITERION = nn.BCELoss().to(device)

# Loading dataset
eric_dataset = ImageDataset()
eric_dataset.__getitem__(0)
eric_loader = DataLoader(eric_dataset, training_facts['batch_size'], shuffle=True, pin_memory=True)

eric_valid_dataset = ImageDataset(mode="valid")
eric_valid_dataset.__getitem__(0)
eric_valid_loader = DataLoader(eric_valid_dataset, training_facts['batch_size'], shuffle=True, pin_memory=True)

if checkpoint_facts['ESRGAN']['load_esrresnet']:
    checkpoint_file = os.path.join(paths_[1], checkpoint_facts[experiment_type]['checkpoint_esrresnet'])
    load_checkpoint(checkpoint_file, generator, p_optimizer, esrgan_facts['learning_rate'])

if checkpoint_facts['ESRGAN']['load_esrgan']:
    checkpoint_file = os.path.join(paths_[1], checkpoint_facts[experiment_type]['checkpoint_gen'])
    disc_checkpoint_file = os.path.join(paths_[1], checkpoint_facts[experiment_type]['checkpoint_disc'])
    load_checkpoint(
        checkpoint_file,
        generator,
        g_optimizer,
        esrgan_facts['learning_rate'],
    )
    load_checkpoint(
        disc_checkpoint_file, discriminator,d_optimizer,esrgan_facts['learning_rate']
    )

if checkpoint_facts['ESRGAN']['load_p_best']:
    checkpoint_file = os.path.join(paths_[1], checkpoint_facts[experiment_type]['checkpoint_gen'])
    print("==========>Loading the latest model from {}".format(checkpoint_file))
    generator.load_state_dict(torch.load(checkpoint_file))

"""Train"""

def train_generator(gen, optimizer_g, loader, epoch):
    batches = len(loader)
    step = 0
    gen.train()
    for index, (lr, hr) in enumerate(loader):
        lr = lr.to(device)
        hr = hr.to(device)
        gen.zero_grad()
        sr = gen(lr)
        pixel_loss = PIXEL_CRITERION(sr, hr)
        pixel_loss.backward()
        p_optimizer.step()
        iters = index + epoch * batches + 1
        # SAVE_DIR = config.SAVE_WRITER.format("Train_Generator/Loss")
        writer.add_scalar("Generator loss", pixel_loss.item(), iters)
        if index % 100 == 0 and index >= 0:
            img_grid_real = torchvision.utils.make_grid(hr, normalize=True)
            img_grid_fake = torchvision.utils.make_grid(sr, normalize=True)
            writer.add_image("Ground Truth", img_grid_real, global_step=index)
            writer.add_image("Fake Image", img_grid_fake, global_step=index)
            print(f"Train Epoch[{epoch + 1:04d}/{esrgan_facts['start_p_epoch']:04d}]({index + 1:05d}/{batches:05d}) "
                  f"Loss: {pixel_loss.item():.6f}.")

            step += 1
        save_checkpoint(generator, optimizer_g, os.path.join(paths_[1], checkpoint_facts['ESRGAN']['checkpoint_gen']))

def train_adversarial(gen, disc, loader, epoch):
    batches = len(loader)
    disc.train()
    gen.train()
    step = 0
    for index, (lr, hr) in enumerate(loader):
        lr = lr.to(device)
        hr = hr.to(device)
        label_size = lr.size(0)
        real_label = torch.full([label_size, 1], 1.0, dtype=lr.dtype, device=device)
        fake_label = torch.full([label_size, 1], 0.0, dtype=lr.dtype, device=device)

        # Initialize the gradient of the discriminator model.
        disc.zero_grad()
        # Generate super-resolution images.
        sr = gen(lr)
        # Calculate the loss of the discriminator model on the high-resolution image.
        hr_output = disc(hr)
        sr_output = disc(sr.detach())
        clamp_loss = torch.clamp(hr_output - torch.mean(sr_output), min=0, max=1)
        d_loss_hr = ADVERSARIAL_CRITERION(clamp_loss, real_label)
        d_loss_hr.backward()
        d_hr = hr_output.mean().item()
        # Calculate the loss of the discriminator model on the super-resolution image.
        hr_output = disc(hr)
        sr_output = disc(sr.detach())
        clamp_sr_loss = torch.clamp(sr_output - torch.mean(hr_output), min=0, max=1)
        d_loss_sr = ADVERSARIAL_CRITERION(clamp_sr_loss, fake_label)
        d_loss_sr.backward()
        d_sr1 = sr_output.mean().item()
        # Update the weights of the discriminator model.
        d_loss = d_loss_hr + d_loss_sr
        d_optimizer.step()

        # Initialize the gradient of the generator model.
        gen.zero_grad()
        # Generate super-resolution images.
        sr = gen(lr)
        # Calculate the loss of the discriminator model on the super-resolution image.
        hr_output = disc(hr.detach())
        sr_output = disc(sr)
        # Perceptual loss=0.01 * pixel loss + 1.0 * content loss + 0.005 * adversarial loss.
        pixel_loss = esrgan_facts['pixel_weight'] * PIXEL_CRITERION(sr, hr.detach())
        content_loss = esrgan_facts['content_weight'] * CONTENT_CRITERION(sr, hr.detach())
        clamp_adversarial_loss = torch.clamp(sr_output - torch.mean(hr_output), min=0, max=1)
        adversarial_loss = esrgan_facts['lambda_adv'] * ADVERSARIAL_CRITERION(clamp_adversarial_loss, real_label)
        # Update the weights of the generator model.
        g_loss = pixel_loss + content_loss + adversarial_loss
        g_loss.backward()
        g_optimizer.step()
        d_sr2 = sr_output.mean().item()

        # Write the loss during training into Tensorboard.
        iters = index + epoch * batches + 1
        writer.add_scalar("Train_Adversarial/D_Loss", d_loss.item(), iters)
        writer.add_scalar("Train_Adversarial/G_Loss", g_loss.item(), iters)
        writer.add_scalar("Train_Adversarial/D_HR", d_hr, iters)
        writer.add_scalar("Train_Adversarial/D_SR1", d_sr1, iters)
        writer.add_scalar("Train_Adversarial/D_SR2", d_sr2, iters)
        if (index + 1) % 100 == 0 and index > 0:
            img_grid_real = torchvision.utils.make_grid(hr, normalize=True)
            img_grid_fake = torchvision.utils.make_grid(sr, normalize=True)
            writer.add_image("Ground Truth", img_grid_real, global_step=iters)
            writer.add_image("Fake Image", img_grid_fake, global_step=iters)
            print(f"Train stage: adversarial "
                  f"Epoch[{epoch + 1:04d}/{EPOCHS:04d}]({index + 1:05d}/{batches:05d}) "
                  f"D Loss: {d_loss.item():.6f} G Loss: {g_loss.item():.6f} "
                  f"D(HR): {d_hr:.6f} D(SR1)/D(SR2): {d_sr1:.6f}/{d_sr2:.6f}.")

        step += 1
def validate(gen, valid_dataloader, epoch, stage):
    batches = len(valid_dataloader)
    gen.eval()
    total_psnr_value = 0.0
    with torch.no_grad():
        for index, (lr, hr) in enumerate(valid_dataloader):
            lr = lr.to(device)
            hr = hr.to(device)
            sr = gen(lr, out_shape=(1024, 768))
            mse_loss = PSNR_CRITERION(sr, hr)
            psnr_value = 10 * torch.log10(1 / mse_loss).item()
            total_psnr_value += psnr_value
        avg_psnr_value = total_psnr_value / batches
        if stage == "generator":
            writer.add_scalar("Val_Generator/PNSR", avg_psnr_value, epoch + 1)
        elif stage == "adversarial":
            writer.add_scalar("Val_Adversarial/PNSR", avg_psnr_value, epoch + 1)
        print(f"Valid stage: {stage} Epoch[{epoch + 1:04d}] avg PSNR: {avg_psnr_value:.2f}.\n")
    return avg_psnr_value


def train_fn(
        loader,
        disc,
        gen,
        opt_gen,
        opt_disc,
        l1,
        vgg_loss,
        g_scaler,
        d_scaler,
        tb_step,
        epoch
):

    for idx, (low_res, high_res) in enumerate(loader):
        high_res = high_res.to(device)
        low_res = low_res.to(device)
        # fake = gen(low_res)
        with torch.cuda.amp.autocast():
            fake = gen(low_res)
            critic_real = disc(high_res)
            critic_fake = disc(fake.detach())
            gp = gradient_penalty(disc, high_res, fake, device=device)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + esrgan_facts['lambda_gp'] * gp
            )

        opt_disc.zero_grad()
        d_scaler.scale(loss_critic).backward(retain_graph=True)
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with torch.cuda.amp.autocast():
            l1_loss = 1e-2 * l1(fake, high_res)
            adversarial_loss = 5e-3 * -torch.mean(disc(fake))  # Decrease slightly if model is stuck
            loss_for_vgg = vgg_loss(fake, high_res)
            gen_loss = l1_loss + loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward(retain_graph=True)
        g_scaler.step(opt_gen)
        g_scaler.update()
        tb_step += 1
        if idx % 100 == 0 and idx > 0:
            print(f"===>Saving Image At Epoch {epoch} with Batch Number {idx}")
            img_grid_real = torchvision.utils.make_grid(hr, normalize=True)
            img_grid_fake = torchvision.utils.make_grid(sr, normalize=True)
            writer.add_image("Ground Truth", img_grid_real, global_step=tb_step)
            writer.add_image("Fake Image", img_grid_fake, global_step=tb_step)

    return tb_step

""" Training """
for epoch in range(esrgan_facts['start_p_epoch'], esrgan_facts['p_epochs']):
    train_generator(generator, g_optimizer, eric_loader, epoch)
    psnr_value = validate(generator, eric_valid_loader, epoch, "generator")
    is_best = psnr_value > best_psnr_value
    best_psnr_value = max(psnr_value, best_psnr_value)
    if checkpoint_facts['save_checkpoint']:
        checkpoint_file = os.path.join(paths_[1], checkpoint_facts['ESRGAN']['checkpoint_esrresnet'])
        save_checkpoint(generator, p_optimizer, checkpoint_file)
    if is_best:
        checkpoint_file = os.path.join(paths_[1], checkpoint_facts['ESRGAN']['best_checkpoint_gen'])
        torch.save(generator.state_dict(), checkpoint_file)
    p_scheduler.step()

checkpoint_file = os.path.join(paths_[1], checkpoint_facts['ESRGAN']['best_checkpoint_gen'])
disc_checkpoint_file = os.path.join(paths_[1], checkpoint_facts['ESRGAN']['checkpoint_disc'])
torch.save(generator.state_dict(), checkpoint_file)
torch.save(discriminator.state_dict(), disc_checkpoint_file)
